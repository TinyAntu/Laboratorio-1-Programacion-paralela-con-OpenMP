// DeviceNBodyState: estado del sistema N-cuerpos en memoria device con layout SoA
// (structure of arrays) para favorecer accesos coalesced en los kernels.
//
// Esquema de transferencias por paso temporal (Euler se integra en HOST según el
// enunciado, por lo que el mínimo de copias es):
//   - masas:          H2D una sola vez (no cambian durante la simulación)
//   - posiciones x,y: H2D en cada paso (el host las actualiza con drift)
//   - aceleraciones:  D2H en cada paso (las produce el kernel)
//   - velocidades:    nunca tocan el device (solo las usa el host en kick)
//
// Este header solo debe incluirse desde código compilado con nvcc (archivos .cu).
#ifndef DEVICE_NBODY_STATE_H
#define DEVICE_NBODY_STATE_H

#include "CudaBuffer.h"
#include "Particle.h"

#include <cstddef>
#include <vector>

class DeviceNBodyState {
private:
    CudaBuffer<double> d_mass;
    CudaBuffer<double> d_x;
    CudaBuffer<double> d_y;
    CudaBuffer<double> d_ax;
    CudaBuffer<double> d_ay;
    CudaBuffer<double> d_vx;
    CudaBuffer<double> d_vy;
    
    // Buffers host reutilizables para empaquetar AoS (Particle) -> SoA sin
    // realocar en cada paso temporal.
    std::vector<double> h_scratch_a;
    std::vector<double> h_scratch_b;

    std::size_t count_ = 0;

public:
    // Reserva (o re-reserva si N cambió) los cinco arreglos SoA en device.
    // Devuelve true si hubo (re)reserva: las masas deben volver a subirse.
    bool ensureCapacity(std::size_t count) {
        if (count == count_ && count != 0) return false;
        d_mass.allocate(count);
        d_x.allocate(count);
        d_y.allocate(count);
        d_ax.allocate(count);
        d_ay.allocate(count);
        d_vx.allocate(count);
        d_vy.allocate(count);
        h_scratch_a.resize(count);
        h_scratch_b.resize(count);
        count_ = count;
        return true;
    }

    // H2D una sola vez: las masas no cambian durante la simulación.
    void uploadMasses(const std::vector<Particle>& bodies) {
        for (std::size_t i = 0; i < count_; ++i) {
            h_scratch_a[i] = bodies[i].getMass();
        }
        d_mass.copyToDevice(h_scratch_a.data(), count_);
    }

    // H2D por paso: el host actualiza posiciones con Euler (drift).
    void uploadPositions(const std::vector<Particle>& bodies) {
        for (std::size_t i = 0; i < count_; ++i) {
            h_scratch_a[i] = bodies[i].getX();
            h_scratch_b[i] = bodies[i].getY();
        }
        d_x.copyToDevice(h_scratch_a.data(), count_);
        d_y.copyToDevice(h_scratch_b.data(), count_);
    }

    // D2H por paso: aceleraciones calculadas por el kernel -> Particle (SoA -> AoS).
    void downloadAccelerations(std::vector<Particle>& bodies) {
        d_ax.copyToHost(h_scratch_a.data(), count_);
        d_ay.copyToHost(h_scratch_b.data(), count_);
        for (std::size_t i = 0; i < count_; ++i) {
            bodies[i].setAcceleration(h_scratch_a[i], h_scratch_b[i]);
        }
    }

    void uploadVelocities(const std::vector<Particle>& bodies) {
        for (std::size_t i = 0; i < count_; ++i) {
            h_scratch_a[i] = bodies[i].getVx();
            h_scratch_b[i] = bodies[i].getVy();
        }

        d_vx.copyToDevice(h_scratch_a.data(), count_);
        d_vy.copyToDevice(h_scratch_b.data(), count_);
    }

    // Punteros device crudos para pasar a los kernels (Rol 1)
    const double* mass() const { return d_mass.data(); }
    const double* x() const { return d_x.data(); }
    const double* y() const { return d_y.data(); }
    double* ax() { return d_ax.data(); }
    double* ay() { return d_ay.data(); }
    const double* vx() const { return d_vx.data(); }
    const double* vy() const { return d_vy.data(); }

    std::size_t size() const { return count_; }
};

#endif // DEVICE_NBODY_STATE_H
