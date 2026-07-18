// Este archivo solo se compila cuando CUDA está habilitado, define el estado
// device (SoA), las transferencias mínimas por paso y los lanzadores host.
#include "NBodySystem.h"
#include "DeviceNBodyState.h"
#include "CudaUtils.h"
#include "accelerations.cuh"

#include <stdexcept>

// Estado GPU opaco declarado en NBodySystem.h: buffers SoA en device.
struct NBodySystem::GpuState {
    DeviceNBodyState device_state;
};

NBodySystem::~NBodySystem() {
    delete gpu_state; // libera los CudaBuffer (RAII) si se usó la GPU
}

void NBodySystem::computeAccelerationsGPU() {
    computeAccelerationsGPU(0, 256); // variante básica, block_size por defecto
}

void NBodySystem::computeAccelerationsGPU(int variant) {
    computeAccelerationsGPU(variant, 256);
}

void NBodySystem::computeAccelerationsGPU(int variant, int block_size) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0) return;

    if (gpu_state == nullptr) {
        gpu_state = new GpuState();
    }
    DeviceNBodyState& dev = gpu_state->device_state;

    // (Re)reserva los arreglos SoA si N cambió. Las masas no cambian durante
    // la simulación: se suben una sola vez (solo tras una (re)reserva).
    if (dev.ensureCapacity(static_cast<std::size_t>(n))) {
        dev.uploadMasses(bodies);
    }

    // H2D por paso: el host actualizó posiciones en el drift del paso anterior
    dev.uploadPositions(bodies);

    // Lanzamiento del kernel: variante básica o con memoria compartida
    const double eps2 = softening_eps * softening_eps;
    switch (variant) {
        case 0:
            launchAccelerationsBasic(dev.mass(), dev.x(), dev.y(),
                                     dev.ax(), dev.ay(), n,
                                     G_const, eps2, block_size);
            break;
        case 1:
            launchAccelerationsShared(dev.mass(), dev.x(), dev.y(),
                                      dev.ax(), dev.ay(), n,
                                      G_const, eps2, block_size);
            break;
        default:
            throw std::invalid_argument(
                "computeAccelerationsGPU: variant invalido (0=basico, 1=shared)");
    }

    // Orden fijo del enunciado: sincronizar device antes de usar los resultados
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H por paso: aceleraciones hacia las Particle para el Euler en host
    dev.downloadAccelerations(bodies);
}
