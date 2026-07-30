// Implementación GPU de NBodySystem (Lab 2) — Rol 2: host/device y memoria.
// Este archivo solo se compila cuando CUDA está habilitado, define el estado
// device (SoA), las transferencias mínimas por paso y conecta los buffers con
// los kernels del Rol 1 (src/kernels/accelerations.cu).
#include "NBodySystem.h"
#include "kernels/metrics.cuh"
#include "DeviceNBodyState.h"
#include "CudaUtils.h"
#include "kernels/accelerations.cuh"

#include <stdexcept>
#include <string>
#include <chrono>

// Estado GPU opaco declarado en NBodySystem.h: buffers SoA en device.
struct NBodySystem::GpuState {
    DeviceNBodyState device_state;
};

NBodySystem::~NBodySystem() {
    delete gpu_state; // libera los CudaBuffer (RAII) si se usó la GPU
}

void NBodySystem::computeAccelerationsGpu() {
    computeAccelerationsGpu(0, 256); // variante básica, block_size por defecto
}

void NBodySystem::computeAccelerationsGpu(int variant) {
    computeAccelerationsGpu(variant, 256);
}

void NBodySystem::computeAccelerationsGpu(int variant, int block_size) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0) return;

    // Los lanzadores del Rol 1 no validan block_size: se valida aquí
    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument(
            "computeAccelerationsGpu: block_size invalido (" +
            std::to_string(block_size) + "); use 1..1024");
    }

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

    // Grilla con división techo (convención del enunciado)
    const int grid_size = (n + block_size - 1) / block_size;

    // Los kernels del Rol 1 reciben eps sin elevar (calculan eps*eps adentro)
    switch (variant) {
        case 0:
            launchAccelerationsBasic(grid_size, block_size, n,
                                     dev.x(), dev.y(), dev.mass(),
                                     dev.ax(), dev.ay(),
                                     softening_eps, G_const);
            break;
        case 1:
            launchAccelerationsShared(grid_size, block_size, n,
                                      dev.x(), dev.y(), dev.mass(),
                                      dev.ax(), dev.ay(),
                                      softening_eps, G_const);
            break;
        default:
            throw std::invalid_argument(
                "computeAccelerationsGpu: variant invalido (0=basico, 1=shared)");
    }
    CUDA_CHECK(cudaGetLastError());

    // Orden fijo del enunciado: sincronizar device antes de usar los resultados
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H por paso: aceleraciones hacia las Particle para el Euler en host
    dev.downloadAccelerations(bodies);
}

std::pair<double, double> NBodySystem::computeEnergyGpu(
    int method,
    int block_size
) {
    if (method != 0 && method != 1) {
        throw std::invalid_argument(
            "computeEnergyGpu: method invalido "
            "(0=reduccion shared, 1=atomicAdd)"
        );
    }

    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument(
            "computeEnergyGpu: block_size invalido"
        );
    }

    if ((block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument(
            "computeEnergyGpu: block_size debe ser potencia de 2"
        );
    }

    const int n = static_cast<int>(bodies.size());

    if (n == 0) {
        return {0.0, 0.0};
    }

    if (gpu_state == nullptr) {
        gpu_state = new GpuState();
    }

    DeviceNBodyState& dev =
        gpu_state->device_state;

    // Cuando N cambia, también deben subirse nuevamente las masas.
    if (dev.ensureCapacity(static_cast<std::size_t>(n))) {
        dev.uploadMasses(bodies);
    }

    /*
     * Euler se ejecutó en host, por lo que posiciones y velocidades deben
     * actualizarse antes de calcular K y U en device.
     */
    dev.uploadPositions(bodies);
    dev.uploadVelocities(bodies);

    switch (method) {
        case 0:
            return calculateEnergyReductionGpu(
                n,
                block_size,
                dev.mass(),
                dev.x(),
                dev.y(),
                dev.vx(),
                dev.vy(),
                G_const,
                softening_eps
            );

        case 1:
            return calculateEnergyAtomicGpu(
                n,
                block_size,
                dev.mass(),
                dev.x(),
                dev.y(),
                dev.vx(),
                dev.vy(),
                G_const,
                softening_eps
            );

        default:
            throw std::invalid_argument(
                "computeEnergyGpu: metodo invalido"
            );
    }
}

double NBodySystem::computeAccelerationsGpuKernelOnly(int variant, int block_size) {
    const int n = static_cast<int>(bodies.size());
    if (n == 0) return 0.0;

    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument("block_size invalido");
    }

    if (gpu_state == nullptr) {
        gpu_state = new GpuState();
    }
    DeviceNBodyState& dev = gpu_state->device_state;

    if (dev.ensureCapacity(static_cast<std::size_t>(n))) {
        dev.uploadMasses(bodies);
    }
    dev.uploadPositions(bodies);

    const int grid_size = (n + block_size - 1) / block_size;

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    switch (variant) {
        case 0:
            launchAccelerationsBasic(grid_size, block_size, n,
                                     dev.x(), dev.y(), dev.mass(),
                                     dev.ax(), dev.ay(),
                                     softening_eps, G_const);
            break;
        case 1:
            launchAccelerationsShared(grid_size, block_size, n,
                                      dev.x(), dev.y(), dev.mass(),
                                      dev.ax(), dev.ay(),
                                      softening_eps, G_const);
            break;
        default:
            throw std::invalid_argument("variant invalido (0=basico, 1=shared)");
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    return elapsed.count();
}

double NBodySystem::computeAccelerationsGpuEndToEnd(
    int variant,
    int block_size,
    int steps,
    double dt
) {
    const int n = static_cast<int>(bodies.size());

    if (n == 0) {
        return 0.0;
    }

    if (variant != 0 && variant != 1) {
        throw std::invalid_argument(
            "variant invalido (0=basico, 1=shared)"
        );
    }

    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument(
            "block_size invalido"
        );
    }

    if (steps < 1) {
        throw std::invalid_argument(
            "steps debe ser >= 1"
        );
    }

    if (dt <= 0.0) {
        throw std::invalid_argument(
            "dt debe ser > 0"
        );
    }

    if (gpu_state == nullptr) {
        gpu_state = new GpuState();
    }

    DeviceNBodyState& dev =
        gpu_state->device_state;

    /*
     * La reserva y la copia de masas quedan fuera del cronómetro.
     *
     * Las masas no cambian durante la simulación y no deben copiarse
     * nuevamente en cada paso.
     */
    if (dev.ensureCapacity(static_cast<std::size_t>(n))) {
        dev.uploadMasses(bodies);
    }

    const int grid_size =
        (n + block_size - 1) / block_size;

    /*
     * Se sincroniza antes de iniciar el cronómetro para evitar que trabajo
     * CUDA pendiente de una operación anterior contamine la medición.
     *
     * No se lanza un kernel de calentamiento aquí porque las posiciones aún
     * no han sido copiadas al device para esta medición.
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto t0 =
        std::chrono::steady_clock::now();

    for (int step = 0; step < steps; ++step) {
        /*
         * End-to-end de un paso:
         *
         * 1. H2D de posiciones.
         * 2. Kernel.
         * 3. Sincronización.
         * 4. D2H de aceleraciones.
         * 5. Kick y drift en host.
         */

        dev.uploadPositions(bodies);

        switch (variant) {
            case 0:
                launchAccelerationsBasic(
                    grid_size,
                    block_size,
                    n,
                    dev.x(),
                    dev.y(),
                    dev.mass(),
                    dev.ax(),
                    dev.ay(),
                    softening_eps,
                    G_const
                );
                break;

            case 1:
                launchAccelerationsShared(
                    grid_size,
                    block_size,
                    n,
                    dev.x(),
                    dev.y(),
                    dev.mass(),
                    dev.ax(),
                    dev.ay(),
                    softening_eps,
                    G_const
                );
                break;
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dev.downloadAccelerations(bodies);

        for (auto& body : bodies) {
            body.kick(dt);
            body.drift(dt);
        }
    }

    const auto t1 =
        std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed =
        t1 - t0;

    /*
     * Se devuelve el tiempo promedio de un paso.
     *
     * Por ejemplo, si 100 pasos tardaron 0.2 segundos,
     * se devuelve 0.002 segundos por paso.
     */
    return elapsed.count() /
           static_cast<double>(steps);
}

