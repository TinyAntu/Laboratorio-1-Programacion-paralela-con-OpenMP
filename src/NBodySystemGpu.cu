// Implementación GPU de NBodySystem (Lab 2) — Rol 2: host/device y memoria.
// Este archivo solo se compila cuando CUDA está habilitado, define el estado
// device (SoA), las transferencias mínimas por paso y conecta los buffers con
// los kernels del Rol 1 (src/kernels/accelerations.cu).
#include "NBodySystem.h"
#include "DeviceNBodyState.h"
#include "CudaUtils.h"
#include "kernels/accelerations.cuh"

#include <stdexcept>
#include <string>

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

    // Orden fijo del enunciado: sincronizar device antes de usar los resultados
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H por paso: aceleraciones hacia las Particle para el Euler en host
    dev.downloadAccelerations(bodies);
}
