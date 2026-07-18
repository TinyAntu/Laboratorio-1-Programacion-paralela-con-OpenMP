// Kernels de aceleraciones N-cuerpos 2D (gravedad todo-pares con suavizado Plummer).
//
// Convención del enunciado (sección 4.1):
//   - Variante básica: un hilo CUDA por cuerpo i; cada hilo escribe solo a_i y
//     recorre en serial todo j != i. Mapeo 1D: i = blockIdx.x*blockDim.x + threadIdx.x.
//   - Grilla con división techo: ceil(N / blockDim.x); protección de borde i >= N.
//
// NOTA Rol 1: este kernel básico es una versión provisional funcional escrita por
// el Rol 2 para validar la capa de memoria end-to-end. El Rol 1 es dueño de este
// archivo: puede refinarlo y debe implementar launchAccelerationsShared.
#include "accelerations.cuh"
#include "CudaUtils.h"

#include <stdexcept>
#include <string>

namespace {

__global__ void accelerationsKernelBasic(const double* mass,
                                         const double* x,
                                         const double* y,
                                         double* ax,
                                         double* ay,
                                         int n,
                                         double G,
                                         double eps2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return; // protección de borde: la grilla puede exceder N

    const double xi = x[i];
    const double yi = y[i];
    double axi = 0.0;
    double ayi = 0.0;

    // Misma física que la referencia CPU serial (NBodySystem::computeAccelerations)
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        const double dx = x[j] - xi;
        const double dy = y[j] - yi;
        const double distSqr = dx * dx + dy * dy + eps2;
        const double invDist3 = 1.0 / (distSqr * sqrt(distSqr));
        const double commonFactor = G * mass[j] * invDist3;
        axi += commonFactor * dx;
        ayi += commonFactor * dy;
    }

    ax[i] = axi;
    ay[i] = ayi;
}

void validateLaunchParams(int n, int block_size) {
    if (n <= 0) {
        throw std::invalid_argument("launchAccelerations: n debe ser > 0");
    }
    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument(
            "launchAccelerations: block_size invalido (" +
            std::to_string(block_size) + "); use 1..1024");
    }
}

} // namespace

void launchAccelerationsBasic(const double* d_mass,
                              const double* d_x,
                              const double* d_y,
                              double* d_ax,
                              double* d_ay,
                              int n,
                              double G,
                              double eps2,
                              int block_size) {
    validateLaunchParams(n, block_size);

    // División techo: la última bloque puede quedar parcialmente ocupada
    const int grid_size = (n + block_size - 1) / block_size;

    accelerationsKernelBasic<<<grid_size, block_size>>>(
        d_mass, d_x, d_y, d_ax, d_ay, n, G, eps2);
    CUDA_CHECK_KERNEL();
}

void launchAccelerationsShared(const double* /*d_mass*/,
                               const double* /*d_x*/,
                               const double* /*d_y*/,
                               double* /*d_ax*/,
                               double* /*d_ay*/,
                               int /*n*/,
                               double /*G*/,
                               double /*eps2*/,
                               int /*block_size*/) {
    throw std::runtime_error(
        "Variante shared memory pendiente de implementacion (Rol 1: Kernels CUDA)");
}
