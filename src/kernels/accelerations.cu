#include "accelerations.cuh"
#include <cmath>

// Variante Basica: Un hilo por cuerpo i, bucle interno j serial sobre memoria global (SoA)
__global__ void computeAccelerationsKernel(int N, 
                                           const double* __restrict__ x, 
                                           const double* __restrict__ y, 
                                           const double* __restrict__ mass, 
                                           double* ax, 
                                           double* ay, 
                                           double eps, 
                                           double G) 
{
    // Calculo de indice global 1D
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Proteccion de bordes
    if (i >= N) return;

    double ax_local = 0.0;
    double ay_local = 0.0;
    double xi = x[i];
    double yi = y[i];

    // Bucle interno sobre todos los j
    for (int j = 0; j < N; ++j) {
        if (i != j) {
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double distSqr = dx * dx + dy * dy + eps * eps;
            double invDist3 = 1.0 / (distSqr * sqrt(distSqr));
            double commonFactor = G * mass[j] * invDist3;
            
            ax_local += commonFactor * dx;
            ay_local += commonFactor * dy;
        }
    }

    ax[i] = ax_local;
    ay[i] = ay_local;
}

// Variante Optimizada: Uso cooperativo de Memoria Compartida (Shared Memory por Tiles)
__global__ void computeAccelerationsKernelShared(int N, 
                                                 const double* __restrict__ x, 
                                                 const double* __restrict__ y, 
                                                 const double* __restrict__ mass, 
                                                 double* ax, 
                                                 double* ay, 
                                                 double eps, 
                                                 double G) 
{
    // Memoria compartida dinamica asignada en el lanzamiento del kernel
    // Requiere espacio para (sh_x, sh_y, sh_mass), cada uno de tamaño blockDim.x
    extern __shared__ double shared_mem[];
    double* sh_x = shared_mem;
    double* sh_y = &sh_x[blockDim.x];
    double* sh_mass = &sh_y[blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    double ax_local = 0.0;
    double ay_local = 0.0;
    double xi = 0.0;
    double yi = 0.0;

    if (i < N) {
        xi = x[i];
        yi = y[i];
    }

    // Iteracion por bloques (tiles) de tamaño blockDim.x
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        int idx_j = tile * blockDim.x + tx;

        // Carga cooperativa a memoria compartida con protección de bordes
        if (idx_j < N) {
            sh_x[tx] = x[idx_j];
            sh_y[tx] = y[idx_j];
            sh_mass[tx] = mass[idx_j];
        } else {
            sh_x[tx] = 0.0;
            sh_y[tx] = 0.0;
            sh_mass[tx] = 0.0;
        }

        // Sincronizar: Asegurar que todo el bloque terminó de cargar datos a shared
        __syncthreads();

        // Calcular interacciones usando los datos del tile actual en Shared Memory
        if (i < N) {
            for (int j = 0; j < blockDim.x; ++j) {
                int current_j = tile * blockDim.x + j;
                if (current_j < N && i != current_j) {
                    double dx = sh_x[j] - xi;
                    double dy = sh_y[j] - yi;
                    double distSqr = dx * dx + dy * dy + eps * eps;
                    double invDist3 = 1.0 / (distSqr * sqrt(distSqr));
                    double commonFactor = G * sh_mass[j] * invDist3;

                    ax_local += commonFactor * dx;
                    ay_local += commonFactor * dy;
                }
            }
        }

        // Sincronizar antes de cargar el siguiente tile
        __syncthreads();
    }

    if (i < N) {
        ax[i] = ax_local;
        ay[i] = ay_local;
    }
}

// Lanzadores Host
void launchAccelerationsBasic(int gridDim, int blockDim, int N, 
                              const double* d_x, const double* d_y, const double* d_mass, 
                              double* d_ax, double* d_ay, double eps, double G) 
{
    computeAccelerationsKernel<<<gridDim, blockDim>>>(N, d_x, d_y, d_mass, d_ax, d_ay, eps, G);
    CUDA_CHECK(cudaGetLastError()); // Capturar errores inmediatamente después del lanzamiento
    CUDA_CHECK(cudaDeviceSynchronize()); // Sincronizar dispositivo con host
}

void launchAccelerationsShared(int gridDim, int blockDim, int N, 
                               const double* d_x, const double* d_y, const double* d_mass, 
                               double* d_ax, double* d_ay, double eps, double G) 
{
    // Tamano en bytes: 3 arrays de doubles por cada hilo en el bloque
    size_t sharedMemSize = 3 * blockDim * sizeof(double);
    computeAccelerationsKernelShared<<<gridDim, blockDim, sharedMemSize>>>(N, d_x, d_y, d_mass, d_ax, d_ay, eps, G);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Sincronizar dispositivo con host
}
