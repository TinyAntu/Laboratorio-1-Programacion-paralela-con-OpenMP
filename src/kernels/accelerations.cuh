#ifndef ACCELERATIONS_CUH
#define ACCELERATIONS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Macro CUDA_CHECK para capturar errores de la API y de ejecución de kernels
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error en " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

// Lanzadores desde Host (llamados por NBodySystem)
void launchAccelerationsBasic(int gridDim, int blockDim, int N, 
                              const double* d_x, const double* d_y, const double* d_mass, 
                              double* d_ax, double* d_ay, double eps, double G);

void launchAccelerationsShared(int gridDim, int blockDim, int N, 
                               const double* d_x, const double* d_y, const double* d_mass, 
                               double* d_ax, double* d_ay, double eps, double G);

#endif