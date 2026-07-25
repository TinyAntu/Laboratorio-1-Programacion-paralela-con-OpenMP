#ifndef ACCELERATIONS_CUH
#define ACCELERATIONS_CUH

#include "CudaUtils.h"

// Lanzadores desde Host (llamados por NBodySystemGpu.cu)
void launchAccelerationsBasic(int gridDim, int blockDim, int N, 
                              const double* d_x, const double* d_y, const double* d_mass, 
                              double* d_ax, double* d_ay, double eps, double G);

void launchAccelerationsShared(int gridDim, int blockDim, int N, 
                               const double* d_x, const double* d_y, const double* d_mass, 
                               double* d_ax, double* d_ay, double eps, double G);

#endif