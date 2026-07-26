#ifndef METRICS_CUH
#define METRICS_CUH

#include <utility>

// Método 0: reducción por bloque usando memoria shared y reducción final.
std::pair<double, double> calculateEnergyReductionGpu(
    int n,
    int block_size,
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    double G,
    double softening
);

// Método 1: cada hilo calcula su contribución y acumula con atomicAdd.
std::pair<double, double> calculateEnergyAtomicGpu(
    int n,
    int block_size,
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    double G,
    double softening
);

#endif