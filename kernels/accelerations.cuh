// Interfaz de los kernels de aceleraciones (dueño: Rol 1 - Kernels CUDA).
// El Rol 2 (host/device y memoria) define aquí el contrato de datos SoA que
// consumen los kernels; el Rol 1 implementa/optimiza las variantes en
// accelerations.cu (básica y shared memory).
#ifndef ACCELERATIONS_CUH
#define ACCELERATIONS_CUH

// Lanzador host de la variante BÁSICA (un hilo por cuerpo i, bucle serial en j).
// Todos los punteros son device (layout SoA); eps2 = epsilon^2 ya elevado.
// Lanza la grilla con división techo ceil(n / block_size) y verifica errores
// con CUDA_CHECK_KERNEL. La sincronización queda a cargo del llamador.
void launchAccelerationsBasic(const double* d_mass,
                              const double* d_x,
                              const double* d_y,
                              double* d_ax,
                              double* d_ay,
                              int n,
                              double G,
                              double eps2,
                              int block_size);

// Lanzador de la variante con MEMORIA COMPARTIDA (tiles + __syncthreads()).
// PENDIENTE: implementación del Rol 1. Debe producir el mismo resultado físico
// que la variante básica dentro de la tolerancia acordada (rtol=1e-4, atol=1e-8).
void launchAccelerationsShared(const double* d_mass,
                               const double* d_x,
                               const double* d_y,
                               double* d_ax,
                               double* d_ay,
                               int n,
                               double G,
                               double eps2,
                               int block_size);

#endif // ACCELERATIONS_CUH
