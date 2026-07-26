#include "metrics.cuh"

#include "CudaBuffer.h"
#include "CudaUtils.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// Comprueba que el tamaño del bloque sea apropiado para la reducción binaria.
void validateBlockSize(int block_size) {
    if (block_size <= 0 || block_size > 1024) {
        throw std::invalid_argument(
            "calculateEnergyGpu: block_size debe estar entre 1 y 1024"
        );
    }

    // La reducción implementada usa divisiones sucesivas por 2.
    if ((block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument(
            "calculateEnergyGpu: block_size debe ser potencia de 2"
        );
    }
}

/*
 * Cada hilo representa un cuerpo i.
 *
 * K_i = 1/2 m_i (vx_i² + vy_i²)
 *
 * Para U se recorren únicamente j > i. De esta manera cada pareja
 * gravitatoria aparece exactamente una vez, igual que en la referencia CPU.
 */
__global__ void calculateEnergyPartialsKernel(
    int n,
    const double* __restrict__ mass,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double G,
    double softening,
    double* partial_kinetic,
    double* partial_potential
) {
    extern __shared__ double shared_memory[];

    double* shared_kinetic = shared_memory;
    double* shared_potential = shared_memory + blockDim.x;

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;

    double kinetic_local = 0.0;
    double potential_local = 0.0;

    if (i < n) {
        const double mi = mass[i];
        const double vxi = vx[i];
        const double vyi = vy[i];

        kinetic_local =
            0.5 * mi * (vxi * vxi + vyi * vyi);

        const double xi = x[i];
        const double yi = y[i];
        const double eps2 = softening * softening;

        // j > i evita contar dos veces las parejas (i,j) y (j,i).
        for (int j = i + 1; j < n; ++j) {
            const double dx = x[j] - xi;
            const double dy = y[j] - yi;

            const double distance =
                sqrt(dx * dx + dy * dy + eps2);

            potential_local -=
                G * mi * mass[j] / distance;
        }
    }

    shared_kinetic[tid] = kinetic_local;
    shared_potential[tid] = potential_local;

    __syncthreads();

    // Reducción dentro del bloque.
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < static_cast<int>(stride)) {
            shared_kinetic[tid] +=
                shared_kinetic[tid + stride];

            shared_potential[tid] +=
                shared_potential[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial_kinetic[blockIdx.x] = shared_kinetic[0];
        partial_potential[blockIdx.x] = shared_potential[0];
    }
}

/*
 * Segunda pasada de reducción.
 *
 * Recibe los resultados parciales de todos los bloques anteriores y genera
 * un único K y un único U.
 */
__global__ void reduceEnergyPartialsKernel(
    int partial_count,
    const double* partial_kinetic,
    const double* partial_potential,
    double* total_kinetic,
    double* total_potential
) {
    extern __shared__ double shared_memory[];

    double* shared_kinetic = shared_memory;
    double* shared_potential = shared_memory + blockDim.x;

    const int tid = threadIdx.x;

    double kinetic_local = 0.0;
    double potential_local = 0.0;

    // Permite reducir cualquier cantidad de resultados parciales.
    for (int i = tid; i < partial_count; i += blockDim.x) {
        kinetic_local += partial_kinetic[i];
        potential_local += partial_potential[i];
    }

    shared_kinetic[tid] = kinetic_local;
    shared_potential[tid] = potential_local;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < static_cast<int>(stride)) {
            shared_kinetic[tid] +=
                shared_kinetic[tid + stride];

            shared_potential[tid] +=
                shared_potential[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        total_kinetic[0] = shared_kinetic[0];
        total_potential[0] = shared_potential[0];
    }
}

/*
 * Variante atomicAdd.
 *
 * Cada hilo calcula sus contribuciones locales completas y realiza solamente
 * dos operaciones atómicas: una para K y otra para U.
 */
__global__ void calculateEnergyAtomicKernel(
    int n,
    const double* __restrict__ mass,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    double G,
    double softening,
    double* total_kinetic,
    double* total_potential
) {
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const double mi = mass[i];
    const double vxi = vx[i];
    const double vyi = vy[i];

    const double kinetic_local =
        0.5 * mi * (vxi * vxi + vyi * vyi);

    const double xi = x[i];
    const double yi = y[i];
    const double eps2 = softening * softening;

    double potential_local = 0.0;

    for (int j = i + 1; j < n; ++j) {
        const double dx = x[j] - xi;
        const double dy = y[j] - yi;

        const double distance =
            sqrt(dx * dx + dy * dy + eps2);

        potential_local -=
            G * mi * mass[j] / distance;
    }

    atomicAdd(total_kinetic, kinetic_local);
    atomicAdd(total_potential, potential_local);
}

} // namespace

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
) {
    if (n <= 0) {
        return {0.0, 0.0};
    }

    validateBlockSize(block_size);

    const int grid_size =
        (n + block_size - 1) / block_size;

    CudaBuffer<double> partial_kinetic(
        static_cast<std::size_t>(grid_size)
    );

    CudaBuffer<double> partial_potential(
        static_cast<std::size_t>(grid_size)
    );

    CudaBuffer<double> total_kinetic(1);
    CudaBuffer<double> total_potential(1);

    const std::size_t shared_bytes =
        2ULL * block_size * sizeof(double);

    calculateEnergyPartialsKernel<<<
        grid_size,
        block_size,
        shared_bytes
    >>>(
        n,
        d_mass,
        d_x,
        d_y,
        d_vx,
        d_vy,
        G,
        softening,
        partial_kinetic.data(),
        partial_potential.data()
    );

    CUDA_CHECK_KERNEL();

    reduceEnergyPartialsKernel<<<
        1,
        block_size,
        shared_bytes
    >>>(
        grid_size,
        partial_kinetic.data(),
        partial_potential.data(),
        total_kinetic.data(),
        total_potential.data()
    );

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double kinetic = 0.0;
    double potential = 0.0;

    total_kinetic.copyToHost(&kinetic, 1);
    total_potential.copyToHost(&potential, 1);

    return {kinetic, potential};
}

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
) {
    if (n <= 0) {
        return {0.0, 0.0};
    }

    validateBlockSize(block_size);

    const int grid_size =
        (n + block_size - 1) / block_size;

    CudaBuffer<double> total_kinetic(1);
    CudaBuffer<double> total_potential(1);

    CUDA_CHECK(cudaMemset(
        total_kinetic.data(),
        0,
        sizeof(double)
    ));

    CUDA_CHECK(cudaMemset(
        total_potential.data(),
        0,
        sizeof(double)
    ));

    calculateEnergyAtomicKernel<<<
        grid_size,
        block_size
    >>>(
        n,
        d_mass,
        d_x,
        d_y,
        d_vx,
        d_vy,
        G,
        softening,
        total_kinetic.data(),
        total_potential.data()
    );

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    double kinetic = 0.0;
    double potential = 0.0;

    total_kinetic.copyToHost(&kinetic, 1);
    total_potential.copyToHost(&potential, 1);

    return {kinetic, potential};
}