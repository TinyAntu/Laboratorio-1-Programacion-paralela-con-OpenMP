#include "Benchmark.h"
#include "Visualizer.h"

#include <iostream>
#include <omp.h>
#include <string>


int main(int argc, char* argv[]) {
    int N = 1000;
    unsigned int seed = 42;
    double G = 1.0;
    double softening = 0.1;
    double dt = 0.01;
    int repetitions = 10;

    int steps = 500;
    int sample_every = 10;
    const bool cuda_benchmarks_only =
        argc > 1 &&
        std::string(argv[1]) == "--cuda-benchmarks-only";

    #ifdef NBODY_HAS_CUDA
    if (cuda_benchmarks_only) {
        Benchmark cuda_benchmark(
            N,
            seed,
            G,
            softening,
            dt,
            repetitions
        );

        cuda_benchmark.runGpuBenchmarks(
            100,
            "blockdim_study.dat"
        );

        return 0;
    }
    #else
    if (cuda_benchmarks_only) {
        std::cerr
            << "Este binario fue compilado sin soporte CUDA."
            << std::endl;

        return 1;
    }
    #endif

    std::cout << "=== INICIANDO BENCHMARKS N-BODY ===" << std::endl;
    std::cout << "Parámetros:" << std::endl;
    std::cout << "  N (partículas): " << N << std::endl;
    std::cout << "  Repeticiones: " << repetitions << std::endl;
    std::cout << "  Seed: " << seed << std::endl;

    std::cout << "OpenMP:" << std::endl;
    std::cout << "  omp_get_max_threads(): " << omp_get_max_threads() << std::endl;
    std::cout << "  omp_get_num_procs(): " << omp_get_num_procs() << std::endl;

    std::cout << std::endl;

    Benchmark bench(N, seed, G, softening, dt, repetitions);
    bench.runAll();

#ifdef NBODY_HAS_CUDA
    bench.runGpuBenchmarks();
#endif

    Visualizer visualizer("snapshots.dat", "energy_timeseries.dat");
    visualizer.runAll(N, seed, G, softening, dt, steps, sample_every);

    std::cout << std::endl << "=== EJECUCIÓN COMPLETADA ===" << std::endl;
    std::cout << "Archivos generados:" << std::endl;
    std::cout << "  - benchmark_results.dat" << std::endl;
    std::cout << "  - scaling_analysis.dat" << std::endl;
    std::cout << "  - snapshots.dat" << std::endl;
    std::cout << "  - energy_timeseries.dat" << std::endl;
    #ifdef NBODY_HAS_CUDA
    std::cout << "  - blockdim_study.dat" << std::endl;
    #endif

    return 0;
}