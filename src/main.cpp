#include "Benchmark.h"
#include "Visualizer.h"

#include <iostream>
#include <omp.h>


int main() {
    int N = 1000;
    unsigned int seed = 42;
    double G = 1.0;
    double softening = 0.1;
    double dt = 0.01;
    int repetitions = 10;

    int steps = 500;
    int sample_every = 10;

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

    return 0;
}