#include "Benchmark.h"
#include <iostream>

int main() {
    int N = 1000;
    unsigned int seed = 42;
    double G = 1.0;
    double softening = 0.1;
    double dt = 0.01;
    int repetitions = 10;

    std::cout << "=== INICIANDO BENCHMARKS N-BODY ===" << std::endl;
    std::cout << "Parámetros:" << std::endl;
    std::cout << "  N (partículas): " << N << std::endl;
    std::cout << "  Repeticiones: " << repetitions << std::endl;
    std::cout << "  Seed: " << seed << std::endl;
    std::cout << std::endl;
    
    Benchmark bench(N, seed, G, softening, dt, repetitions);
    bench.runAll();
    
    std::cout << std::endl << "=== BENCHMARKS COMPLETADOS ===" << std::endl;
    std::cout << "Archivos generados:" << std::endl;
    std::cout << "  - benchmark_schedule_results.txt" << std::endl;
    std::cout << "  - benchmark_sync_results.txt" << std::endl;
    std::cout << "  - benchmark_data_clause_results.txt" << std::endl;
    std::cout << "  - benchmark_adv_sync_results.txt" << std::endl;
    std::cout << "  - scaling_analysis.txt" << std::endl;
    
    return 0;
}
