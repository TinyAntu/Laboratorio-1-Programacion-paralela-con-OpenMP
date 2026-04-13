#include "Benchmark.h"
#include <iostream>

int main() {
    int N = 1000;
    unsigned int seed = 42;
    double G = 1.0;
    double softening = 0.1;
    double dt = 0.01;
    int repetitions = 10;

    Benchmark bench(N, seed, G, softening, dt, repetitions);

    bench.runAll();

    std::cout << "[PASS] Benchmark ejecutado correctamente\n";
    return 0;
}