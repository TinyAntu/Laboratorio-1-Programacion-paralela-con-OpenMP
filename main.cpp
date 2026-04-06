#include "Integrator.h"

int main() {
    // Definimos los parámetros del experimento
    int N = 500;
    unsigned int seed = 42;
    double dt = 0.01;
    double G = 1.0;
    double softening = 0.5;
    int total_steps = 1000;
    int save_every = 10;

    Integrator sim(N, seed, dt, G, softening, total_steps, save_every);
    
    sim.runSimulation();

    return 0;
}