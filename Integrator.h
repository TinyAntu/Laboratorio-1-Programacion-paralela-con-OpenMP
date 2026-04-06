#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include "MetricsCalculator.h"
#include "Visualizer.h"
#include <vector>

class Integrator {
private:
    // Punteros a los componentes del sistema
    NBodySystem* system;
    NBodySimulator* simulator;
    MetricsCalculator* metrics;
    Visualizer* visualizer;

    // Parámetros de control
    int total_steps;
    int save_every;

public:
    Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps, int save_freq);
    
    ~Integrator();

    void runSimulation();
};

#endif