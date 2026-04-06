#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include <vector>

class Integrator {
private:
    NBodySystem* system;
    NBodySimulator* simulator;
    int total_steps;
    std::vector<std::vector<Particle>> state_history; 

public:
    Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps);
    ~Integrator();
    void runSimulation();
    const std::vector<std::vector<Particle>>& getStateHistory() const;
};

#endif