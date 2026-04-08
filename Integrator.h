#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include <vector>

class Integrator {
private:
    NBodySimulator* simulator;
    int total_steps;
    std::vector<std::vector<Particle>> state_history;
    std::vector<std::pair<double, double>> energy_history; 

public:
    Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps);
    ~Integrator();
    void runSimulation();
    void runSimulationSchedule();
    void runSimulationChunk();
    void runSimulationCollapse();

    const std::vector<std::vector<Particle>>& getStateHistory() const;
    const NBodySimulator* getSimulator() const;
    const std::vector<std::pair<double, double>>& getEnergyHistory() const;

};

#endif