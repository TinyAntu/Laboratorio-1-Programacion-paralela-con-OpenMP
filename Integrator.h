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
    int save_every;

    std::vector<std::vector<Particle>> state_history; 
    std::vector<double> kinetic_energy_history;
    std::vector<double> potential_energy_history;

public:
    Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps, int save_freq);
    ~Integrator();

    void runSimulation();

    const std::vector<std::vector<Particle>>& getStateHistory() const { return state_history; }
    const std::vector<double>& getKineticHistory() const { return kinetic_energy_history; }
    const std::vector<double>& getPotentialHistory() const { return potential_energy_history; }
    int getSaveFreq() const { return save_every; }
};

#endif