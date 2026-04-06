#include "Integrator.h"

Integrator::Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps, int save_freq) 
    : total_steps(steps), save_every(save_freq) 
{
    system = new NBodySystem(G, softening);
    system->loadFromSeed(seed, N);

    simulator = new NBodySimulator(system, dt);

    // Reserva de memoria para evitar reasignaciones
    int num_saves = (total_steps / save_every) + 1;
    state_history.reserve(num_saves);
    kinetic_energy_history.reserve(num_saves);
    potential_energy_history.reserve(num_saves);
}

Integrator::~Integrator() {
    delete simulator;
    delete system;
}

void Integrator::runSimulation() {
    for (int step = 0; step <= total_steps; ++step) {
        
        if (step % save_every == 0) {
            state_history.push_back(system->getBodies());
            // Creamos una variable temporal para almacenar la energía cinética y potencial calculada
            auto [ke, pe] = simulator->calculateEnergy();
            kinetic_energy_history.push_back(ke);
            potential_energy_history.push_back(pe);
        }

        simulator->integrateEuler();
    }
}