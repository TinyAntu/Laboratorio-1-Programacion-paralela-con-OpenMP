#include "Integrator.h"

Integrator::Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps) 
    : total_steps(steps) {  
    
    system = new NBodySystem(G, softening);
    system->loadFromSeed(seed, N);
    simulator = new NBodySimulator(system, dt);
    // Reserva de memoria para evitar reasignaciones y mejorar el rendimiento
    state_history.reserve(total_steps  + 1);
}

Integrator::~Integrator() {
    delete simulator;
    delete system;
}

void Integrator::runSimulation() {
    state_history.push_back(system->getBodies()); // almacenamos el estado inicial antes de comenzar la integración

    for (int step = 0; step <= total_steps; ++step) {
        
        // hacemos la integración temporal
        simulator->integrateEuler();

        // almacenamos el estado luego de la integración de este paso
        state_history.push_back(system->getBodies());
    }
}

void Integrator::runSimulationSchedule() {
    
    state_history.push_back(system->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante por schedule
        simulator->integrateEulerSchedule();

        state_history.push_back(system->getBodies());
    }
}

void Integrator::runSimulationChunk() {
    
    state_history.push_back(system->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante con chunk
        simulator->integrateEulerChunk();

        state_history.push_back(system->getBodies());
    }
}

void Integrator::runSimulationCollapse() {
    
    state_history.push_back(system->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante collapse
        simulator->integrateEulerCollapse();

        state_history.push_back(system->getBodies());
    }
}

// Se retorna una referencia constante para evitar copias innecesarias y garantizar que el estado no se modifique desde fuera de la clase
const std::vector<std::vector<Particle>>& Integrator::getStateHistory() const {
    return state_history;
}

