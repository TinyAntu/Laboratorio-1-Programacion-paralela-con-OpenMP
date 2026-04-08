#include "Integrator.h"

Integrator::Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps) 
    : total_steps(steps) {  
    simulator = new NBodySimulator(N, seed, G, softening, dt);
    // Reserva de memoria para evitar reasignaciones y mejorar el rendimiento
    state_history.reserve(total_steps  + 1);
    energy_history.reserve(total_steps + 1);
}

Integrator::~Integrator() {
    delete simulator;
}

void Integrator::runSimulation() {
    state_history.push_back(simulator->getBodies()); // almacenamos el estado inicial antes de comenzar la integración

    for (int step = 0; step <= total_steps; ++step) {
        
        // hacemos la integración temporal
        simulator->integrateEuler();

        // almacenamos el estado luego de la integración de este paso
        state_history.push_back(simulator->getBodies());
        energy_history.push_back(simulator->getSystemEnergy());
    }
}

void Integrator::runSimulationSchedule() {
    
    state_history.push_back(simulator->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante por schedule
        simulator->integrateEulerSchedule();

        state_history.push_back(simulator->getBodies());
        energy_history.push_back(simulator->getSystemEnergy()); 
    }
}

void Integrator::runSimulationChunk() {
    
    state_history.push_back(simulator->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante con chunk
        simulator->integrateEulerChunk();

        state_history.push_back(simulator->getBodies());
        energy_history.push_back(simulator->getSystemEnergy()); 
    }
}

void Integrator::runSimulationCollapse() {
    
    state_history.push_back(simulator->getBodies());

    for (int step = 0; step <= total_steps; ++step) {
        // Integración usando la variante collapse
        simulator->integrateEulerCollapse();

        state_history.push_back(simulator->getBodies());
        energy_history.push_back(simulator->getSystemEnergy());
    }
}

const NBodySimulator* Integrator::getSimulator() const {
    return simulator;
}

// Se retorna una referencia constante para evitar copias innecesarias y garantizar que el estado no se modifique desde fuera de la clase
const std::vector<std::vector<Particle>>& Integrator::getStateHistory() const {
    return state_history;
}
const std::vector<std::pair<double, double>>& Integrator::getEnergyHistory() const {
    return energy_history;
}


