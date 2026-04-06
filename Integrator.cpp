#include "Integrator.h"
#include <iostream>

Integrator::Integrator(int N, unsigned int seed, double dt, double G, double softening, int steps, int save_freq) 
    : total_steps(steps), save_every(save_freq) 
{
    // Instanciamos los componentes (Memoria dinámica para que vivan en el objeto)
    system = new NBodySystem(G, softening);
    system->loadFromSeed(seed, N);

    simulator = new NBodySimulator(system, dt);
    metrics = new MetricsCalculator(system);
    visualizer = new Visualizer("trayectorias.dat", "energia.dat");

    // Limpiar archivos previos
    visualizer->clearFiles();
}

Integrator::~Integrator() {
    // Liberamos la memoria en el orden inverso a la creación
    delete simulator;
    delete metrics;
    delete visualizer;
    delete system;
}

void Integrator::runSimulation() {
    std::cout << "Simulando sistema N-Body..." << std::endl;

    for (int step = 0; step <= total_steps; ++step) {
        simulator->integrateEuler();
        if (step % save_every == 0) {
            // Guardar estado actual de las partículas
            visualizer->saveState(step, system->getBodies());

            // Calcular y guardar energías
            double K = metrics->calculateKineticEnergy();
            double U = metrics->calculatePotentialEnergy();
            visualizer->saveEnergy(step, K, U);
        }
    }

    std::cout << "\n¡Simulación completada! Datos listos en Visualizer." << std::endl;
}