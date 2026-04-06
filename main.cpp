#include "Integrator.h"
#include "Visualizer.h"
#include <iostream>
#include <chrono> // para medir el tiempo del OpenMP

int main() {
    // Parámetros
    int N = 500;
    unsigned int seed = 42;
    double dt = 0.01;
    double G = 1.0;
    double softening = 0.5;
    int total_steps = 1000;
    int save_every = 10;

    // Construcción del integrador
    Integrator sim(N, seed, dt, G, softening, total_steps, save_every);

    // RELOJ
    std::cout << "Iniciando simulacion N-Body (" << N << " particulas)..." << std::endl;
    sim.runSimulation();
    std::cout << "Simulacion completada." << std::endl;

    std::cout << "Guardando datos en disco..." << std::endl;
    Visualizer vis("trayectorias.dat", "energia.dat");
    vis.clearFiles();

    // Extraemos los datos calculados de la simulación
    const auto& states = sim.getStateHistory();
    const auto& ke_hist = sim.getKineticHistory();
    const auto& pe_hist = sim.getPotentialHistory();
    int freq = sim.getSaveFreq();

    // Recorremos los datos y los volcamos a los archivos
    for (size_t i = 0; i < states.size(); ++i) {
        int actual_step = i * freq;
        vis.saveState(actual_step, states[i]);
        vis.saveEnergy(actual_step, ke_hist[i], pe_hist[i]);
    }

    std::cout << "Datos guardados exitosamente. Fin del programa." << std::endl;

    return 0;
}