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

    // Construcción del integrador
    Integrator sim(N, seed, dt, G, softening, total_steps);

    // RELOJ
    std::cout << "Iniciando simulacion N-Body (" << N << " particulas)..." << std::endl;

    sim.runSimulation();
    std::cout << "Simulacion completada." << std::endl;

    // ESTO HAY QUE CAMBIAR, Iniciamos el visualizador para obtener graficas de trayectorias y energía
    Visualizer vis("trayectorias.dat", "energia.dat");
    vis.clearFiles();
    const auto& states = sim.getStateHistory();                     // Obtenemos la referencia a la historia de estados del integrador

    //Creamos el archivo de trayectorias
    for (size_t step = 0; step < states.size(); ++step) {
        vis.saveState(step, states[step]);
    }

    // Automaticamente se liberan los recursos al salir del main, se invocan los destructores de Integrator y Visualizer
    return 0;
}