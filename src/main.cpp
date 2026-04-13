#include "Integrator.h"
#include "Visualizer.h"
#include "MetricsCalculator.h"
#include <iostream>
#include <chrono> // para medir el tiempo del OpenMP
#include <omp.h>

int main() {
    // Parámetros
    int N = 500;
    unsigned int seed = 42;
    double dt = 0.01;
    double G = 1.0;
    double softening = 0.5;
    int total_steps = 1000;

    // -------------------------------
    // VERSION ORIGINAL / BASE
    // -------------------------------
    // Construcción del integrador
    Integrator sim(N, seed, dt, G, softening, total_steps);

    std::cout << "Iniciando simulacion N-Body (" << N << " particulas)..." << std::endl;

    double start_base = omp_get_wtime();
    sim.runSimulation();
    double end_base = omp_get_wtime();

    std::cout << "Simulacion completada." << std::endl;
    std::cout << "Tiempo runSimulation(): " << (end_base - start_base) << " segundos" << std::endl;

    Visualizer vis("trayectorias_base.dat", "energia_base.dat");
    vis.clearFiles();
    const auto& states = sim.getStateHistory(); // Obtenemos la referencia a la historia de estados del integrador

    //Creamos el archivo de trayectorias
    for (size_t step = 0; step < states.size(); ++step) {
        vis.saveState(step, states[step]);
        vis.saveEnergy(step, sim.getEnergyHistory()[step].first, sim.getEnergyHistory()[step].second); // Guardamos la energía cinética y potencial del sistema en cada paso
    }

    // -------------------------------
    // VERSION SCHEDULE
    // -------------------------------
    Integrator simSchedule(N, seed, dt, G, softening, total_steps);

    std::cout << "\nIniciando simulacion runSimulationSchedule()..." << std::endl;

    double start_schedule = omp_get_wtime();
    simSchedule.runSimulationSchedule();
    double end_schedule = omp_get_wtime();

    std::cout << "Simulacion Schedule completada." << std::endl;
    std::cout << "Tiempo runSimulationSchedule(): " 
              << (end_schedule - start_schedule) << " segundos" << std::endl;

    Visualizer visSchedule("trayectorias_schedule.dat", "energia_schedule.dat");
    visSchedule.clearFiles();
    const auto& statesSchedule = simSchedule.getStateHistory();

    for (size_t step = 0; step < statesSchedule.size(); ++step) {
        visSchedule.saveState(step, statesSchedule[step]);
        visSchedule.saveEnergy(step, simSchedule.getEnergyHistory()[step].first, simSchedule.getEnergyHistory()[step].second); 
    }

    // -------------------------------
    // VERSION CHUNK
    // -------------------------------
    Integrator simChunk(N, seed, dt, G, softening, total_steps);

    std::cout << "\nIniciando simulacion runSimulationChunk()..." << std::endl;

    double start_chunk = omp_get_wtime();
    simChunk.runSimulationChunk();
    double end_chunk = omp_get_wtime();

    std::cout << "Simulacion Chunk completada." << std::endl;
    std::cout << "Tiempo runSimulationChunk(): " 
              << (end_chunk - start_chunk) << " segundos" << std::endl;

    Visualizer visChunk("trayectorias_chunk.dat", "energia_chunk.dat");
    visChunk.clearFiles();
    const auto& statesChunk = simChunk.getStateHistory();

    for (size_t step = 0; step < statesChunk.size(); ++step) {
        visChunk.saveState(step, statesChunk[step]);
        visChunk.saveEnergy(step, simChunk.getEnergyHistory()[step].first, simChunk.getEnergyHistory()[step].second);
    }

    // -------------------------------
    // VERSION COLLAPSE
    // -------------------------------
    Integrator simCollapse(N, seed, dt, G, softening, total_steps);

    std::cout << "\nIniciando simulacion runSimulationCollapse()..." << std::endl;

    double start_collapse = omp_get_wtime();
    simCollapse.runSimulationCollapse();
    double end_collapse = omp_get_wtime();

    std::cout << "Simulacion Collapse completada." << std::endl;
    std::cout << "Tiempo runSimulationCollapse(): " 
              << (end_collapse - start_collapse) << " segundos" << std::endl;

    Visualizer visCollapse("trayectorias_collapse.dat", "energia_collapse.dat");
    visCollapse.clearFiles();
    const auto& statesCollapse = simCollapse.getStateHistory();

    for (size_t step = 0; step < statesCollapse.size(); ++step) {
        visCollapse.saveState(step, statesCollapse[step]);
        visCollapse.saveEnergy(step, simCollapse.getEnergyHistory()[step].first, simCollapse.getEnergyHistory()[step].second);
    }

    // -------------------------------
    // VERSION NEWTON3
    // -------------------------------
    Integrator simNewton3(N, seed, dt, G, softening, total_steps);

    std::cout << "\nIniciando simulacion runSimulationNewton3()..." << std::endl;

    double start_newton3 = omp_get_wtime();
    simNewton3.runSimulationNewton3();
    double end_newton3 = omp_get_wtime();

    std::cout << "Simulacion Newton3 completada." << std::endl;
    std::cout << "Tiempo runSimulationNewton3(): " 
              << (end_newton3 - start_newton3) << " segundos" << std::endl;

    Visualizer visNewton3("trayectorias_newton3.dat", "energia_newton3.dat");
    visNewton3.clearFiles();
    const auto& statesNewton3 = simNewton3.getStateHistory();

    for (size_t step = 0; step < statesNewton3.size(); ++step) {
        visNewton3.saveState(step, statesNewton3[step]);
        visNewton3.saveEnergy(step, simNewton3.getEnergyHistory()[step].first, simNewton3.getEnergyHistory()[step].second);
    }


    // Automaticamente se liberan los recursos al salir del main, se invocan los destructores de Integrator y Visualizer
    return 0;
}