#include "Visualizer.h"
#include "NBodySimulator.h"
#include <fstream>
#include <iomanip>
#include <iostream>

Visualizer::Visualizer(const std::string& traj_f, const std::string& ener_f)
    : trajectory_file(traj_f), energy_file(ener_f) {}

void Visualizer::clearFiles() {
    std::ofstream f1(trajectory_file, std::ios::trunc);
    if (f1.is_open()) {
        f1 << "Step ID X Y Mass\n";
    }

    std::ofstream f2(energy_file, std::ios::trunc);
    if (f2.is_open()) {
        f2 << "Step Kinetic Potential Total "
           << "CenterOfMassX CenterOfMassY RMSRadius "
           << "MomentumX MomentumY MomentumMag MinDistance\n";
    }
}

void Visualizer::saveState(int step, const std::vector<Particle>& bodies) {
    std::ofstream outFile(trajectory_file, std::ios::app);
    if (outFile.is_open()) {
        // Almacena por cada elemento del sistema su estado (posición, masa) en el archivo de trayectorias
        for (size_t i = 0; i < bodies.size(); ++i) {
            outFile << step << " " << i << " " 
                    << std::fixed << std::setprecision(6) << bodies[i].getX() << " " 
                    << std::fixed << std::setprecision(6) << bodies[i].getY() << " " 
                    << std::fixed << std::setprecision(6) << bodies[i].getMass() << "\n";
        }
    }
}

void Visualizer::saveEnergy(int step, const SystemMetrics& metrics) {
    std::ofstream outFile(energy_file, std::ios::app);
    if (outFile.is_open()) {
        outFile << step << " "
                << std::fixed << std::setprecision(6)
                << metrics.kineticEnergy << " "
                << metrics.potentialEnergy << " "
                << metrics.totalEnergy << " "
                << metrics.centerOfMassX << " "
                << metrics.centerOfMassY << " "
                << metrics.rmsRadius << " "
                << metrics.momentumX << " "
                << metrics.momentumY << " "
                << metrics.momentumMag << " "
                << metrics.minDistance << "\n";
    }
}

void Visualizer::runAll(
    int N,
    unsigned int seed,
    double G,
    double softening,
    double dt,
    int steps,
    int sample_every
) {
    std::cout << "[Visualizer] Generando salidas físicas...\n";
    std::cout << "[Visualizer] N=" << N
              << "  steps=" << steps
              << "  sample_every=" << sample_every << "\n";

    if (sample_every <= 0) {
        std::cerr << "[Visualizer] sample_every debe ser mayor que 0. Usando sample_every=1.\n";
        sample_every = 1;
    }

    clearFiles();


    NBodySimulator simulator(N, seed, G, softening, dt);

    MetricsCalculator metricsCalculator;

    for (int step = 0; step <= steps; ++step) {

        if (step % sample_every == 0) {
            const auto& bodies = simulator.getBodies();

            SystemMetrics sysMetrics = metricsCalculator.calculateAll(bodies, G, softening);

            saveState(step, bodies);
            saveEnergy(step, sysMetrics);
        }

        simulator.integrateEuler();
    }

    std::cout << "[Visualizer] Archivos generados:\n";
    std::cout << "  - " << trajectory_file << "\n";
    std::cout << "  - " << energy_file << "\n";
    std::cout << "[Visualizer] Finalizado.\n";
}