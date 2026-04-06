#include "Visualizer.h"
#include <fstream>
#include <iomanip>

Visualizer::Visualizer(const std::string& traj_f, const std::string& ener_f)
    : trajectory_file(traj_f), energy_file(ener_f) {}

void Visualizer::clearFiles() {
    std::ofstream f1(trajectory_file, std::ios::trunc);
    if (f1.is_open()) f1 << "Step ID X Y Mass\n";
    
    std::ofstream f2(energy_file, std::ios::trunc);
    if (f2.is_open()) f2 << "Step Kinetic Potential Total\n";
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

void Visualizer::saveEnergy(int step, double kinetic, double potential) {
    std::ofstream outFile(energy_file, std::ios::app);
    if (outFile.is_open()) {
        outFile << step << " " 
                << std::fixed << std::setprecision(6) << kinetic << " " 
                << std::fixed << std::setprecision(6) << potential << " " 
                << std::fixed << std::setprecision(6) << (kinetic + potential) << "\n";
    }
}