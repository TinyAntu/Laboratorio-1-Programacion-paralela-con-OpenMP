#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Particle.h"
#include <string>
#include <vector>

class Visualizer {
private:
    std::string trajectory_file;
    std::string energy_file;

public:
    // Constructor independiente del sistema "HAY QUE CAMBIAR NOMBRES"
    Visualizer(const std::string& traj_f = "trayectorias.dat", 
               const std::string& ener_f = "energia.dat");

    void clearFiles();
    
    // Recibe las partículas explícitamente
    void saveState(int step, const std::vector<Particle>& bodies);
    
    // Recibe los valores de energía explícitamente
    void saveEnergy(int step, double kinetic, double potential);
};

#endif // VISUALIZER_H