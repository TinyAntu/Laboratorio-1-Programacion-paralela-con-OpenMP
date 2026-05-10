#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Particle.h"
#include "MetricsCalculator.h"

#include <string>
#include <vector>

class Visualizer {
private:
    std::string trajectory_file;
    std::string energy_file;

public:
    Visualizer(const std::string& traj_f = "snapshots.dat",
               const std::string& ener_f = "energy_timeseries.dat");

    void clearFiles();

    void saveState(int step, const std::vector<Particle>& bodies);

    void saveEnergy(int step, const SystemMetrics& metrics);

    void runAll(
        int N,
        unsigned int seed,
        double G,
        double softening,
        double dt,
        int steps,
        int sample_every
    );
};

#endif // VISUALIZER_H