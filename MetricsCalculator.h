#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include "Particle.h"
#include <vector>
#include <array>
#include <cmath>

struct SystemMetrics {
        double kineticEnergy;
        double potentialEnergy;
        double totalEnergy;
        double momentumX;
        double momentumY;
        double momentumMag;
        double centerOfMassX;
        double centerOfMassY;
        double rmsRadius;
        double minDistance;
};

class MetricsCalculator {
public:
    double calculateKineticEnergy(const std::vector<Particle>& bodies) const;
    double calculatePotentialEnergy(const std::vector<Particle>& bodies,
                                    double G, double softening) const;
    double calculateTotalEnergy(const std::vector<Particle>& bodies,
                                double G, double softening) const;
    std::array<double, 2> calculateLinearMomentum(
            const std::vector<Particle>& bodies) const;
    std::array<double, 2> calculateCenterOfMass(
            const std::vector<Particle>& bodies) const;
    double calculateRMSRadius(const std::vector<Particle>& bodies) const;
    double calculateMinDistance(const std::vector<Particle>& bodies) const;
    SystemMetrics calculateAll(const std::vector<Particle>& bodies,
                               double G, double softening) const;
    double calculateKineticEnergyParallel(const std::vector<Particle>& bodies,
                                          int method) const;

    double calculatePotentialEnergyParallel(const std::vector<Particle>& bodies,
                                            double G, double softening,
                                            int method) const;
    double calculateKineticEnergyParallel(const std::vector<Particle>& bodies,
                                          int method, bool use_private) const;
    SystemMetrics calculateMetricsFirstprivate(const std::vector<Particle>& bodies,
                                               double G, double softening) const;
    SystemMetrics calculateFinalStateLastprivate(const std::vector<Particle>& bodies,
                                                 double G, double softening) const;
};

#endif
