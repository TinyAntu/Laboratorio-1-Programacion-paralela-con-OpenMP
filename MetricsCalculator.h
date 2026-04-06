#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include "Particle.h"
#include <vector>
#include <cmath>


class MetricsCalculator {
public:
    double calculateKineticEnergy(const std::vector<Particle>& bodies) const;
    double calculatePotentialEnergy(const std::vector<Particle>& bodies, double G, double softening) const;
    double calculateTotalEnergy(const std::vector<Particle>& bodies, double G, double softening) const;
};

#endif