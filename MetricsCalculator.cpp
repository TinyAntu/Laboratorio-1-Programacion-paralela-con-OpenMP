#include "MetricsCalculator.h"


double MetricsCalculator::calculateKineticEnergy(const std::vector<Particle>& bodies) const {
    double K = 0.0;
    for (const auto& p : bodies) {
        double vx = p.getVx();
        double vy = p.getVy();
        K += 0.5 * p.getMass() * (vx * vx + vy * vy);
    }
    return K;
}

double MetricsCalculator::calculatePotentialEnergy(const std::vector<Particle>& bodies, double G, double softening) const {
    double U = 0.0;
    int n = bodies.size();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + softening * softening);
            
            U -= (G * bodies[i].getMass() * bodies[j].getMass()) / dist;
        }
    }
    return U;
}

double MetricsCalculator::calculateTotalEnergy(const std::vector<Particle>& bodies, double G, double softening) const {
    return calculateKineticEnergy(bodies) + calculatePotentialEnergy(bodies, G, softening);
}