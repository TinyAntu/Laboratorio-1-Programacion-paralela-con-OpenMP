#include "MetricsCalculator.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <omp.h>

MetricsCalculator::MetricsCalculator(NBodySystem* sys) : system(sys) {}

// Cálculo de Energía Cinética (K = 1/2 * sum(m_i * v_i^2)) [cite: 78, 79]
double MetricsCalculator::calculateKineticEnergy() const {
    double K = 0.0;
    const auto& bodies = system->getBodies();
    int n = bodies.size();

    // Reducción paralela estándar
    #pragma omp parallel for reduction(+:K)
    for (int i = 0; i < n; ++i) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
    }
    return K;
}

// Cálculo de Energía Potencial (U = -G * sum(m_i * m_j / dist)) [cite: 79]
double MetricsCalculator::calculatePotentialEnergy() const {
    double U = 0.0;
    const auto& bodies = system->getBodies();
    int n = bodies.size();
    double G = system->getG();
    double eps = system->getSoftening();

    // Aquí usamos reduction explícito para sumar la energía potencial de forma segura 
    #pragma omp parallel for reduction(-:U) schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps * eps);
            U -= (G * bodies[i].getMass() * bodies[j].getMass()) / dist;
        }
    }
    return U;
}

double MetricsCalculator::calculateTotalEnergy() const {
    double K = calculateKineticEnergy();
    double U = calculatePotentialEnergy();
    return K + U;
}
