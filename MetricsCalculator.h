#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include "NBodySystem.h"

class MetricsCalculator {
private:
    NBodySystem* system;

public:
    MetricsCalculator(NBodySystem* sys);

    // Métodos para el cálculo de energías K y U
    double calculateKineticEnergy() const;
    double calculatePotentialEnergy() const;
    double calculateTotalEnergy() const;
};

#endif // METRICSCALCULATOR_H