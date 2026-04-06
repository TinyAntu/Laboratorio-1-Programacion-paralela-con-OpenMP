#include "NBodySimulator.h"
#include "MetricsCalculator.h"
#include <iostream>
#include <cmath>

// Constructor del simulador, recibe un puntero al sistema de cuerpos y el paso de tiempo a usar en la integración temporal
NBodySimulator::NBodySimulator(NBodySystem* sys, double dt) 
    : system(sys), time_step(dt) {}

// Integración temporal secuencial (Lineal)
void NBodySimulator::integrateEuler() {
    // Limpiar las aceleraciones del paso temporal anterior
    system->zeroAccelerations();

    // Calcular las nuevas fuerzas/aceleraciones basadas en las posiciones actuales
    system->computeAccelerations();

    // Aplicar los cambios a la velocidad y posición en base al método de Euler (Kick & Drift)
    processBodies();
}

std::pair<double, double> NBodySimulator::calculateEnergy() {
    MetricsCalculator metrics_calc;
    const std::vector<Particle>& bodies = system->getBodies();
    double ke = metrics_calc.calculateKineticEnergy(bodies);
    double pe = metrics_calc.calculatePotentialEnergy(bodies, system->getG(), system->getSoftening());
    return {ke, pe}; 
}

// Procesamiento secuencial de las partículas (Kick & Drift)
void NBodySimulator::processBodies() {
    // Obtenemos la referencia a las partículas
    std::vector<Particle>& bodies = system->getBodies(); 

    // Actualizamos el estado de cada partícula en base a las aceleraciones calculadas
    for (auto& body : bodies) {
        body.kick(time_step);  // Actualiza velocidad (v += a*dt)
        body.drift(time_step); // Actualiza posición (r += v*dt)
    }
}
