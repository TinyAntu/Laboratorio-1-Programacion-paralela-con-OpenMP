#include "NBodySimulator.h"
#include "MetricsCalculator.h"
#include <iostream>
#include <cmath>

// Constructor del simulador, recibe un puntero al sistema de cuerpos y el paso de tiempo a usar en la integración temporal
NBodySimulator::NBodySimulator(int N, unsigned int seed, double G, double softening, double dt) 
    : time_step(dt) 
{
    system = new NBodySystem(G, softening);
    system->loadFromSeed(seed, N);
}

NBodySimulator::~NBodySimulator() {
    delete system;
}

// Integración temporal secuencial (Lineal)
void NBodySimulator::integrateEuler() {
    // Limpiar las aceleraciones del paso temporal anterior
    system->zeroAccelerations();

    // Calcular las nuevas fuerzas/aceleraciones basadas en las posiciones actuales
    system->computeAccelerations();

    // Aplicar los cambios a la velocidad y posición en base al método de Euler (Kick & Drift)
    processBodies();

    // Calculamos la energía del sistema después de la integración de este paso
    calculateEnergy();
}

// Versión que prueba distintos schedules (static/dynamic/guided)
void NBodySimulator::integrateEulerSchedule() {
    // 0 = static, 1 = dynamic, 2 = guided
    int schedule_type = 0; 

    system->zeroAccelerations();

    system->computeAccelerations(schedule_type);

    processBodies();

    calculateEnergy();
}

// Versión que además controla chunk_size
void NBodySimulator::integrateEulerChunk() {
    // 0 = static, 1 = dynamic, 2 = guided
    int schedule_type = 1;   
    int chunk_size = 8;      

    system->zeroAccelerations();

    system->computeAccelerations(schedule_type, chunk_size);

    processBodies();

    calculateEnergy();
}

// Versión usando collapse(2) en el cálculo de aceleraciones
void NBodySimulator::integrateEulerCollapse() {
    
    system->zeroAccelerations();

    system->computeAccelerationsCollapse();

    processBodies();

    calculateEnergy();
}

void NBodySimulator::integrateEulerNewton3() {
    
    system->zeroAccelerations();

    system->computeAccelerationsNewton3();

    processBodies();

    calculateEnergy();
}

// Implementación secuencial
void NBodySimulator::calculateEnergy() {
    const auto& bodies = system->getBodies();
    double G = system->getG();
    double softening = system->getSoftening();
    MetricsCalculator metrics_calc;
    double K = metrics_calc.calculateKineticEnergy(bodies);
    double U = metrics_calc.calculatePotentialEnergy(bodies, G, softening);
    energy_system = {K, U};                
}


// Procesamiento secuencial de las partículas (Kick & Drift)
void NBodySimulator::processBodies() {
    // Obtenemos la referencia a las partículas para editar sus estados (posiciones, velocidades, aceleraciones)
    std::vector<Particle>& bodies = system->getBodies(); 

    // Actualizamos el estado de cada partícula en base a las aceleraciones calculadas
    for (auto& body : bodies) {
        body.kick(time_step);  // Actualiza velocidad (v += a*dt)
        body.drift(time_step); // Actualiza posición (r += v*dt)
    }
}
