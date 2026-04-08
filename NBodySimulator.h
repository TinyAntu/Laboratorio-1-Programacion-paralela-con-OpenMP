//Lazo temporal, variantes OpenMP (orquestaci ́on)
#ifndef NBODYSIMULATOR_H
#define NBODYSIMULATOR_H

#include "NBodySystem.h"
#include "MetricsCalculator.h"
#include <utility>

class NBodySimulator {
private:
    NBodySystem* system;
    double time_step;

public:
    NBodySimulator(NBodySystem* sys, double dt);
    
    // integracion temporal
    void integrateEuler();
    void integrateEulerSchedule();
    void integrateEulerChunk();
    void integrateEulerCollapse();
    void integrateEuler(int sync_type); // 0=atomic, 1=critical, 2=nowait,→ (convenci ́on del enunciado)
    void integrateEuler(int sync_type, bool use_barrier);

    // energia y metricas globales
    void calculateEnergy();
    void calculateEnergy(int method); // reduce=0, atomic=1
    void calculateEnergy(int method, bool use_private);

    // repatro de trabajos sobre indices
    void processBodies();
    void processBodies(int task_type); // task=0, parallel_for=1
    void processBodies(int task_type, bool use_single);
    
    // Metodos dedicados a cl´ausulas puntuales:
    void simulatePhasesBarrier();
    void parallelInitializationSingle();
    
    // Estos pueden ir en el metrics calculator
    void calculateMetricsFirstprivate();
    void calculateFinalStateLastprivate();

};

#endif // NBODYSIMULATOR_H