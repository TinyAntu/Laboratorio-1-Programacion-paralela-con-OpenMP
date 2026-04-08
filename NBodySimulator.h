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
    std::pair<double, double> energy_system;        // Para almacenar energía cinética y potencial del sistema

public:
    NBodySimulator(int N, unsigned int seed, double G, double softening, double dt);
    ~NBodySimulator();
    
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

    const std::vector<Particle>& getBodies() const { return system->getBodies(); }
    const NBodySystem* getSystem() const { return system; }
    const std::pair<double, double>& getSystemEnergy() const { return energy_system; } 

};

#endif // NBODYSIMULATOR_H