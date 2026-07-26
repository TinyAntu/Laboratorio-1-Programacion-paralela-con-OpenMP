#include "NBodySimulator.h"
#include "MetricsCalculator.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

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

// ===== Lab2: paso temporal con aceleraciones calculadas en GPU =====
// Orden fijo del enunciado (sección 4.1): (1) lanzar kernel de aceleraciones y
// (2) cudaDeviceSynchronize ocurren dentro de computeAccelerationsGpu; luego
// (3) Euler kick/drift en host; (4) la subida de posiciones a device del paso
// siguiente la hace computeAccelerationsGpu (masas se suben una sola vez).
void NBodySimulator::stepEulerGpu() {
    // 1. Subir posiciones y calcular aceleraciones en GPU.
    // 2. La sincronización y descarga ocurren dentro de este método.
    system->computeAccelerationsGpu();

    // 3. Euler explícito en host: primero kick, luego drift.
    processBodies();

    // 4. Calcular K y U en GPU mediante reducción shared.
    calculateEnergyGpu();
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



void NBodySimulator::integrateEuler(int sync_type) {
    system->zeroAccelerations();
    system->computeAccelerations();

    std::vector<Particle>& bodies = system->getBodies();
    int n = static_cast<int>(bodies.size());

    if (sync_type == 0) {
        double total_kinetic = 0.0;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].kick(time_step);
            bodies[i].drift(time_step);

            double vx = bodies[i].getVx();
            double vy = bodies[i].getVy();
            double contrib = 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
            #pragma omp atomic
            total_kinetic += contrib;
        }
        (void)total_kinetic;

    } else if (sync_type == 1) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx() + bodies[i].getAx() * time_step;
            double vy = bodies[i].getVy() + bodies[i].getAy() * time_step;
            #pragma omp critical
            {
                bodies[i].setVx(vx);
                bodies[i].setVy(vy);
            }
            bodies[i].drift(time_step);
        }

    } else if (sync_type == 2) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; ++i) {
                bodies[i].kick(time_step);
            }
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n; ++i) {
                bodies[i].drift(time_step);
            }
        }

    } else {
        throw std::invalid_argument(
            "sync_type invalido: use 0=atomic, 1=critical, 2=nowait");
    }

    calculateEnergy();
}

void NBodySimulator::integrateEuler(int sync_type, bool use_barrier) {
    system->zeroAccelerations();
    system->computeAccelerations();

    std::vector<Particle>& bodies = system->getBodies();
    int n = static_cast<int>(bodies.size());

    double total_kinetic = 0.0;

    #pragma omp parallel shared(total_kinetic)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx() + bodies[i].getAx() * time_step;
            double vy = bodies[i].getVy() + bodies[i].getAy() * time_step;

            if (sync_type == 0) {
                bodies[i].setVx(vx);
                bodies[i].setVy(vy);

                double contrib =
                    0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
                #pragma omp atomic
                total_kinetic += contrib;

            } else if (sync_type == 1) {
                #pragma omp critical
                {
                    bodies[i].setVx(vx);
                    bodies[i].setVy(vy);
                }

            } else if (sync_type == 2) {
                bodies[i].setVx(vx);
                bodies[i].setVy(vy);

            } else {
                throw std::invalid_argument(
                    "sync_type invalido: use 0=atomic, 1=critical, 2=nowait");
            }
        }

        if (use_barrier) {
            #pragma omp barrier
        }

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            bodies[i].drift(time_step);
        }
    }

    (void)total_kinetic;

    calculateEnergy();
}

void NBodySimulator::calculateEnergy(int method) {
    const auto& bodies = system->getBodies();
    double G         = system->getG();
    double soft      = system->getSoftening();
    MetricsCalculator calc;

    double K = calc.calculateKineticEnergyParallel(bodies, method);
    double U = calc.calculatePotentialEnergyParallel(bodies, G, soft, method);
    energy_system = {K, U};
}

void NBodySimulator::calculateEnergy(int method, bool use_private) {
    const auto& bodies = system->getBodies();
    double G         = system->getG();
    double soft      = system->getSoftening();
    MetricsCalculator calc;

    double K = calc.calculateKineticEnergyParallel(bodies, method, use_private);
    double U = calc.calculatePotentialEnergyParallel(bodies, G, soft, method);
    energy_system = {K, U};
}

//Procesamiento task, parallel_for

void NBodySimulator::processBodies(int task_type){
    std::vector<Particle>& bodies = system->getBodies();  

    if (task_type == 0) {
        // Implemtación usando task
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (size_t i = 0; i < bodies.size(); ++i) {
                    #pragma omp task firstprivate(i) // Kick y Drift necesitan la copia privada para trabajar la misma particula
                    {
                        bodies[i].kick(time_step);
                        bodies[i].drift(time_step);
                    }
                }
            }
        }
    } else if (task_type == 1) {
        // Implementación usando parallel for
        #pragma omp parallel for
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].kick(time_step);
            bodies[i].drift(time_step);
        }
    }
}

NBodySystem& NBodySimulator::getSystem() {
    return *system;
}

void NBodySimulator::processBodies(int task_type, bool use_single) {
    std::vector<Particle>& bodies = system->getBodies();
    int n = static_cast<int>(bodies.size());
    const int chunk_size = 64;

    if (task_type == 0) {
        if (use_single) {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    for (int start = 0; start < n; start += chunk_size) {
                        int end = std::min(start + chunk_size, n);
                        #pragma omp task firstprivate(start, end)
                        {
                            for (int i = start; i < end; ++i) {
                                bodies[i].kick(time_step);
                                bodies[i].drift(time_step);
                            }
                        }
                    }
                }
            }
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                int per_thread = (n + nthreads - 1) / nthreads;
                int start_t = tid * per_thread;
                int end_t = std::min(start_t + per_thread, n);

                for (int start = start_t; start < end_t; start += chunk_size) {
                    int end = std::min(start + chunk_size, end_t);
                    #pragma omp task firstprivate(start, end)
                    {
                        for (int i = start; i < end; ++i) {
                            bodies[i].kick(time_step);
                            bodies[i].drift(time_step);
                        }
                    }
                }
                #pragma omp taskwait
            }
        }

    } else if (task_type == 1) {
        if (use_single) {
            #pragma omp parallel
            {
                #pragma omp single nowait
                {}

                #pragma omp for schedule(static)
                for (int i = 0; i < n; ++i) {
                    bodies[i].kick(time_step);
                    bodies[i].drift(time_step);
                }
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                bodies[i].kick(time_step);
                bodies[i].drift(time_step);
            }
        }

    } else {
        throw std::invalid_argument(
            "task_type invalido: use 0=task, 1=parallel_for");
    }
}

void NBodySimulator::simulatePhasesBarrier() {
    std::vector<Particle>& bodies = system->getBodies();
    int n      = static_cast<int>(bodies.size());
    double G   = system->getG();
    double eps = system->getSoftening();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].resetAcceleration();
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            double ax = 0.0, ay = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double dx     = bodies[j].getX() - bodies[i].getX();
                double dy     = bodies[j].getY() - bodies[i].getY();
                double dist2  = dx*dx + dy*dy + eps*eps;
                double inv3   = 1.0 / (dist2 * std::sqrt(dist2));
                double factor = G * bodies[j].getMass() * inv3;
                ax += factor * dx;
                ay += factor * dy;
            }
            bodies[i].setAcceleration(ax, ay);
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].kick(time_step);
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].drift(time_step);
        }
    }

    calculateEnergy();
}

void NBodySimulator::parallelInitializationSingle() {
    std::vector<Particle>& bodies = system->getBodies();
    int n = static_cast<int>(bodies.size());

    double total_mass = 0.0;
    double cm_x = 0.0, cm_y = 0.0;

    #pragma omp parallel shared(total_mass, cm_x, cm_y)
    {
        #pragma omp single
        {
            for (int i = 0; i < n; ++i) {
                double m = bodies[i].getMass();
                total_mass += m;
                cm_x += m * bodies[i].getX();
                cm_y += m * bodies[i].getY();
            }
            if (total_mass > 0.0) {
                cm_x /= total_mass;
                cm_y /= total_mass;
            }
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].kick(time_step);
            bodies[i].drift(time_step);
        }
    }

    calculateEnergy();
}

void NBodySimulator::calculateMetricsFirstprivate() {
    MetricsCalculator calc;
    const auto& bodies = system->getBodies();
    SystemMetrics m = calc.calculateMetricsFirstprivate(
        bodies, system->getG(), system->getSoftening());
    energy_system = {m.kineticEnergy, m.potentialEnergy};
}

void NBodySimulator::calculateFinalStateLastprivate() {
    MetricsCalculator calc;
    const auto& bodies = system->getBodies();
    SystemMetrics m = calc.calculateFinalStateLastprivate(
        bodies, system->getG(), system->getSoftening());
    energy_system = {m.kineticEnergy, m.potentialEnergy};
}

void NBodySimulator::calculateEnergyGpu() {
    // Método por defecto: reducción con shared memory.
    calculateEnergyGpu(0);
}

void NBodySimulator::calculateEnergyGpu(int method) {
    energy_system = system->computeEnergyGpu(method);
}