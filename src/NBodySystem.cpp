#include "NBodySystem.h"
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <stdexcept>

NBodySystem::NBodySystem(double G, double epsilon) : G_const(G), softening_eps(epsilon) {}

void NBodySystem::addParticle(const Particle& p) {
    bodies.push_back(p);
}

void NBodySystem::zeroAccelerations() {
    for (auto& body : bodies) {
        body.resetAcceleration();
    }
}

void NBodySystem::computeAccelerations() {
    zeroAccelerations();

    // Implementación secuencial
    int n = bodies.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dx = bodies[j].getX() - bodies[i].getX();
                double dy = bodies[j].getY() - bodies[i].getY();
                double distSqr = dx * dx + dy * dy + softening_eps * softening_eps;
                double invDist3 = 1.0 / (distSqr * std::sqrt(distSqr));
                // La aceleración solo depende de la masa del OTRO cuerpo (m_j) y la constante G
                double commonFactor = G_const * bodies[j].getMass() * invDist3;
                bodies[i].addAcceleration(commonFactor * dx, commonFactor * dy);
            }
        }
    }
}

// Sobrecarga con schedule runtime, se setea el tipo de schedule con omp_set_schedule antes del parallel for
void NBodySystem::computeAccelerations(int schedule_type) {
    // 0 = static, 1 = dynamic, 2 = guided
    zeroAccelerations();

    int n = bodies.size();
    omp_sched_t omp_schedule;

    // Mapear el schedule_type a los valores de OpenMP
    switch (schedule_type) {
        case 0:
            omp_schedule = omp_sched_static;
            break;
        case 1:
            omp_schedule = omp_sched_dynamic;
            break;
        case 2:
            omp_schedule = omp_sched_guided;
            break;
        default:
            throw std::invalid_argument("schedule_type invalido: use 0=static, 1=dynamic, 2=guided");
    }

    // El chunk_size se ignora en esta version, se usara el valor por defecto del runtime
    omp_set_schedule(omp_schedule, 0);

    #pragma omp parallel for schedule(runtime) // El runtime usara el schedule previamente seteado
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        double ay = 0.0;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double distSqr = dx * dx + dy * dy + softening_eps * softening_eps;
            double invDist3 = 1.0 / (distSqr * std::sqrt(distSqr));
            double commonFactor = G_const * bodies[j].getMass() * invDist3;

            ax += commonFactor * dx;
            ay += commonFactor * dy;
        }

        bodies[i].setAcceleration(ax, ay);
    }
}

// Sobrecarga con schedule runtime y chunk_size, se setea el tipo de schedule y el chunk_size con omp_set_schedule antes del parallel for
void NBodySystem::computeAccelerations(int schedule_type, int chunk_size) {
    // 0 = static, 1 = dynamic, 2 = guided
    zeroAccelerations();

    int n = bodies.size();
    omp_sched_t omp_schedule;

    // Mapear el schedule_type a los valores de OpenMP
    switch (schedule_type) {
        case 0:
            omp_schedule = omp_sched_static;
            break;
        case 1:
            omp_schedule = omp_sched_dynamic;
            break;
        case 2:
            omp_schedule = omp_sched_guided;
            break;
        default:
            throw std::invalid_argument("schedule_type invalido: use 0=static, 1=dynamic, 2=guided");
    }

    // Validar que chunk_size sea positivo
    if (chunk_size <= 0) {
        throw std::invalid_argument("chunk_size debe ser mayor que 0");
    }

    omp_set_schedule(omp_schedule, chunk_size); // Setear el schedule con el chunk_size especificado

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        double ay = 0.0;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double distSqr = dx * dx + dy * dy + softening_eps * softening_eps;
            double invDist3 = 1.0 / (distSqr * std::sqrt(distSqr));
            double commonFactor = G_const * bodies[j].getMass() * invDist3;

            ax += commonFactor * dx;
            ay += commonFactor * dy;
        }

        bodies[i].setAcceleration(ax, ay);
    }
}

// Implementación con collapse(2) para paralelizar ambos bucles anidados, se deben usar variables privadas para acumular las aceleraciones y luego actualizar las partículas al 
//final
void NBodySystem::computeAccelerationsCollapse() {

    // Limpiar aceleraciones previas
    zeroAccelerations();

    int n = bodies.size();
    std::vector<double> ax(n, 0.0);
    std::vector<double> ay(n, 0.0);

    #pragma omp parallel for collapse(2) schedule(static) // Paralelizar ambos bucles anidados con collapse(2). Se usa 2 porque son dos bucles anidados (i,j)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double distSqr = dx * dx + dy * dy + softening_eps * softening_eps;
            double invDist3 = 1.0 / (distSqr * std::sqrt(distSqr));
            double commonFactor = G_const * bodies[j].getMass() * invDist3;

            double dax = commonFactor * dx;
            double day = commonFactor * dy;

            // Acumular las contribuciones a la aceleracion en vectores privados para cada hilo, luego se actualizaran las particulas al final del bucle
            #pragma omp atomic // Usamos atomic para evitar condiciones de carrera al acumular las aceleraciones en los vectores ax, ay
            ax[i] += dax;

            #pragma omp atomic 
            ay[i] += day;
        }
    }

    #pragma omp parallel for schedule(static) // Actualizar las aceleraciones de las partículas después de acumularlas en los vectores ax, ay
    for (int i = 0; i < n; ++i) {
        bodies[i].setAcceleration(ax[i], ay[i]);
    }
}

// Implementación con Newton3, cada interacción se calcula una sola vez y se actualizan ambas partículas simultáneamente, se deben usar variables privadas para acumular las aceleraciones y luego actualizar las partículas al final
// la base de esta implementacion es la tercera ley de Newton: la fuerza que ejerce la partícula i sobre j es igual y opuesta a la fuerza que ejerce j sobre i, por lo que solo calculamos la interacción una vez y actualizamos ambas partículas simultáneamente
void NBodySystem::computeAccelerationsNewton3() {
    zeroAccelerations();
    int n = bodies.size();
    
    std::vector<double> ax_global(n, 0.0);
    std::vector<double> ay_global(n, 0.0);

    #pragma omp parallel
    {
        std::vector<double> ax_local(n, 0.0);
        std::vector<double> ay_local(n, 0.0);


        #pragma omp for schedule(dynamic, 10) 
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                
                double dx = bodies[j].getX() - bodies[i].getX();
                double dy = bodies[j].getY() - bodies[i].getY();
                double distSqr = dx * dx + dy * dy + softening_eps * softening_eps;
                double invDist3 = 1.0 / (distSqr * std::sqrt(distSqr));
                
                double factor_i = G_const * bodies[j].getMass() * invDist3;
                double factor_j = G_const * bodies[i].getMass() * invDist3;


                ax_local[i] += factor_i * dx;
                ay_local[i] += factor_i * dy;

                // Restamos a 'j' (dirección opuesta)
                ax_local[j] -= factor_j * dx;
                ay_local[j] -= factor_j * dy;
            }
        }

        // Una vez que el hilo termina todos sus cálculos, suma su 
        // resultado privado al acumulador global de forma segura.
        #pragma omp critical
        {
            for (int k = 0; k < n; ++k) {
                ax_global[k] += ax_local[k];
                ay_global[k] += ay_local[k];
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].setAcceleration(ax_global[i], ay_global[i]);
    }
}

//Para realizar la reproducidad de los experimentos
//Genera N partículas con posiciones y masas aleatorias a partir de una semilla dada 
void NBodySystem::loadFromSeed(unsigned int seed, int N) {
    bodies.clear();
    std::mt19937 gen(seed); 
    
    std::uniform_real_distribution<double> posDist(-10.0, 10.0);
    std::uniform_real_distribution<double> massDist(0.1, 1.0);

    for (int i = 0; i < N; ++i) {
        double m = massDist(gen);
        double x = posDist(gen);
        double y = posDist(gen);
        addParticle(Particle(m, x, y));
    }
}


// Guarda las posiciones de las partículas en un archivo de texto, con formato: x1 y1 x2 y2 ... xn yn
//Modo "append" guarda toda la trayectoria "trunc" para un snapshot

void NBodySystem::writePositions(const std::string& filename) {
    
    std::ofstream outFile(filename, std::ios::app); 
    if (outFile.is_open()) {
        for (const auto& p : bodies) {
            outFile << std::fixed << std::setprecision(6) 
                    << p.getX() << " " << p.getY() << " ";
        }
        outFile << "\n"; 
        outFile.close();
    }
}

const std::vector<Particle>& NBodySystem::getBodies() const {
    return bodies;
}

std::vector<Particle>& NBodySystem::getBodies() {
    return bodies;
}

int NBodySystem::getCount() const {
    return bodies.size();
}

double NBodySystem::getG() const {
    return G_const;
}

double NBodySystem::getSoftening() const {
    return softening_eps;
}

// ===== Lab2: soporte GPU =====
// Con CUDA habilitado (NBODY_HAS_CUDA, definido por CMake) las definiciones
// reales del destructor y de las sobrecargas GPU viven en src/NBodySystemGpu.cu.
// Sin CUDA se definen aquí como stubs para que los builds solo-CPU (CI) linkeen.
#ifndef NBODY_HAS_CUDA

NBodySystem::~NBodySystem() = default; // sin CUDA nunca se crea estado GPU

void NBodySystem::computeAccelerationsGpu() {
    computeAccelerationsGpu(0, 256);
}

void NBodySystem::computeAccelerationsGpu(int variant) {
    computeAccelerationsGpu(variant, 256);
}

void NBodySystem::computeAccelerationsGpu(int /*variant*/, int /*block_size*/) {
    throw std::runtime_error(
        "computeAccelerationsGpu: el binario fue compilado sin soporte CUDA "
        "(configure con -DENABLE_CUDA=ON y el CUDA Toolkit instalado)");
}

std::pair<double, double> NBodySystem::computeEnergyGpu(
    int /*method*/,
    int /*block_size*/
) {
    throw std::runtime_error(
        "computeEnergyGpu: el binario fue compilado sin soporte CUDA "
        "(configure con -DENABLE_CUDA=ON y el CUDA Toolkit instalado)"
    );
}

double NBodySystem::computeAccelerationsGpuKernelOnly(int /*variant*/, int /*block_size*/,
                                                      int /*steps*/) {
    throw std::runtime_error(
        "computeAccelerationsGpuKernelOnly: el binario fue compilado sin soporte CUDA "
        "(configure con -DENABLE_CUDA=ON y el CUDA Toolkit instalado)"
    );
}

double NBodySystem::computeAccelerationsGpuEndToEnd(
    int /*variant*/,
    int /*block_size*/,
    int /*steps*/,
    double /*dt*/
) {
    throw std::runtime_error(
        "computeAccelerationsGpuEndToEnd: el binario fue compilado "
        "sin soporte CUDA "
        "(configure con -DENABLE_CUDA=ON y el CUDA Toolkit instalado)"
    );
}

#endif // NBODY_HAS_CUDA