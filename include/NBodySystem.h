//Conjunto de partículas, G, epsilon; computeAccelerations*

#ifndef NBODYSYSTEM_H
#define NBODYSYSTEM_H

#include <vector>
#include "Particle.h" //System es el contenedor de las Particulas
#include <string>
#include <utility>

class NBodySystem {
private:
    std::vector<Particle> bodies; //Contenedor de partículas
    double G_const;
    double softening_eps;

    // Lab2: estado GPU opaco (buffers SoA en device). Se define en
    // src/NBodySystemGpu.cu para que este header no dependa de tipos CUDA
    // en builds solo-CPU. Se crea perezosamente en la primera llamada GPU.
    struct GpuState;
    GpuState* gpu_state = nullptr;

public:
    NBodySystem(double G, double epsilon);
    ~NBodySystem();

    // Dueño único del estado GPU: no copiable
    NBodySystem(const NBodySystem&) = delete;
    NBodySystem& operator=(const NBodySystem&) = delete;

    void addParticle(const Particle& p);
    void zeroAccelerations();

    //Uso de semillas
    void loadFromSeed(unsigned int seed, int N);

    // I/O
    void writePositions(const std::string& filename);

    // Sobrecarga: c ́alculo de aceleraciones con distintos schedules / collapse
    void computeAccelerations();
    void computeAccelerations(int schedule_type);
    void computeAccelerations(int schedule_type, int chunk_size);
    void computeAccelerationsCollapse(); // p.ej. collapse(2) en i,j
    void computeAccelerationsNewton3(); // Implementación con tercera ley de Newton

    // Asume un block_size por defecto (ej: 256) y variante por defecto (0 = básica)
    void computeAccelerationsGpu();
    
    // Variante: 0 = basico, 1 = shared memory (block_size por defecto = 256)
    void computeAccelerationsGpu(int variant);
    
    // Control total sobre la variante y el tamaño del bloque CUDA
    void computeAccelerationsGpu(int variant, int block_size);
    
    // Métodos de cronometraje para benchmarks GPU.
    // steps: lanzamientos consecutivos a promediar. Debe coincidir con el steps
    // de computeAccelerationsGpuEndToEnd para que la resta (e2e - kernel) mida
    // transferencias + trabajo en host y no ruido de muestreo.
    double computeAccelerationsGpuKernelOnly(int variant, int block_size = 256,
                                             int steps = 100);

    double computeAccelerationsGpuEndToEnd(
        int variant,
        int block_size,
        int steps,
        double dt
    );
    const std::vector<Particle>& getBodies() const;         // version const para referenciar sin modificar las partículas
    std::vector<Particle>& getBodies();                     // version no const para modificar las partículas
    
    std::pair<double, double> computeEnergyGpu(
        int method,
        int block_size = 256
    );
    
    // Getters de constantes físicas
    int getCount() const;
    double getG() const;
    double getSoftening() const;
};

#endif // NBODYSYSTEM_H