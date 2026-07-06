//Conjunto de partículas, G, epsilon; computeAccelerations*

#ifndef NBODYSYSTEM_H
#define NBODYSYSTEM_H

#include <vector>
#include "Particle.h" //System es el contenedor de las Particulas
#include <string>

class NBodySystem {
private:
    std::vector<Particle> bodies; //Contenedor de partículas
    double G_const;
    double softening_eps;

public:
    NBodySystem(double G, double epsilon);
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
    
    const std::vector<Particle>& getBodies() const;         // version const para referenciar sin modificar las partículas
    std::vector<Particle>& getBodies();                     // version no const para modificar las partículas
    
    // Getters de constantes físicas
    int getCount() const; 
    double getG() const;
    double getSoftening() const;

    // Metodos Cuda Lab2
    void computeAccelerationsGPU();
    void computeAccelerationsGPU(int variant); // 0 = basico, 1 = shared memory
    void computeAccelerationsGPU(int variant, int block_size); // 0 = basico, 1 = shared memory
};

#endif // NBODYSYSTEM_H