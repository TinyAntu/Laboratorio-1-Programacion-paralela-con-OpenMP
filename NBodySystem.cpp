#include "NBodySystem.h"
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

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

int NBodySystem::getCount() const {
    return bodies.size();
}

