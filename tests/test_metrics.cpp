#include <gtest/gtest.h>
#include "MetricsCalculator.h"
#include "Particle.h"
#include <vector>
#include <cmath>

class MetricsTest : public ::testing::Test {
protected:
    MetricsCalculator calc;
    double G = 1.0;
    double softening = 0.1;
};

// Test de Energía Cinética: K = 1/2 * m * v^2
TEST_F(MetricsTest, KineticEnergySimple) {
    std::vector<Particle> bodies;
    Particle p1(2.0, 0.0, 0.0); // m=2
    p1.setVx(1.0); // v=1 -> K = 0.5 * 2 * 1^2 = 1.0
    
    Particle p2(4.0, 0.0, 0.0); // m=4
    p2.setVy(2.0); // v=2 -> K = 0.5 * 4 * 2^2 = 8.0
    
    bodies.push_back(p1);
    bodies.push_back(p2);
    
    double expectedK = 1.0 + 8.0;
    EXPECT_NEAR(calc.calculateKineticEnergy(bodies), expectedK, 1e-10);
}

// Test de Energía Potencial: U = -G * m1 * m2 / sqrt(r^2 + eps^2)
TEST_F(MetricsTest, PotentialEnergySimple) {
    std::vector<Particle> bodies;
    // Distancia r = 1.0 en eje X
    Particle p1(1.0, 0.0, 0.0);
    Particle p2(1.0, 1.0, 0.0);
    bodies.push_back(p1);
    bodies.push_back(p2);
    
    double r2 = 1.0;
    double eps2 = softening * softening;
    double expectedU = -(G * 1.0 * 1.0) / std::sqrt(r2 + eps2);
    
    EXPECT_NEAR(calc.calculatePotentialEnergy(bodies, G, softening), expectedU, 1e-10);
}

// Test de Momento Lineal: P = sum(m * v)
TEST_F(MetricsTest, LinearMomentum) {
    std::vector<Particle> bodies;
    Particle p1(1.0, 0.0, 0.0);
    p1.setVx(10.0);
    p1.setVy(5.0);
    
    Particle p2(2.0, 0.0, 0.0);
    p2.setVx(-2.0);
    p2.setVy(0.0);
    
    bodies.push_back(p1);
    bodies.push_back(p2);
    
    // Px = 1*10 + 2*(-2) = 6.0
    // Py = 1*5 + 2*0 = 5.0
    auto P = calc.calculateLinearMomentum(bodies);
    EXPECT_NEAR(P[0], 6.0, 1e-10);
    EXPECT_NEAR(P[1], 5.0, 1e-10);
}

// Test de Centro de Masa: R = sum(m * r) / sum(m)
TEST_F(MetricsTest, CenterOfMass) {
    std::vector<Particle> bodies;
    bodies.push_back(Particle(1.0, 0.0, 0.0));
    bodies.push_back(Particle(3.0, 4.0, 0.0));
    
    // CM_x = (1*0 + 3*4) / 4 = 3.0
    // CM_y = (1*0 + 3*0) / 4 = 0.0
    auto CM = calc.calculateCenterOfMass(bodies);
    EXPECT_NEAR(CM[0], 3.0, 1e-10);
    EXPECT_NEAR(CM[1], 0.0, 1e-10);
}

// Test de Consistencia Serial vs Paralelo para Métricas
TEST_F(MetricsTest, ParallelConsistency) {
    std::vector<Particle> bodies;
    // Crear algunas partículas aleatorias
    for(int i=0; i<50; ++i) {
        Particle p(1.0, (double)i, (double)(i*i));
        p.setVx(0.1 * i);
        p.setVy(-0.1 * i);
        bodies.push_back(p);
    }
    
    double serialK = calc.calculateKineticEnergy(bodies);
    double parallelK_reduce = calc.calculateKineticEnergyParallel(bodies, 0); // reduction
    double parallelK_atomic = calc.calculateKineticEnergyParallel(bodies, 1); // atomic
    
    EXPECT_NEAR(serialK, parallelK_reduce, 1e-10);
    EXPECT_NEAR(serialK, parallelK_atomic, 1e-10);
    
    double serialU = calc.calculatePotentialEnergy(bodies, G, softening);
    double parallelU_reduce = calc.calculatePotentialEnergyParallel(bodies, G, softening, 0);
    double parallelU_atomic = calc.calculatePotentialEnergyParallel(bodies, G, softening, 1);
    
    EXPECT_NEAR(serialU, parallelU_reduce, 1e-10);
    EXPECT_NEAR(serialU, parallelU_atomic, 1e-10);
}
