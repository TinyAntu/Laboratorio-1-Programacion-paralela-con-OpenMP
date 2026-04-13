#include <gtest/gtest.h> 
#include "NBodySystem.h"

// Prueba de aceleración entre dos cuerpos 
TEST(NBodyPhysicsTest, AnalyticalTwoBody) {
    double G = 1.0;
    double eps = 0.1;
    NBodySystem system(G, eps);
    
    // Configuración: m1 en (0,0), m2 en (1,0) con m2=1
    Particle p1(1.0, 0.0, 0.0);
    Particle p2(1.0, 1.0, 0.0);
    
    system.addParticle(p1);
    system.addParticle(p2);
    
    system.computeAccelerations(); // Tu método secuencial
    
    // a = G * m2 * d / (d^2 + eps^2)^(3/2) = 1 * 1 * 1 / (1 + 0.01)^1.5 ≈ 0.971
    double expected_ax = 0.971; 
    EXPECT_NEAR(system.getBodies()[0].getAx(), expected_ax, 0.001);
}

// Prueba Tercera Ley de Newton 
TEST(NBodyPhysicsTest, ActionReaction) {
    NBodySystem system(1.0, 0.01);
    system.loadFromSeed(1234, 10); // 10 partículas aleatorias 
    
    system.computeAccelerations();
    
    double total_force_x = 0;
    double total_force_y = 0;
    
    for(const auto& p : system.getBodies()) {
        total_force_x += p.getMass() * p.getAx();
        total_force_y += p.getMass() * p.getAy();
    }
    
    // La suma de todas las fuerzas internas debe ser cero 
    EXPECT_NEAR(total_force_x, 0.0, 1e-9);
    EXPECT_NEAR(total_force_y, 0.0, 1e-9);
}