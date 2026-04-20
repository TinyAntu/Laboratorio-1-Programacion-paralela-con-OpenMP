#include <gtest/gtest.h>
#include "Particle.h"

// Test constructor e inicialización
TEST(ParticleTest, Constructor) {
    double mass = 1.5;
    double x = 10.0;
    double y = -5.0;
    Particle p(mass, x, y);

    EXPECT_DOUBLE_EQ(p.getMass(), mass);
    EXPECT_DOUBLE_EQ(p.getX(), x);
    EXPECT_DOUBLE_EQ(p.getY(), y);
    EXPECT_DOUBLE_EQ(p.getVx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getVy(), 0.0);
    EXPECT_DOUBLE_EQ(p.getAx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getAy(), 0.0);
}

// Test manipulación de aceleración
TEST(ParticleTest, AccelerationMethods) {
    Particle p(1.0, 0.0, 0.0);

    p.setAcceleration(1.0, 2.0);
    EXPECT_DOUBLE_EQ(p.getAx(), 1.0);
    EXPECT_DOUBLE_EQ(p.getAy(), 2.0);

    p.addAcceleration(0.5, -1.0);
    EXPECT_DOUBLE_EQ(p.getAx(), 1.5);
    EXPECT_DOUBLE_EQ(p.getAy(), 1.0);

    p.resetAcceleration();
    EXPECT_DOUBLE_EQ(p.getAx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getAy(), 0.0);
}

// Test integración Euler: kick (v += a*dt)
TEST(ParticleTest, KickMethod) {
    Particle p(1.0, 0.0, 0.0);
    p.setAcceleration(2.0, -1.0);
    double dt = 0.1;

    p.kick(dt);
    EXPECT_NEAR(p.getVx(), 0.2, 1e-9);
    EXPECT_NEAR(p.getVy(), -0.1, 1e-9);

    // Kick acumulativo
    p.kick(dt);
    EXPECT_NEAR(p.getVx(), 0.4, 1e-9);
    EXPECT_NEAR(p.getVy(), -0.2, 1e-9);
}

// Test integración Euler: drift (r += v*dt)
TEST(ParticleTest, DriftMethod) {
    Particle p(1.0, 1.0, 1.0);
    p.setVx(2.0);
    p.setVy(-3.0);
    double dt = 0.5;

    p.drift(dt);
    EXPECT_NEAR(p.getX(), 2.0, 1e-9); 
    EXPECT_NEAR(p.getY(), -0.5, 1e-9); 
}

// Test de masa no negativa
TEST(ParticleTest, NonNegativeMass) {
    // Caso válido
    EXPECT_NO_THROW(Particle p1(1.0, 0.0, 0.0));
    
    // Caso límite (masa cero permitida)
    EXPECT_NO_THROW(Particle p0(0.0, 0.0, 0.0));

    // Caso inválido (debe lanzar excepción)
    EXPECT_THROW(Particle p2(-1.0, 0.0, 0.0), std::invalid_argument);
}
