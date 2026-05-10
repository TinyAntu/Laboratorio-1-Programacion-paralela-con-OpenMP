#include <gtest/gtest.h> 
#include "NBodySystem.h"

// Prueba de aceleración entre dos cuerpos en el eje x
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
    
    // en el ppt el calculo de la aceleracion esta mal, se nos indica 0.971 pero el valor caluclado real es el siguiente:
    // a = G * m2 * d / (d^2 + eps^2)^(3/2) = 1 * 1 * 1 / (1 + 0.01)^1.5 ≈ 0.9851853368415735
    double expected_ax = 0.9851853368415735; 
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

// Prueba de zeroAccelerations
TEST(NBodyPhysicsTest, ZeroAccelerations) {
    NBodySystem system(1.0, 0.1);
    Particle p1(1.0, 0.0, 0.0);
    p1.setAcceleration(5.0, -3.0);
    system.addParticle(p1);
    
    Particle p2(2.0, 1.0, 1.0);
    p2.setAcceleration(1.0, 1.0);
    system.addParticle(p2);
    
    system.zeroAccelerations();
    
    for (const auto& p : system.getBodies()) {
        EXPECT_DOUBLE_EQ(p.getAx(), 0.0);
        EXPECT_DOUBLE_EQ(p.getAy(), 0.0);
    }
}

// Prueba de tres cuerpos
TEST(NBodyPhysicsTest, AnalyticalThreeBody) {
    double G = 1.0;
    double eps = 0.0; // Sin suavizado para cálculo exacto fácil si no hay colisiones
    NBodySystem system(G, eps);
    
    // Triángulo equilátero
    Particle p1(1.0, 0.0, 1.0);
    Particle p2(1.0, -0.86602540378, -0.5);
    Particle p3(1.0, 0.86602540378, -0.5);
    
    system.addParticle(p1);
    system.addParticle(p2);
    system.addParticle(p3);
    
    system.computeAccelerations();
    
    // Las fuerzas deben apuntar al centroide (0,0) por simetría
    EXPECT_NEAR(system.getBodies()[0].getAx(), 0.0, 1e-5);
    EXPECT_LT(system.getBodies()[0].getAy(), 0.0); // Atraído hacia abajo
}

// Test de regresión: Insensibilidad a una masa no nula
TEST(NBodyPhysicsTest, RegressionMassSensitivity) {
    double G = 1.0;
    double eps = 0.1;
    
    NBodySystem sys_zero(G, eps);
    sys_zero.addParticle(Particle(1.0, 0.0, 0.0));
    sys_zero.addParticle(Particle(0.0, 1.0, 0.0)); // Masa cero
    sys_zero.computeAccelerations();
    
    NBodySystem sys_mass(G, eps);
    sys_mass.addParticle(Particle(1.0, 0.0, 0.0));
    sys_mass.addParticle(Particle(10.0, 1.0, 0.0)); // Masa grande
    sys_mass.computeAccelerations();
    
    // La partícula 0 debe sentir distinta aceleración en ambos sistemas
    double ax_zero = sys_zero.getBodies()[0].getAx();
    double ax_mass = sys_mass.getBodies()[0].getAx();
    
    EXPECT_NE(ax_zero, ax_mass);
    EXPECT_DOUBLE_EQ(ax_zero, 0.0); // Con m2=0, la aceleración de p1 debe ser 0
}

// TEST DE INTEGRACIÓN: Consistencia Serial vs Paralelo
// Verifica que todas las implementaciones de OpenMP den el mismo resultado que la secuencial
TEST(NBodyIntegrationTest, ConsistencySerialVsParallel) {
    double G = 1.0;
    double eps = 0.1;
    int N = 50;
    unsigned int seed = 42;

    // Sistema de referencia (Secuencial)
    NBodySystem serialSystem(G, eps);
    serialSystem.loadFromSeed(seed, N);
    serialSystem.computeAccelerations();

    auto referenceBodies = serialSystem.getBodies();

    // 1. Probar variantes de Schedule (0=static, 1=dynamic, 2=guided)
    for (int sched = 0; sched <= 2; ++sched) {
        NBodySystem parallelSystem(G, eps);
        parallelSystem.loadFromSeed(seed, N);
        parallelSystem.computeAccelerations(sched);

        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR(parallelSystem.getBodies()[i].getAx(), referenceBodies[i].getAx(), 1e-12) 
                << "Falla en Schedule tipo " << sched << " para partícula " << i;
            EXPECT_NEAR(parallelSystem.getBodies()[i].getAy(), referenceBodies[i].getAy(), 1e-12);
        }
    }

    // 2. Probar variante Collapse
    NBodySystem collapseSystem(G, eps);
    collapseSystem.loadFromSeed(seed, N);
    collapseSystem.computeAccelerationsCollapse();
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(collapseSystem.getBodies()[i].getAx(), referenceBodies[i].getAx(), 1e-12)
            << "Falla en implementacion Collapse para particula " << i;
    }

    // 3. Probar variante Newton3
    NBodySystem newtonSystem(G, eps);
    newtonSystem.loadFromSeed(seed, N);
    newtonSystem.computeAccelerationsNewton3();
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(newtonSystem.getBodies()[i].getAx(), referenceBodies[i].getAx(), 1e-12)
            << "Falla en implementacion Newton3 para particula " << i;
    }

    // 4. Probar consistencia con distintos chunk_size
    std::vector<int> test_chunks = {1, 4, 16};
    for (int chunk : test_chunks) {
        NBodySystem chunkSystem(G, eps);
        chunkSystem.loadFromSeed(seed, N);
        chunkSystem.computeAccelerations(1, chunk); // 1 = Dynamic
        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR(chunkSystem.getBodies()[i].getAx(), referenceBodies[i].getAx(), 1e-12)
                << "Falla con chunk_size=" << chunk << " en particula " << i;
        }
    }
    }