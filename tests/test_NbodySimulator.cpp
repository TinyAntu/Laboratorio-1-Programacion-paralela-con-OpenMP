#include <gtest/gtest.h> 
#include "NBodySimulator.h"
#include "Particle.h"
#include <cmath>
#include <vector>

class NBodySimulatorIntegrationTest : public ::testing::Test {
protected:
    const int N = 20;
    const unsigned int seed = 1234;
    const double G = 1.0;
    const double softening = 0.05;
    const double dt = 0.001;

    void compareSimulators(const NBodySimulator& sim1, const NBodySimulator& sim2, double tolerance = 1e-10) {
        const auto& bodies1 = sim1.getBodies();
        const auto& bodies2 = sim2.getBodies();
        
        ASSERT_EQ(bodies1.size(), bodies2.size());
        for (size_t i = 0; i < bodies1.size(); ++i) {
            EXPECT_NEAR(bodies1[i].getX(), bodies2[i].getX(), tolerance) << "Mismatch in X at particle " << i;
            EXPECT_NEAR(bodies1[i].getY(), bodies2[i].getY(), tolerance) << "Mismatch in Y at particle " << i;
            EXPECT_NEAR(bodies1[i].getVx(), bodies2[i].getVx(), tolerance) << "Mismatch in Vx at particle " << i;
            EXPECT_NEAR(bodies1[i].getVy(), bodies2[i].getVy(), tolerance) << "Mismatch in Vy at particle " << i;
        }
    }
};

// Test que valida que todas las variantes de integrateEuler coinciden con la secuencial
TEST_F(NBodySimulatorIntegrationTest, ConsistencyOfAllEulerVariants) {
    // 1. Referencia Secuencial
    NBodySimulator serialSim(N, seed, G, softening, dt);
    serialSim.integrateEuler();

    // 2. Variante Schedule
    NBodySimulator schedSim(N, seed, G, softening, dt);
    schedSim.integrateEulerSchedule();
    compareSimulators(serialSim, schedSim);

    // 3. Variante Collapse
    NBodySimulator collapseSim(N, seed, G, softening, dt);
    collapseSim.integrateEulerCollapse();
    compareSimulators(serialSim, collapseSim);

    // 4. Variante Newton3
    NBodySimulator newtonSim(N, seed, G, softening, dt);
    newtonSim.integrateEulerNewton3();
    compareSimulators(serialSim, newtonSim);
}

// Test que valida las variantes de sincronización (Atomic, Critical, Nowait)
TEST_F(NBodySimulatorIntegrationTest, ConsistencyOfSyncVariants) {
    NBodySimulator serialSim(N, seed, G, softening, dt);
    serialSim.integrateEuler();

    // sync_type: 0=atomic, 1=critical, 2=nowait
    for (int sync = 0; sync <= 2; ++sync) {
        NBodySimulator syncSim(N, seed, G, softening, dt);
        syncSim.integrateEuler(sync);
        compareSimulators(serialSim, syncSim) << "Failed for sync_type=" << sync;
    }
}

// Test de conservación del Momento Lineal
TEST_F(NBodySimulatorIntegrationTest, LinearMomentumConservation) {
    NBodySimulator sim(N, seed, G, softening, dt);
    
    auto calculateP = [](const std::vector<Particle>& bodies) {
        double px = 0, py = 0;
        for (const auto& p : bodies) {
            px += p.getMass() * p.getVx();
            py += p.getMass() * p.getVy();
        }
        return std::make_pair(px, py);
    };

    auto P_initial = calculateP(sim.getBodies());
    
    // Simular 10 pasos
    for(int i=0; i<10; ++i) {
        sim.integrateEuler();
    }

    auto P_final = calculateP(sim.getBodies());

    // El momento lineal debe conservarse en un sistema cerrado
    EXPECT_NEAR(P_initial.first, P_final.first, 1e-10);
    EXPECT_NEAR(P_initial.second, P_final.second, 1e-10);
}

// TEST: Cláusulas Avanzadas de OpenMP (Barrier y Phases)
TEST_F(NBodySimulatorIntegrationTest, AdvancedOpenMPPhases) {
    NBodySimulator simSerial(N, seed, G, softening, dt);
    simSerial.integrateEuler(); // Un paso secuencial

    NBodySimulator simBarrier(N, seed, G, softening, dt);
    simBarrier.simulatePhasesBarrier(); // Un paso con barreras explicitas

    compareSimulators(simSerial, simBarrier);
}

// TEST: Cláusulas Single y Firstprivate/Lastprivate
TEST_F(NBodySimulatorIntegrationTest, AdvancedOpenMPClausulas) {
    NBodySimulator sim(N, seed, G, softening, dt);
    
    // 1. Probar parallelInitializationSingle
    // Esto hace un kick/drift tras calcular CM en bloque single
    sim.parallelInitializationSingle();
    EXPECT_GT(sim.getBodies()[0].getX(), -100.0); // Verificación básica de que no explotó

    // 2. Probar calculateMetricsFirstprivate
    sim.calculateMetricsFirstprivate();
    double K_fp = sim.getSystemEnergy().first;
    
    // 3. Probar calculateFinalStateLastprivate
    sim.calculateFinalStateLastprivate();
    double K_lp = sim.getSystemEnergy().first;

    // Ambas deben dar la misma energía cinética que la versión secuencial
    sim.calculateEnergy();
    double K_serial = sim.getSystemEnergy().first;

    EXPECT_NEAR(K_fp, K_serial, 1e-10);
    EXPECT_NEAR(K_lp, K_serial, 1e-10);
}

