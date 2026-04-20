#include <gtest/gtest.h>
#include "Integrator.h"
#include <cmath>

class IntegratorTest : public ::testing::Test {
protected:
    const int N = 5;
    const unsigned int seed = 42;
    const double dt = 0.01;
    const double G = 1.0;
    const double softening = 0.1;
    const int steps = 10;
};

// Verifica que el historial de estados tenga el tamaño correcto
TEST_F(IntegratorTest, HistorySize) {
    Integrator integrator(N, seed, dt, G, softening, steps);
    integrator.runSimulation();
    
    // Integrator guarda el estado inicial + 'steps' integraciones
    // Nota: Según src/Integrator.cpp, el bucle es <= total_steps, lo que hace steps + 1 iteraciones.
    // +1 estado inicial = steps + 2 estados totales.
    EXPECT_EQ(integrator.getStateHistory().size(), steps + 2);
    EXPECT_EQ(integrator.getEnergyHistory().size(), steps + 1);
}

// Verifica que la simulación paralela sea consistente con la secuencial en una trayectoria
TEST_F(IntegratorTest, TrajectoryConsistency) {
    const int short_steps = 5;
    
    // 1. Simulación Secuencial
    Integrator serialInt(N, seed, dt, G, softening, short_steps);
    serialInt.runSimulation();
    auto serialHistory = serialInt.getStateHistory();

    // 2. Simulación Newton3 (Paralela)
    Integrator parallelInt(N, seed, dt, G, softening, short_steps);
    parallelInt.runSimulationNewton3();
    auto parallelHistory = parallelInt.getStateHistory();

    ASSERT_EQ(serialHistory.size(), parallelHistory.size());

    // Comparar estado final
    const auto& finalSerial = serialHistory.back();
    const auto& finalParallel = parallelHistory.back();

    for(size_t i=0; i<finalSerial.size(); ++i) {
        EXPECT_NEAR(finalSerial[i].getX(), finalParallel[i].getX(), 1e-10);
        EXPECT_NEAR(finalSerial[i].getY(), finalParallel[i].getY(), 1e-10);
    }
}

// Verifica que la energía se calcule y almacene en cada paso
TEST_F(IntegratorTest, EnergyRecording) {
    Integrator integrator(N, seed, dt, G, softening, 5);
    integrator.runSimulation();
    
    auto energyHist = integrator.getEnergyHistory();
    for (const auto& e : energyHist) {
        // La energía cinética debe ser >= 0
        EXPECT_GE(e.first, 0.0);
        // La energía potencial debe ser <= 0 (atractiva)
        EXPECT_LE(e.second, 0.0);
    }
}
