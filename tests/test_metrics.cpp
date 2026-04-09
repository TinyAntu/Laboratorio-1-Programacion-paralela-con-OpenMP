#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "Particle.h"
#include "MetricsCalculator.h"

//  Utilidades
static bool approxEqual(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

static void pass(const char* name) {
    std::cout << "  [PASS] " << name << "\n";
}

//  1. Energía cinética
static void test_kineticEnergy() {
    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.back().setVx(3.0);
    bodies.back().setVy(4.0);
    bodies.emplace_back(1.0, 1.0, 0.0);
    bodies.back().setVx(0.0);
    bodies.back().setVy(0.0);

    MetricsCalculator mc;
    double K = mc.calculateKineticEnergy(bodies);
    assert(approxEqual(K, 25.0));
    pass("kineticEnergy — dos cuerpos, valores exactos");

    std::vector<Particle> rest;
    rest.emplace_back(5.0, 1.0, 2.0);  // velocidad default 0,0
    assert(approxEqual(mc.calculateKineticEnergy(rest), 0.0));
    pass("kineticEnergy — sistema en reposo");
}

//  2. Energía potencial
static void test_potentialEnergy() {

    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.emplace_back(1.0, 3.0, 4.0);

    MetricsCalculator mc;
    double U = mc.calculatePotentialEnergy(bodies, 1.0, 0.0);
    assert(approxEqual(U, -0.4));
    pass("potentialEnergy — dos cuerpos sin suavizado");

    double eps  = 0.1;
    double dist = std::sqrt(25.0 + eps * eps);
    double U_eps = mc.calculatePotentialEnergy(bodies, 1.0, eps);
    assert(approxEqual(U_eps, -2.0 / dist));
    pass("potentialEnergy — dos cuerpos con suavizado");

    std::vector<Particle> solo;
    solo.emplace_back(3.0, 0.0, 0.0);
    assert(approxEqual(mc.calculatePotentialEnergy(solo, 1.0, 0.0), 0.0));
    pass("potentialEnergy — un solo cuerpo");
}

//  3. Energía total
static void test_totalEnergy() {
    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.back().setVx(3.0);
    bodies.back().setVy(4.0);
    bodies.emplace_back(1.0, 3.0, 4.0);

    MetricsCalculator mc;
    double E = mc.calculateTotalEnergy(bodies, 1.0, 0.0);
    assert(approxEqual(E, 25.0 - 0.4));
    pass("totalEnergy = K + U");
}

//  4. Momento lineal
static void test_linearMomentum() {

    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.back().setVx(3.0);
    bodies.back().setVy(4.0);
    bodies.emplace_back(1.0, 1.0, 0.0);
    bodies.back().setVx(1.0);
    bodies.back().setVy(2.0);

    MetricsCalculator mc;
    auto P = mc.calculateLinearMomentum(bodies);
    assert(approxEqual(P[0], 7.0));
    assert(approxEqual(P[1], 10.0));
    pass("linearMomentum — dos cuerpos");

    std::vector<Particle> rest;
    rest.emplace_back(5.0, 0.0, 0.0);
    auto Pr = mc.calculateLinearMomentum(rest);
    assert(approxEqual(Pr[0], 0.0));
    assert(approxEqual(Pr[1], 0.0));
    pass("linearMomentum — sistema en reposo");
}

//  5. Centro de masas
static void test_centerOfMass() {

    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.emplace_back(1.0, 3.0, 4.0);

    MetricsCalculator mc;
    auto cm = mc.calculateCenterOfMass(bodies);
    assert(approxEqual(cm[0], 1.0));
    assert(approxEqual(cm[1], 4.0 / 3.0));
    pass("centerOfMass — dos cuerpos");

    std::vector<Particle> zero_mass;
    zero_mass.emplace_back(0.0, 5.0, 5.0);
    auto cmz = mc.calculateCenterOfMass(zero_mass);
    assert(approxEqual(cmz[0], 0.0));
    assert(approxEqual(cmz[1], 0.0));
    pass("centerOfMass — masa total cero, no crash");
}

//  6. Radio RMS
static void test_rmsRadius() {

    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, -1.0, 0.0);
    bodies.emplace_back(1.0,  1.0, 0.0);

    MetricsCalculator mc;
    double rms = mc.calculateRMSRadius(bodies);
    assert(approxEqual(rms, 1.0));
    pass("rmsRadius — dos masas iguales simétricas");

    std::vector<Particle> solo;
    solo.emplace_back(3.0, 7.0, 7.0);
    assert(approxEqual(mc.calculateRMSRadius(solo), 0.0));
    pass("rmsRadius — un solo cuerpo");
}

//  7. Distancia mínima
static void test_minDistance() {

    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 0.0, 0.0);
    bodies.emplace_back(1.0, 3.0, 4.0);
    bodies.emplace_back(1.0, 1.0, 0.0);

    MetricsCalculator mc;
    double dmin = mc.calculateMinDistance(bodies);
    assert(approxEqual(dmin, 1.0));
    pass("minDistance — tres cuerpos, mínimo correcto");

    std::vector<Particle> two;
    two.emplace_back(1.0, 0.0, 0.0);
    two.emplace_back(1.0, 3.0, 4.0);
    assert(approxEqual(mc.calculateMinDistance(two), 5.0));
    pass("minDistance — dos cuerpos distancia 5");
}

//  8. calculateAll — consistencia interna
static void test_calculateAll() {
    std::vector<Particle> bodies;
    bodies.emplace_back(2.0, 0.0, 0.0);
    bodies.back().setVx(3.0);
    bodies.back().setVy(4.0);
    bodies.emplace_back(1.0, 3.0, 4.0);

    MetricsCalculator mc;
    SystemMetrics m = mc.calculateAll(bodies, 1.0, 0.0);

    assert(approxEqual(m.kineticEnergy,   mc.calculateKineticEnergy(bodies)));
    assert(approxEqual(m.potentialEnergy, mc.calculatePotentialEnergy(bodies, 1.0, 0.0)));
    assert(approxEqual(m.totalEnergy,     m.kineticEnergy + m.potentialEnergy));
    assert(approxEqual(m.momentumMag,     std::sqrt(m.momentumX*m.momentumX + m.momentumY*m.momentumY)));
    assert(approxEqual(m.minDistance,     mc.calculateMinDistance(bodies)));
    pass("calculateAll — coherencia interna entre campos");
}

//  9. Versiones paralelas — resultados iguales a serial
static void test_parallelConsistency() {

    std::vector<Particle> bodies;
    for (int i = 0; i < 20; ++i) {
        bodies.emplace_back(1.0 + i * 0.1,
                            i * 0.5,
                            i * 0.3);
        bodies.back().setVx(i * 0.2);
        bodies.back().setVy(-i * 0.1);
    }

    MetricsCalculator mc;
    double K_serial = mc.calculateKineticEnergy(bodies);

    double K_par0 = mc.calculateKineticEnergyParallel(bodies, 0);
    assert(approxEqual(K_serial, K_par0, 1e-10));
    pass("parallelKinetic method=0 (reduction) == serial");

    double K_par1 = mc.calculateKineticEnergyParallel(bodies, 1);
    assert(approxEqual(K_serial, K_par1, 1e-10));
    pass("parallelKinetic method=1 (atomic) == serial");

    double K_priv = mc.calculateKineticEnergyParallel(bodies, 0, true);
    assert(approxEqual(K_serial, K_priv, 1e-10));
    pass("parallelKinetic use_private=true == serial");

    double U_serial = mc.calculatePotentialEnergy(bodies, 1.0, 0.01);
    double U_par0   = mc.calculatePotentialEnergyParallel(bodies, 1.0, 0.01, 0);
    double U_par1   = mc.calculatePotentialEnergyParallel(bodies, 1.0, 0.01, 1);
    assert(approxEqual(U_serial, U_par0, 1e-9));
    assert(approxEqual(U_serial, U_par1, 1e-9));
    pass("parallelPotential method=0 y 1 == serial (tol 1e-9)");
}

//  10. firstprivate / lastprivate — coherencia con calculateAll
static void test_firstprivateLastprivate() {
    std::vector<Particle> bodies;
    for (int i = 0; i < 15; ++i) {
        bodies.emplace_back(1.0, i * 1.0, 0.0);
        bodies.back().setVx(0.1 * i);
        bodies.back().setVy(0.0);
    }

    MetricsCalculator mc;
    double G = 1.0, eps = 0.05;

    SystemMetrics ref  = mc.calculateAll(bodies, G, eps);
    SystemMetrics mfp  = mc.calculateMetricsFirstprivate(bodies, G, eps);
    SystemMetrics mlp  = mc.calculateFinalStateLastprivate(bodies, G, eps);

    assert(approxEqual(ref.kineticEnergy, mfp.kineticEnergy,  1e-9));
    assert(approxEqual(ref.kineticEnergy, mlp.kineticEnergy,  1e-9));

    assert(approxEqual(ref.momentumX, mfp.momentumX, 1e-9));
    assert(approxEqual(ref.momentumY, mfp.momentumY, 1e-9));

    assert(approxEqual(ref.totalEnergy, mfp.totalEnergy, 1e-9));
    assert(approxEqual(ref.totalEnergy, mlp.totalEnergy, 1e-9));

    pass("firstprivate — K, Px, Py coinciden con referencia serial");
    pass("lastprivate  — K, totalEnergy coinciden con referencia serial");
}

int main() {
    std::cout << "Test de MetricsCalculator\n";

    test_kineticEnergy();
    test_potentialEnergy();
    test_totalEnergy();
    test_linearMomentum();
    test_centerOfMass();
    test_rmsRadius();
    test_minDistance();
    test_calculateAll();
    test_parallelConsistency();
    test_firstprivateLastprivate();

    std::cout << "\nTodos los tests de MetricsCalculator funcionaron correctamente.\n";
    return 0;
}
