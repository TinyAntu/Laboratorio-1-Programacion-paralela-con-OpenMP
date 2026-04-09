#include "MetricsCalculator.h"
#include <stdexcept>
#include <limits>
#include <omp.h>

double MetricsCalculator::calculateKineticEnergy(
        const std::vector<Particle>& bodies) const {
    double K = 0.0;
    for (const auto& p : bodies) {
        double vx = p.getVx();
        double vy = p.getVy();
        K += 0.5 * p.getMass() * (vx * vx + vy * vy);
    }
    return K;
}

double MetricsCalculator::calculatePotentialEnergy(
        const std::vector<Particle>& bodies, double G, double softening) const {
    double U = 0.0;
    int n = static_cast<int>(bodies.size());
    double eps2 = softening * softening;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx   = bodies[j].getX() - bodies[i].getX();
            double dy   = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps2);
            U -= (G * bodies[i].getMass() * bodies[j].getMass()) / dist;
        }
    }
    return U;
}

double MetricsCalculator::calculateTotalEnergy(
        const std::vector<Particle>& bodies, double G, double softening) const {
    return calculateKineticEnergy(bodies) + calculatePotentialEnergy(bodies, G, softening);
}

std::array<double, 2> MetricsCalculator::calculateLinearMomentum(
        const std::vector<Particle>& bodies) const {
    double Px = 0.0, Py = 0.0;
    for (const auto& b : bodies) {
        Px += b.getMass() * b.getVx();
        Py += b.getMass() * b.getVy();
    }
    return {Px, Py};
}

std::array<double, 2> MetricsCalculator::calculateCenterOfMass(
        const std::vector<Particle>& bodies) const {
    double Mx = 0.0, My = 0.0, M = 0.0;
    for (const auto& b : bodies) {
        double m = b.getMass();
        Mx += m * b.getX();
        My += m * b.getY();
        M  += m;
    }
    if (M == 0.0) return {0.0, 0.0};
    return {Mx / M, My / M};
}

double MetricsCalculator::calculateRMSRadius(
        const std::vector<Particle>& bodies) const {
    auto cm  = calculateCenterOfMass(bodies);
    double M = 0.0, rms2 = 0.0;
    for (const auto& b : bodies) {
        double m  = b.getMass();
        double dx = b.getX() - cm[0];
        double dy = b.getY() - cm[1];
        rms2 += m * (dx * dx + dy * dy);
        M    += m;
    }
    if (M == 0.0) return 0.0;
    return std::sqrt(rms2 / M);
}

double MetricsCalculator::calculateMinDistance(
        const std::vector<Particle>& bodies) const {
    int n = static_cast<int>(bodies.size());
    double dmin = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double d  = std::sqrt(dx * dx + dy * dy);
            if (d < dmin) dmin = d;
        }
    }
    return dmin;
}

SystemMetrics MetricsCalculator::calculateAll(
        const std::vector<Particle>& bodies, double G, double softening) const {
    SystemMetrics m;
    m.kineticEnergy   = calculateKineticEnergy(bodies);
    m.potentialEnergy = calculatePotentialEnergy(bodies, G, softening);
    m.totalEnergy     = m.kineticEnergy + m.potentialEnergy;

    auto mom      = calculateLinearMomentum(bodies);
    m.momentumX   = mom[0];
    m.momentumY   = mom[1];
    m.momentumMag = std::sqrt(mom[0] * mom[0] + mom[1] * mom[1]);

    auto cm         = calculateCenterOfMass(bodies);
    m.centerOfMassX = cm[0];
    m.centerOfMassY = cm[1];

    m.rmsRadius   = calculateRMSRadius(bodies);
    m.minDistance = calculateMinDistance(bodies);
    return m;
}

double MetricsCalculator::calculateKineticEnergyParallel(
        const std::vector<Particle>& bodies, int method) const {
    int n = static_cast<int>(bodies.size());
    double K = 0.0;

    if (method == 0) {
        #pragma omp parallel for schedule(static) reduction(+:K)
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx();
            double vy = bodies[i].getVy();
            K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
        }
    } else if (method == 1) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx();
            double vy = bodies[i].getVy();
            double contrib = 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
            #pragma omp atomic
            K += contrib;
        }
    } else {
        throw std::invalid_argument("method invalido: use 0=reduce, 1=atomic");
    }
    return K;
}

double MetricsCalculator::calculatePotentialEnergyParallel(
        const std::vector<Particle>& bodies,
        double G, double softening, int method) const {
    int n    = static_cast<int>(bodies.size());
    double U = 0.0;
    double eps2 = softening * softening;

    if (method == 0) {
        #pragma omp parallel for schedule(dynamic) reduction(+:U)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx   = bodies[j].getX() - bodies[i].getX();
                double dy   = bodies[j].getY() - bodies[i].getY();
                double dist = std::sqrt(dx * dx + dy * dy + eps2);
                U -= G * bodies[i].getMass() * bodies[j].getMass() / dist;
            }
        }
    } else if (method == 1) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx     = bodies[j].getX() - bodies[i].getX();
                double dy     = bodies[j].getY() - bodies[i].getY();
                double dist   = std::sqrt(dx * dx + dy * dy + eps2);
                double contrib = -G * bodies[i].getMass() * bodies[j].getMass() / dist;
                #pragma omp atomic
                U += contrib;
            }
        }
    } else {
        throw std::invalid_argument("method invalido: use 0=reduce, 1=atomic");
    }
    return U;
}

double MetricsCalculator::calculateKineticEnergyParallel(
        const std::vector<Particle>& bodies, int method, bool use_private) const {
    if (!use_private) {
        return calculateKineticEnergyParallel(bodies, method);
    }

    int n    = static_cast<int>(bodies.size());
    double K = 0.0;

    #pragma omp parallel shared(K)
    {
        double K_local = 0.0;

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx();
            double vy = bodies[i].getVy();
            K_local += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
        }

        #pragma omp critical
        {
            K += K_local;
        }
    }
    return K;
}

SystemMetrics MetricsCalculator::calculateMetricsFirstprivate(
        const std::vector<Particle>& bodies, double G, double softening) const {
    int n = static_cast<int>(bodies.size());

    double K_init  = 0.0;
    double Px_init = 0.0;
    double Py_init = 0.0;
    double K = 0.0, Px = 0.0, Py = 0.0;

    #pragma omp parallel firstprivate(K_init, Px_init, Py_init) shared(K, Px, Py)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            double vx = bodies[i].getVx();
            double vy = bodies[i].getVy();
            K_init  += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);
            Px_init += bodies[i].getMass() * vx;
            Py_init += bodies[i].getMass() * vy;
        }

        #pragma omp critical
        {
            K  += K_init;
            Px += Px_init;
            Py += Py_init;
        }
    }

    double U    = calculatePotentialEnergy(bodies, G, softening);
    auto cm     = calculateCenterOfMass(bodies);
    double rms  = calculateRMSRadius(bodies);
    double dmin = calculateMinDistance(bodies);

    SystemMetrics m;
    m.kineticEnergy   = K;
    m.potentialEnergy = U;
    m.totalEnergy     = K + U;
    m.momentumX       = Px;
    m.momentumY       = Py;
    m.momentumMag     = std::sqrt(Px * Px + Py * Py);
    m.centerOfMassX   = cm[0];
    m.centerOfMassY   = cm[1];
    m.rmsRadius       = rms;
    m.minDistance     = dmin;
    return m;
}

SystemMetrics MetricsCalculator::calculateFinalStateLastprivate(
        const std::vector<Particle>& bodies, double G, double softening) const {
    int n = static_cast<int>(bodies.size());

    double last_x    = 0.0;
    double last_y    = 0.0;
    double last_mass = 0.0;
    double K         = 0.0;

    #pragma omp parallel for schedule(static)       \
            reduction(+:K)                          \
            lastprivate(last_x, last_y, last_mass)
    for (int i = 0; i < n; ++i) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        K += 0.5 * bodies[i].getMass() * (vx * vx + vy * vy);

        last_x    = bodies[i].getX();
        last_y    = bodies[i].getY();
        last_mass = bodies[i].getMass();
    }

    double U    = calculatePotentialEnergy(bodies, G, softening);
    auto cm     = calculateCenterOfMass(bodies);
    double rms  = calculateRMSRadius(bodies);
    double dmin = calculateMinDistance(bodies);
    auto mom    = calculateLinearMomentum(bodies);

    (void)last_x; (void)last_y; (void)last_mass;

    SystemMetrics m;
    m.kineticEnergy   = K;
    m.potentialEnergy = U;
    m.totalEnergy     = K + U;
    m.momentumX       = mom[0];
    m.momentumY       = mom[1];
    m.momentumMag     = std::sqrt(mom[0]*mom[0] + mom[1]*mom[1]);
    m.centerOfMassX   = cm[0];
    m.centerOfMassY   = cm[1];
    m.rmsRadius       = rms;
    m.minDistance     = dmin;
    return m;
}
