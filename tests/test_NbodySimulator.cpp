#include <gtest/gtest.h> 
#include "../NBodySimulator.h"
#include "../Particle.h"
#include <cmath>

class ProcessBodiesTest : public ::testing::Test {
protected:
    NBodySimulator* simulator1;
    NBodySimulator* simulator2;

    void SetUp() override {
        // Create two identical simulators with same seed
        unsigned int seed = 42;
        int N = 10; // Small number for testing
        double G = 1.0;
        double softening = 0.1;
        double dt = 0.01;

        simulator1 = new NBodySimulator(N, seed, G, softening, dt);
        simulator2 = new NBodySimulator(N, seed, G, softening, dt);
    }

    void TearDown() override {
        delete simulator1;
        delete simulator2;
    }

    bool particlesEqual(const Particle& p1, const Particle& p2, double tolerance = 1e-10) {
        // Compare positions (x, y)
        if (std::abs(p1.getX() - p2.getX()) > tolerance) return false;
        if (std::abs(p1.getY() - p2.getY()) > tolerance) return false;

        // Compare velocities (vx, vy)
        if (std::abs(p1.getVx() - p2.getVx()) > tolerance) return false;
        if (std::abs(p1.getVy() - p2.getVy()) > tolerance) return false;

        return true;
    }
};

// Test that task version and parallel for produce same results
TEST_F(ProcessBodiesTest, TaskVsParallelForSameResults) {
    // Setup: compute accelerations for both simulators
    simulator1->getSystem().zeroAccelerations();
    simulator2->getSystem().zeroAccelerations();

    simulator1->getSystem().computeAccelerations();
    simulator2->getSystem().computeAccelerations();

    // Apply processBodies with task version (task_type = 0)
    simulator1->processBodies(0);

    // Apply processBodies with parallel for version (task_type = 1)
    simulator2->processBodies(1);

    // Compare results
    const auto& bodies1 = simulator1->getSystem().getBodies();
    const auto& bodies2 = simulator2->getSystem().getBodies();

    ASSERT_EQ(bodies1.size(), bodies2.size());

    for (size_t i = 0; i < bodies1.size(); ++i) {
        EXPECT_TRUE(particlesEqual(bodies1[i], bodies2[i]))
            << "Particle " << i << " differs between task and parallel for versions";
    }
}

// Test that processBodies modifies particle velocities correctly
TEST_F(ProcessBodiesTest, ProcessBodiesModifiesVelocities) {
    simulator1->getSystem().zeroAccelerations();
    simulator1->getSystem().computeAccelerations();

    auto bodies_before = simulator1->getSystem().getBodies();
    double vel_before_x = bodies_before[0].getVx();
    double vel_before_y = bodies_before[0].getVy();

    simulator1->processBodies(1); // parallel for version

    const auto& bodies_after = simulator1->getSystem().getBodies();
    double vel_after_x = bodies_after[0].getVx();
    double vel_after_y = bodies_after[0].getVy();

    // Velocities should change after kick
    EXPECT_FALSE(std::abs(vel_before_x - vel_after_x) < 1e-10 &&
                 std::abs(vel_before_y - vel_after_y) < 1e-10)
        << "Velocities should change after processBodies";
}

// Test that processBodies modifies particle positions correctly
TEST_F(ProcessBodiesTest, ProcessBodiesModifiesPositions) {
    simulator1->getSystem().zeroAccelerations();
    simulator1->getSystem().computeAccelerations();

    auto bodies_before = simulator1->getSystem().getBodies();
    double pos_before_x = bodies_before[0].getX();
    double pos_before_y = bodies_before[0].getY();

    simulator1->processBodies(0); // task version

    const auto& bodies_after = simulator1->getSystem().getBodies();
    double pos_after_x = bodies_after[0].getX();
    double pos_after_y = bodies_after[0].getY();

    // Positions should change after drift
    EXPECT_FALSE(std::abs(pos_before_x - pos_after_x) < 1e-10 &&
                 std::abs(pos_before_y - pos_after_y) < 1e-10)
        << "Positions should change after processBodies";
}

