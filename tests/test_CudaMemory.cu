// Tests de la capa host/device y memoria (Lab2, Rol 2) + equivalencia CPU vs GPU
// para ambas variantes de kernel (Rol 1): 0 = básica, 1 = shared memory (tiles).
// Requieren una GPU NVIDIA disponible (se compilan solo con ENABLE_CUDA=ON).
//
// Tolerancias CPU vs GPU (documentadas en README, punto de partida del enunciado):
//   rtol = 1e-4, atol = 1e-8
// Criterio: |gpu - cpu| <= atol + rtol * |cpu|
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "CudaBuffer.h"
#include "NBodySystem.h"
#include "NBodySimulator.h"

#include <cmath>
#include <numeric>
#include <vector>

namespace {

constexpr double RTOL = 1e-4;
constexpr double ATOL = 1e-8;

// Compara con tolerancia mixta absoluta/relativa (estilo numpy.isclose)
::testing::AssertionResult NearTol(double gpu, double cpu) {
    const double tol = ATOL + RTOL * std::fabs(cpu);
    if (std::fabs(gpu - cpu) <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "gpu=" << gpu << " cpu=" << cpu
           << " |diff|=" << std::fabs(gpu - cpu) << " > tol=" << tol;
}

// Referencia CPU serial (fuente de verdad del Lab 1) para una semilla dada
void computeCpuReference(NBodySystem& sys, unsigned int seed, int N) {
    sys.loadFromSeed(seed, N);
    sys.computeAccelerations();
}

// Verifica si existe hardware GPU CUDA disponible en tiempo de ejecución
bool isCudaDeviceAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

} // namespace

// ===== CudaBuffer: RAII y transferencias =====

TEST(CudaBufferTest, RoundTripH2DD2H) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    std::vector<double> host_in(1000);
    std::iota(host_in.begin(), host_in.end(), 0.0); // 0, 1, 2, ...

    CudaBuffer<double> buf(host_in.size());
    ASSERT_EQ(buf.size(), host_in.size());
    ASSERT_NE(buf.data(), nullptr);

    buf.copyToDevice(host_in);

    std::vector<double> host_out;
    buf.copyToHost(host_out);

    ASSERT_EQ(host_out.size(), host_in.size());
    for (std::size_t i = 0; i < host_in.size(); ++i) {
        EXPECT_DOUBLE_EQ(host_out[i], host_in[i]) << "indice " << i;
    }
}

TEST(CudaBufferTest, MoveTransfiereLaPropiedad) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    CudaBuffer<double> a(64);
    double* raw = a.data();

    CudaBuffer<double> b(std::move(a));
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(b.size(), 64u);
    EXPECT_EQ(a.data(), nullptr); // a quedó vacío: no habrá double-free
    EXPECT_EQ(a.size(), 0u);

    CudaBuffer<double> c;
    c = std::move(b);
    EXPECT_EQ(c.data(), raw);
    EXPECT_EQ(b.data(), nullptr);
}

TEST(CudaBufferTest, TransferenciasInvalidasLanzan) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    CudaBuffer<double> vacio;
    std::vector<double> datos(4, 1.0);
    EXPECT_THROW(vacio.copyToDevice(datos), std::logic_error);

    CudaBuffer<double> chico(2);
    EXPECT_THROW(chico.copyToDevice(datos), std::out_of_range);
}

// ===== Aceleraciones GPU: caso analítico del enunciado (ambas variantes) =====
// Dos masas en el eje x: m1 en x=0, m2 en x=d. Sobre la partícula 1:
//   a1x = G * m2 * d / (d^2 + eps^2)^(3/2)
// Con G=m2=d=1, eps=0.1 -> a1x = 1/(1.01)^1.5 ≈ 0.98519 (el CPU serial del
// Lab 1 es la fuente de verdad; el valor "0.971" del PDF no coincide con la
// fórmula (1) del propio enunciado con estos parámetros).
TEST(GpuAccelerationsTest, CasoAnaliticoDosCuerpos) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    const double G = 1.0, eps = 0.1, d = 1.0, m2 = 1.0;
    const double analitico = G * m2 * d / std::pow(d * d + eps * eps, 1.5);

    for (int variant : {0, 1}) {
        NBodySystem gpu_sys(G, eps);
        gpu_sys.addParticle(Particle(1.0, 0.0, 0.0));
        gpu_sys.addParticle(Particle(m2, d, 0.0));
        gpu_sys.computeAccelerationsGpu(variant);

        EXPECT_TRUE(NearTol(gpu_sys.getBodies()[0].getAx(), analitico))
            << "variante " << variant;
        EXPECT_TRUE(NearTol(gpu_sys.getBodies()[0].getAy(), 0.0))
            << "variante " << variant;
        // Acción-reacción con masas iguales: a2x = -a1x
        EXPECT_TRUE(NearTol(gpu_sys.getBodies()[1].getAx(), -analitico))
            << "variante " << variant;
    }
}

// ===== Equivalencia CPU serial vs GPU (N pequeño, semilla fija) =====

TEST(GpuAccelerationsTest, ConsistenciaCpuVsGpuSeedFija) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    const int N = 32;
    const unsigned int seed = 123;

    NBodySystem cpu_sys(1.0, 0.1);
    computeCpuReference(cpu_sys, seed, N);
    const auto& cpu = cpu_sys.getBodies();

    for (int variant : {0, 1}) {
        NBodySystem gpu_sys(1.0, 0.1);
        gpu_sys.loadFromSeed(seed, N);
        gpu_sys.computeAccelerationsGpu(variant);

        const auto& gpu = gpu_sys.getBodies();
        ASSERT_EQ(cpu.size(), gpu.size());
        for (int i = 0; i < N; ++i) {
            EXPECT_TRUE(NearTol(gpu[i].getAx(), cpu[i].getAx()))
                << "variante " << variant << ", ax, cuerpo " << i;
            EXPECT_TRUE(NearTol(gpu[i].getAy(), cpu[i].getAy()))
                << "variante " << variant << ", ay, cuerpo " << i;
        }
    }
}

// Requisito del enunciado §4.1: la variante shared debe producir el mismo
// resultado físico que la básica dentro de la tolerancia acordada.
TEST(GpuAccelerationsTest, BasicaVsSharedCoinciden) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    const int N = 100;
    const unsigned int seed = 99;

    NBodySystem basica(1.0, 0.1);
    basica.loadFromSeed(seed, N);
    basica.computeAccelerationsGpu(0);

    NBodySystem shared(1.0, 0.1);
    shared.loadFromSeed(seed, N);
    shared.computeAccelerationsGpu(1);

    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(NearTol(shared.getBodies()[i].getAx(),
                            basica.getBodies()[i].getAx())) << "ax, cuerpo " << i;
        EXPECT_TRUE(NearTol(shared.getBodies()[i].getAy(),
                            basica.getBodies()[i].getAy())) << "ay, cuerpo " << i;
    }
}

// El resultado no debe depender de blockDim.x. Crítico para la variante shared:
// con N=100 (no múltiplo de ningún block size) los tiles quedan parcialmente
// llenos y se ejercita la protección de bordes en la carga cooperativa.
TEST(GpuAccelerationsTest, IndependienteDelBlockSize) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    const int N = 100;
    const unsigned int seed = 7;

    NBodySystem ref_sys(1.0, 0.1);
    computeCpuReference(ref_sys, seed, N);
    const auto& ref = ref_sys.getBodies();

    for (int variant : {0, 1}) {
        for (int block_size : {64, 128, 256, 512, 1024}) {
            NBodySystem gpu_sys(1.0, 0.1);
            gpu_sys.loadFromSeed(seed, N);
            gpu_sys.computeAccelerationsGpu(variant, block_size);
            const auto& gpu = gpu_sys.getBodies();
            for (int i = 0; i < N; ++i) {
                EXPECT_TRUE(NearTol(gpu[i].getAx(), ref[i].getAx()))
                    << "variante " << variant << " block=" << block_size
                    << " ax cuerpo " << i;
                EXPECT_TRUE(NearTol(gpu[i].getAy(), ref[i].getAy()))
                    << "variante " << variant << " block=" << block_size
                    << " ay cuerpo " << i;
            }
        }
    }
}

TEST(GpuAccelerationsTest, ParametrosInvalidosLanzan) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    NBodySystem sys(1.0, 0.1);
    sys.loadFromSeed(42, 8);
    EXPECT_THROW(sys.computeAccelerationsGpu(99), std::invalid_argument);
    EXPECT_THROW(sys.computeAccelerationsGpu(0, 0), std::invalid_argument);
    EXPECT_THROW(sys.computeAccelerationsGpu(0, 2048), std::invalid_argument);
}

// ===== Integración: stepEulerGpu vs integrateEuler serial =====

TEST(GpuIntegrationTest, StepEulerGpuCoincideConSerial) {
    if (!isCudaDeviceAvailable()) {
        GTEST_SKIP() << "No CUDA GPU detected. Skipping test.";
    }

    const int N = 16;
    const unsigned int seed = 2026;
    const double G = 1.0, eps = 0.1, dt = 0.01;
    const int steps = 10;

    NBodySimulator cpu_sim(N, seed, G, eps, dt);
    NBodySimulator gpu_sim(N, seed, G, eps, dt);

    for (int s = 0; s < steps; ++s) {
        cpu_sim.integrateEuler();
        gpu_sim.stepEulerGpu();
    }

    const auto& cpu = cpu_sim.getBodies();
    const auto& gpu = gpu_sim.getBodies();
    ASSERT_EQ(cpu.size(), gpu.size());
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(NearTol(gpu[i].getX(), cpu[i].getX())) << "x, cuerpo " << i;
        EXPECT_TRUE(NearTol(gpu[i].getY(), cpu[i].getY())) << "y, cuerpo " << i;
        EXPECT_TRUE(NearTol(gpu[i].getVx(), cpu[i].getVx())) << "vx, cuerpo " << i;
        EXPECT_TRUE(NearTol(gpu[i].getVy(), cpu[i].getVy())) << "vy, cuerpo " << i;
    }

    // La energía se calculó en host en ambos casos: también debe coincidir
    EXPECT_TRUE(NearTol(gpu_sim.getSystemEnergy().first,
                        cpu_sim.getSystemEnergy().first));
    EXPECT_TRUE(NearTol(gpu_sim.getSystemEnergy().second,
                        cpu_sim.getSystemEnergy().second));
}
