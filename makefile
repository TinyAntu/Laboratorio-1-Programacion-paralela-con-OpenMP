NVCC = nvcc
CXX = g++
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -Wall,-Wextra
CXXFLAGS = -Wall -Wextra -O3 -std=c++17
LDFLAGS = -lcudart
TARGET = nbody_2d_cuda
CPP_SOURCES = main.cpp Particle.cpp NBodySystem.cpp ...
CU_SOURCES = kernels/accelerations.cu kernels/metrics.cu

$(TARGET): $(CPP_SOURCES) $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CPP_SOURCES) $(CU_SOURCES) $(LDFLAGS)

test:
	./run_tests

.PHONY: clean benchmark test