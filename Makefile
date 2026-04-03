CXX = g++
2 CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
3 LDFLAGS = -fopenmp
4
5 TARGET = nbody_2d
6 SOURCES = main.cpp Particle.cpp NBodySystem.cpp NBodySimulator.cpp Integrator.cpp
,→ MetricsCalculator.cpp Benchmark.cpp Visualizer.cpp
7 HEADERS = Particle.h NBodySystem.h NBodySimulator.h Integrator.h MetricsCalculator.h
,→ Benchmark.h Visualizer.h
8
9 $(TARGET): $(SOURCES) $(HEADERS)
10 $(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
clean:
13 rm -f $(TARGET) *.o *.dat *.png
14
15 benchmark: $(TARGET)
16 ./$(TARGET) -benchmark
17
18 analysis: $(TARGET)
19 ./$(TARGET) -analysis
20
21 # Enlazar con GoogleTest/Catch2 seg ́un el proyecto; deber ́a ejecutarse en CI
22 test:
23 # Compilar fuentes de tests/ y dependencias; ejemplo:
24 # £(CXX) £(CXXFLAGS) -o run_tests tests/test_main.cpp NBodySystem.cpp Particle.cpp

,→ £(LDFLAGS) -lgtest -lgtest_main

25 ./run_tests
26
27 .PHONY: clean benchmark analysis test