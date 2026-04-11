# Variables del compilador
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp
# Flag -static ayuda a que el .exe encuentre las librerías en Windows
TEST_LDFLAGS = -lgtest -lgtest_main -lpthread -static

# Nombre del ejecutable principal
TARGET = nbody_2d.exe

# Lista de archivos de lógica (SIN main.cpp)
SOURCES_LIB = Particle.cpp \
              NBodySystem.cpp \
              NBodySimulator.cpp \
              MetricsCalculator.cpp \
              Visualizer.cpp \
              Integrator.cpp \
              Benchmark.cpp

# Lista de cabeceras
HEADERS = Particle.h \
          NBodySystem.h \
          NBodySimulator.h \
          MetricsCalculator.h \
          Visualizer.h \
          Integrator.h \
          Benchmark.h

# Regla principal: Compilar el simulador
$(TARGET): main.cpp $(SOURCES_LIB) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(SOURCES_LIB) $(LDFLAGS)

# Regla de prueba corregida para Windows

test:
	@echo "Compilando y ejecutando pruebas unitarias..."
	$(CXX) $(CXXFLAGS) -I. -o tests/test_metrics.exe \
		tests/test_metrics.cpp Particle.cpp MetricsCalculator.cpp $(LDFLAGS)
	./tests/test_metrics.exe
	$(CXX) $(CXXFLAGS) -o run_tests_system.exe tests/test_main.cpp tests/test_NBodySystem.cpp $(SOURCES_LIB) $(LDFLAGS) $(TEST_LDFLAGS)
	-./run_tests_system.exe

	$(CXX) $(CXXFLAGS) -o run_tests_simulator.exe tests/test_main.cpp tests/test_NBodySimulator.cpp $(SOURCES_LIB) $(LDFLAGS) $(TEST_LDFLAGS)
	./run_tests_simulator.exe

test_metrics:
	$(CXX) $(CXXFLAGS) -I. -o tests/test_metrics.exe \
		tests/test_metrics.cpp Particle.cpp MetricsCalculator.cpp $(LDFLAGS)
	./tests/test_metrics.exe

test_benchmark:
	g++ -Wall -Wextra -O3 -fopenmp -std=c++17 -I. \
	    tests/test_Benchmark.cpp \
	    Particle.cpp NBodySystem.cpp NBodySimulator.cpp \
	    MetricsCalculator.cpp Benchmark.cpp \
	    -o test_benchmark
	./test_benchmark

# Limpiar archivos generados (Comando del para Windows)

clean:
	del /Q $(TARGET) run_tests.exe tests\test_metrics.exe *.o *.dat *.png 2>nul || exit 0

.PHONY: clean benchmark analysis test test_metrics