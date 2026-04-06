# Variables del compilador
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

# Nombre del ejecutable
TARGET = a

# Lista de archivos fuente
SOURCES = main.cpp \
          Particle.cpp \
          NBodySystem.cpp \
          NBodySimulator.cpp \
          MetricsCalculator.cpp \
          Visualizer.cpp \
          Integrator.cpp

# Lista de cabeceras (para detectar cambios y recompilar)
HEADERS = Particle.h \
          NBodySystem.h \
          NBodySimulator.h \
          MetricsCalculator.h \
          Visualizer.h \
          Integrator.h

# Regla principal: Compilar el ejecutable
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Limpiar archivos generados
clean:
	rm -f $(TARGET) *.o *.dat *.png *.gif 

# Regla para ejecutar un benchmark
benchmark: $(TARGET)
	./$(TARGET) -benchmark

# Regla para análisis
analysis: $(TARGET)
	./$(TARGET) -analysis

# Regla de prueba
test:
	@echo "Ejecutando pruebas unitarias..."
	# $(CXX) $(CXXFLAGS) -o run_tests tests/test_main.cpp $(SOURCES) $(LDFLAGS) -lgtest
	# ./run_tests

.PHONY: clean benchmark analysis test