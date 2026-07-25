# Laboratorio 1: Programación paralela con OpenMP
## Simulador gravitatorio N-cuerpos en 2D (C++)

Este proyecto implementa un simulador gravitatorio de N-cuerpos en el plano utilizando C++ y OpenMP para la paralelización. El sistema integra física newtoniana, diseño orientado a objetos y análisis de rendimiento.

### 1. Organización del Equipo (Roles)

De acuerdo a lo solicitado en el punto 2 del enunciado, a continuación se detallan los roles y responsabilidades:

| Rol | Encargado | Responsabilidades |
| :--- | :--- | :--- |
| **1. Modelo y datos** | Alonso Henriquez | Clase `Particle`, contenedor del sistema, constantes, inicialización reproducible, I/O. |
| **2. Núcleo paralelo** | Benjamin Moya | `NBodySystem::computeAccelerations`, bucles todo-pares, OpenMP schedules, evitar condiciones de carrera. |
| **3. Integración y física** | Braulio Bravo | `Integrator` / `NBodySimulator`, paso temporal $\Delta t$, estabilidad y conservación de energía. |
| **4. Métricas y benchmarks** | Diego Molina | `MetricsCalculator`, `Benchmark`, mediciones de tiempo, speedup, eficiencia, Ley de Amdahl. |
| **5. Calidad, CI y visualización** | Sebastian de la Fuente | Pruebas unitarias e integración, Dockerfile, GitHub Actions, `Visualizer`, scripts de graficación. |

### 2. URL del Repositorio
El código fuente y el historial de versiones se encuentran en:
https://github.com/TinyAntu/Laboratorio-1-Programacion-paralela-con-OpenMP.git

### 3. Instrucciones de Compilación local

El proyecto utiliza **CMake** para la gestión de la compilación.

**Requisitos:**
- Compilador C++ (compatible con C++17)
- OpenMP
- CMake (>= 3.16)
- GoogleTest (se descarga automáticamente vía CMake)

**Pasos para compilar:**
```bash
mkdir build
cd build
cmake ..
make
```

Esto generará los ejecutables:
- `nbody_app`: Aplicación principal del simulador.
- `nbody_tests`: Suite de pruebas unitarias e integración.
- `test_benchmark`: Pruebas específicas para el módulo de benchmarking.

### 4. Ejecución de Pruebas

Para ejecutar la suite completa de pruebas automáticas:
```bash
# Dentro de la carpeta build
make test
# O directamente
./nbody_tests
```
Las pruebas cubren la verificación de aceleraciones analíticas, conservación de momento lineal (acción-reacción), y consistencia entre versiones seriales y paralelas.

### 5. Repetición de Experimentos

Para repetir los experimentos y generar los resultados presentados en el reporte:

#### 5.1 Parámetros por Defecto
El simulador está configurado en `src/main.cpp` con los siguientes parámetros base:
- **Número de cuerpos (N):** 1000
- **Semilla (Seed):** 42 (para reproducibilidad)
- **Paso temporal ($\Delta t$):** 0.01
- **Constante gravitacional (G):** 1.0
- **Pasos totales:** 500

#### 5.2 Ejecución del Simulador
Ejecute el binario principal para generar los archivos de datos (`.dat`):
```bash
./nbody_app
```
Esto generará archivos como `trayectorias_base.dat`, `energia_base.dat`, y sus variantes para diferentes configuraciones de OpenMP (schedule, chunk, collapse, Newton3).

#### 5.3 Generación de Gráficos
Para visualizar los resultados y generar las figuras de rendimiento (`performance_plots.png`):
```bash
# Requiere python3, numpy, pandas, matplotlib y pillow
python3 plot.py
```
*Nota: El script `plot.py` debe ejecutarse manualmente para generar los gráficos finales de trayectorias, energía y análisis de escalabilidad. y debe ejecutarse el que este dentro de la carpeta build*

### 5.4 Lab 2: Soporte GPU (CUDA)

El Lab 2 porta el cálculo de aceleraciones a GPU. El build CUDA es **opcional** y se
autodetecta: si no hay CUDA Toolkit, el proyecto compila igual que en el Lab 1.

```bash
# Build con CUDA (requiere CUDA Toolkit >= 12 y una GPU NVIDIA)
cmake -B build -DENABLE_CUDA=ON
cmake --build build --parallel
./build/nbody_gpu_tests   # tests de la capa GPU

# Build solo CPU (comportamiento del Lab 1 / CI)
cmake -B build -DENABLE_CUDA=OFF
```

En el clúster DIINF fijar la arquitectura de la GPU del nodo, por ejemplo:
`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75`

**Tolerancia CPU vs GPU:** las comparaciones entre la referencia CPU serial (Lab 1,
fuente de verdad) y la GPU usan `rtol = 1e-4` y `atol = 1e-8` (criterio
`|gpu - cpu| <= atol + rtol*|cpu|`), punto de partida sugerido por el enunciado, que
compensa el orden distinto de las sumas y el uso de FMA en GPU.

**Esquema de transferencias por paso** (Euler se integra en host según el enunciado):
- Masas: H2D **una sola vez** (no cambian durante la simulación).
- Posiciones (x, y): H2D en cada paso (el host las actualiza con drift).
- Aceleraciones (ax, ay): D2H en cada paso (las produce el kernel).
- Velocidades: **nunca** tocan el device (solo las usa el host en kick).

El layout en device es **SoA** (`d_mass`, `d_x`, `d_y`, `d_ax`, `d_ay`) para favorecer
accesos coalesced. GPU usada en desarrollo local: NVIDIA GeForce GTX 1660 SUPER
(sm_75, driver 596.36); las mediciones finales se ejecutan en el nodo GPU del clúster
DIINF (documentar nodo, GPU, driver y versión de CUDA en cada corrida).

### 6. Consideraciones Técnicas

#### 6.1 Justificación de G = 1
Se ha fijado la constante gravitacional $G = 1$ por las siguientes razones:
- **Técnica:** Previene problemas de *underflow* o *overflow* al trabajar con variables `double` y mejora la eficiencia computacional al evitar multiplicaciones por constantes pequeñas.
- **Física:** Representa un sistema de "unidades N-cuerpos" adimensional, donde las magnitudes se expresan en función de valores referenciales del propio sistema.

#### 6.2 Criterio de Tolerancia
En las pruebas de comparación entre versiones seriales y paralelas (o validaciones analíticas), se utiliza una tolerancia de **$1 \times 10^{-10}$** para compensar pequeñas desviaciones inherentes a la aritmética de punto flotante y el orden de las sumas en paralelo.
