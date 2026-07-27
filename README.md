# Laboratorio 2: Programación Paralela en GPU con CUDA
## Simulador Gravitatorio N-Cuerpos en 2D (C++ / CUDA)

Este proyecto extiende el simulador gravitatorio de N-cuerpos en el plano desarrollado en el Laboratorio 1, portando el núcleo computacional de física $O(N^2)$ a GPU utilizando **NVIDIA CUDA**. Incluye una arquitectura con layout SoA (*Structure of Arrays*), kernels de aceleración con memoria compartida y reducciones/atómicas para métricas de energía. Además, implementa un flujo estricto de desarrollo en Git con automatizaciones mediante **Agentes de IA** e integración continua (CI).

### 1. Organización del Equipo (Roles)

De acuerdo a lo solicitado en el punto 3 del enunciado, a continuación se detallan los roles y responsabilidades:

| Rol | Encargado | Responsabilidades Concretas |
| :--- | :--- | :--- |
| **1. Kernels CUDA** | Benjamin Moya | Desarrollo y optimización de `computeAccelerationsKernel` (versión básica) y `computeAccelerationsKernelShared` (uso de memoria compartida y `__syncthreads()`). Lanzadores host, macros `CUDA_CHECK`, manejo de bordes y divergencia de hilos. |
| **2. Host/Device y Memoria** | Braulio Bravo | Encapsulamiento RAII vía `CudaBuffer`, gestión de memoria en GPU con layout SoA (`d_mass`, `d_x`, `d_y`, `d_ax`, `d_ay`), gestión de `cudaMalloc`/`cudaFree`/`cudaMemcpy`, minimización de transferencias por paso e integración con `cudaDeviceSynchronize()`. |
| **3. Integración y Validación** | Diego Molina | Integración temporal Euler explícita en host, sobrecarga de métodos (`stepEulerGpu`, `computeAccelerationsGpu`), validación CPU vs. GPU bajo criterio de tolerancia ($rtol=10^{-4}, atol=10^{-8}$), y cálculo de energía ($K$ y $U$) en GPU vía reducción paralela y `atomicAdd`. |
| **4. Git, Releases y Agentes** | Alonso Henriquez | Gestión del flujo Git (protección de `main`, PR/MR obligatorios), configuración del archivo `CHANGELOG.md` (*Keep a Changelog*), marcado de releases (`v2.0.0-lab2`), y despliegue/supervisión de los 3 agentes de IA en el repositorio. |
| **5. Calidad, CI y Visualización** | Sebastian de la Fuente | Extensión del pipeline de CI (GitHub Actions / Dockerfile con soporte CUDA), revisión humana de MRs e issues, ejecución de la matriz de benchmarks en el clúster DIINF, y generación de scripts de graficación (Speedup, Ley de Amdahl, estudio de `blockDim.x`). |

### 2. URL del Repositorio
El código fuente y el historial de versiones se encuentran en:
https://github.com/TinyAntu/Laboratorio-1-Programacion-paralela-con-OpenMP.git

### 3. Instrucciones de compilación local

El proyecto utiliza **CMake** como sistema de construcción y soporta dos modos de compilación:

- **CPU (OpenMP)**: siempre disponible.
- **GPU (CUDA)**: se habilita automáticamente si se encuentra un CUDA Toolkit instalado.

# Requisitos

- Compilador compatible con **C++17**
- **CMake 3.16** o superior
- **OpenMP**
- *(Opcional)* **CUDA Toolkit** para habilitar la aceleración por GPU
- GoogleTest se descarga automáticamente durante la configuración mediante CMake.

# Compilación

```bash
mkdir build
cd build
cmake ..
make
```

# Compilación solo CPU

Si no se desea compilar el soporte CUDA:

```bash
mkdir build
cd build
cmake -DENABLE_CUDA=OFF ..
make
```

# Compilación con CUDA

Si el CUDA Toolkit está instalado, el soporte GPU se habilita automáticamente:

```bash
mkdir build
cd build
cmake -DENABLE_CUDA=ON ..
make
```

Si es necesario especificar la arquitectura de la GPU:

```bash
cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ..
make
```

(Reemplace `75` por la arquitectura correspondiente a su GPU.)

# Ejecutar la aplicación

```bash
./nbody_app
```

# Ejecutar las pruebas

Ejecutar todas las pruebas registradas por CTest:

```bash
ctest
```

O ejecutar un ejecutable específico:

```bash
./nbody_tests
./test_benchmark
```

Si CUDA está habilitado, también se generará:

```bash
./nbody_gpu_tests
```

# Ejecutables generados

Dependiendo de la configuración, se generarán los siguientes ejecutables:

- `nbody_app`: aplicación principal del simulador.
- `nbody_tests`: pruebas unitarias e integración.
- `test_benchmark`: pruebas del módulo de benchmarking.
- `nbody_gpu_tests`: pruebas de CUDA (solo si CUDA está habilitado).

## 4. Ejecución de Pruebas

El proyecto incorpora una suite de pruebas basada en **GoogleTest**, la cual valida el correcto funcionamiento de los distintos módulos del simulador tanto en CPU como, opcionalmente, en GPU.

### Ejecutar todas las pruebas

Una vez compilado el proyecto, desde la carpeta `build` es posible ejecutar todas las pruebas registradas mediante CTest:

```bash
cd build
ctest
```

También pueden ejecutarse individualmente los ejecutables de prueba:

```bash
./nbody_tests
./test_benchmark
```

Si el proyecto fue compilado con soporte CUDA (`ENABLE_CUDA=ON`) y existe una GPU NVIDIA disponible, también se genera el ejecutable:

```bash
./nbody_gpu_tests
```

### Cobertura de las pruebas

Las pruebas automáticas verifican, entre otros aspectos:

- Correcta inicialización y comportamiento de la clase `Particle`.
- Integración temporal mediante el método de Euler.
- Correctitud del cálculo de aceleraciones gravitacionales.
- Conservación del momento lineal mediante el principio de acción-reacción.
- Cálculo de métricas físicas del sistema:
  - Energía cinética.
  - Energía potencial.
  - Energía total.
  - Centro de masa.
  - Radio RMS.
  - Momento lineal total.
  - Distancia mínima entre partículas.
- Correcto funcionamiento del simulador (`NBodySimulator`).
- Validación del módulo de benchmarking.
- Comparación entre la implementación serial y las implementaciones paralelas.
- Validación de la capa CUDA y de las transferencias Host-Device (cuando CUDA está habilitado).

---

# 5. Repetición de Experimentos

Los experimentos utilizados para obtener los resultados presentados en el informe pueden reproducirse completamente utilizando el ejecutable principal del proyecto.

## 5.1 Parámetros por Defecto

El archivo `src/main.cpp` define los parámetros utilizados durante la ejecución del simulador y del benchmark.

Por defecto se utilizan los siguientes valores:

| Parámetro | Valor |
|-----------|------:|
| Número de partículas (N) | 1000 |
| Semilla aleatoria | 42 |
| Constante gravitacional (G) | 1.0 |
| Softening | 0.1 |
| Paso temporal (Δt) | 0.01 |
| Repeticiones del benchmark | 10 |
| Número de pasos de simulación | 500 |
| Intervalo de muestreo | 10 pasos |

El uso de una semilla fija garantiza que todas las ejecuciones generen exactamente las mismas condiciones iniciales, permitiendo la reproducibilidad de los experimentos.

---

## 5.2 Ejecución del Simulador

Desde la carpeta `build`, ejecutar:

```bash
./nbody_app
```

Durante la ejecución se realizan automáticamente dos etapas:

1. **Benchmark de rendimiento**, donde se evalúan las distintas implementaciones del cálculo gravitacional.
2. **Generación de datos físicos**, almacenando la evolución temporal del sistema.

Al finalizar la ejecución se generan los siguientes archivos:

| Archivo | Contenido |
|----------|-----------|
| `benchmark_results.dat` | Resultados de tiempo de ejecución de los benchmarks. |
| `scaling_analysis.dat` | Datos para el análisis de escalabilidad. |
| `snapshots.dat` | Posición y masa de cada partícula durante la simulación. |
| `energy_timeseries.dat` | Evolución temporal de las métricas físicas del sistema. |

### Contenido de `snapshots.dat`

Este archivo almacena el estado del sistema para cada instante de muestreo utilizando el formato:

```
Step ID X Y Mass
```

Cada fila representa una partícula e incluye:

- paso de simulación,
- identificador de la partícula,
- coordenadas `(x,y)`,
- masa.

Estos datos permiten reconstruir posteriormente las trayectorias del sistema.

### Contenido de `energy_timeseries.dat`

Para cada instante de muestreo se almacenan las principales métricas físicas del sistema:

```
Step
Kinetic
Potential
Total
CenterOfMassX
CenterOfMassY
RMSRadius
MomentumX
MomentumY
MomentumMag
MinDistance
```

Estas magnitudes permiten analizar la evolución dinámica del sistema, verificar la conservación aproximada de la energía y estudiar el comportamiento global de la simulación.

---

## 5.3 Generación de Gráficos

El proyecto incluye el script `plot.py`, encargado de generar todas las figuras utilizadas en el análisis experimental.

**Importante:** el script debe ejecutarse desde la carpeta `build`, ya que los archivos `.dat` son generados en dicho directorio.

```bash
cd build
python3 plot.py
```

El script requiere:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Pillow

A partir de los archivos generados por `nbody_app`, el script produce las gráficas correspondientes al análisis experimental, incluyendo:

- trayectorias de las partículas;
- evolución de la energía del sistema;
- análisis de rendimiento;
- análisis de escalabilidad.

---

## 5.4 Soporte GPU (CUDA)

El proyecto incorpora una implementación GPU para el cálculo de aceleraciones gravitacionales. El soporte CUDA es **opcional** y se habilita automáticamente cuando se detecta un CUDA Toolkit durante la configuración del proyecto.

### Compilación con CUDA

```bash
cmake -B build -DENABLE_CUDA=ON
cmake --build build --parallel
```

Una vez compilado, los tests específicos de CUDA pueden ejecutarse mediante:

```bash
./build/nbody_gpu_tests
```

### Compilación solo CPU

Si se desea mantener el comportamiento equivalente al Laboratorio 1:

```bash
cmake -B build -DENABLE_CUDA=OFF
cmake --build build --parallel
```

### Arquitectura CUDA

En el clúster DIINF puede especificarse explícitamente la arquitectura de la GPU:

```bash
cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
```

donde el valor debe reemplazarse por el *Compute Capability* de la GPU utilizada.

### Validación CPU vs GPU

La implementación GPU se valida comparando las aceleraciones calculadas contra la implementación serial de referencia.

Se utiliza el criterio:

```
|GPU - CPU| ≤ atol + rtol · |CPU|
```

con:

- `rtol = 1×10⁻⁴`
- `atol = 1×10⁻⁸`

Estos valores compensan pequeñas diferencias numéricas originadas por:

- distinto orden de las sumas en paralelo;
- instrucciones FMA utilizadas por la GPU;
- redondeo de punto flotante.

### Organización de memoria

La simulación mantiene la integración temporal en CPU, mientras que la GPU calcula únicamente las aceleraciones.

Las transferencias por paso son:

- masas: Host → Device únicamente al inicio;
- posiciones `(x,y)`: Host → Device en cada iteración;
- aceleraciones `(ax, ay)`: Device → Host en cada iteración;
- velocidades: permanecen completamente en CPU.

En GPU se utiliza un esquema **Structure of Arrays (SoA)** compuesto por:

- `d_mass`
- `d_x`
- `d_y`
- `d_ax`
- `d_ay`

lo que favorece accesos coalescentes a memoria global y mejora el rendimiento del kernel.

---

# 6. Consideraciones Técnicas

## 6.1 Justificación de G = 1

La simulación utiliza una constante gravitacional normalizada (`G = 1`) debido a que:

- evita trabajar con constantes físicas extremadamente pequeñas, reduciendo problemas numéricos de precisión;
- simplifica las ecuaciones del sistema;
- representa un sistema de unidades adimensionales ampliamente utilizado en simulaciones N-Body, donde las magnitudes físicas se expresan en unidades relativas.

## 6.2 Criterios de Tolerancia

Las validaciones numéricas consideran la naturaleza de la aritmética en punto flotante.

Para las comparaciones entre implementaciones CPU (serial/paralela) se utiliza una tolerancia absoluta del orden de:

```
1 × 10⁻¹⁰
```

permitiendo pequeñas diferencias debidas al distinto orden de acumulación de operaciones.

Para las comparaciones entre CPU y GPU se emplea una tolerancia relativa y absoluta:

- `rtol = 1×10⁻⁴`
- `atol = 1×10⁻⁸`

criterio adecuado para compensar las diferencias inherentes a la ejecución masivamente paralela y al uso de instrucciones FMA en dispositivos CUDA.