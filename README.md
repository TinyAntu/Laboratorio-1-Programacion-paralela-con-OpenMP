# Laboratorio 2: Programación GPGPU con CUDA
## Simulador gravitatorio N-cuerpos en 2D (C++/CUDA)

Este proyecto extiende el simulador gravitatorio del Laboratorio 1, portando el núcleo computacional a GPU utilizando C++ y CUDA. El sistema integra física newtoniana, diseño orientado a objetos, aceleración por hardware y análisis de rendimiento en el clúster.

### 1. Organización del Equipo (Roles y Responsabilidades)

De acuerdo a lo solicitado en la Sección 3 del enunciado, a continuación se detallan los roles y responsabilidades de los integrantes del equipo para este laboratorio:

| Rol | Encargado | Responsabilidades concretas (Lab 2) |
| :--- | :--- | :--- |
| **1. Kernels CUDA** | Benjamin Moya | Implementación de `computeAccelerationsKernel` (1 hilo por cuerpo) y `computeAccelerationsKernelShared` (memoria compartida/tiles); lanzadores host, macros `CUDA_CHECK` y protección de bordes. |
| **2. Host/device y memoria** | Braulio Bravo | Estructura `CudaBuffer` (RAII); layout SoA en device; gestión de transferencias `cudaMalloc`/`cudaMemcpy`/`cudaFree`; sincronización de memoria y minimización de copias. |
| **3. Integración y validación** | Diego Molina | Integración Euler en Host tras sincronizar; tests CPU vs GPU con tolerancias; cálculo de métricas de energía cinética $K$ y potencial $U$ en GPU (reducción en shared y atomicAdd). |
| **4. Git, releases y agentes** | Alonso Henriquez  | Protección de rama main; flujo de ramas feature/fix; bitácora en `CHANGELOG.md`; tags de release (`v2.0.0-lab2`); prompts y configuración de los 3 agentes de IA. |
| **5. Calidad, CI y visualización** | Sebastian de la Fuente | Extender CI del Lab 1; revisión de calidad de issues/MR; Dockerfile CUDA; gráficos de speedup, blockDim.x y trayectorias del clúster. |

---

### 2. Requisitos de Ejecución en GPU y Driver (Host)
Para compilar y ejecutar con aceleración por hardware localmente o en el clúster DIINF, se requiere que el host cumpla con:
*   **Hardware:** GPU NVIDIA con arquitectura Kepler o superior (Compute Capability $\ge$ 5.0).
*   **Driver de NVIDIA:** Versión mínima del driver $\ge 525.xx$ (requerido para compatibilidad con CUDA 12.2).
*   **Entorno Docker:** Requiere `nvidia-container-toolkit` instalado en el sistema anfitrión y ejecutar el contenedor con la flag `--gpus all`.

---

### 3. Instrucciones de Compilación y Ejecución (Docker)

Dado que los servidores de Integración Continua (GitHub Actions) no disponen de una GPU física, los tests están diseñados para detectar dinámicamente la presencia de CUDA y omitir de forma limpia las pruebas de GPU si no hay hardware compatible, manteniendo la pipeline en verde.

#### 3.1 Construcción del Entorno
Construir la imagen de Docker localmente:
```bash
docker build -t nbody-cuda-test -f Dockerfile .
```

#### 3.2 Compilación del Proyecto
Generar archivos de construcción y compilar dentro del contenedor:
```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace nbody-cuda-test cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CXX_FLAGS="-Wall -Wextra -Werror"
docker run --rm -v "${PWD}:/workspace" -w /workspace nbody-cuda-test cmake --build build --parallel
```

Esto generará los siguientes ejecutables dentro de la carpeta `build`:
*   `nbody_app`: Aplicación principal del simulador.
*   `nbody_tests`: Suite de pruebas unitarias e integración de CPU y GPU.
*   `test_benchmark`: Pruebas específicas para el módulo de benchmarking.

---

### 4. Ejecución de Pruebas

Para ejecutar la suite completa de pruebas automáticas (en el contenedor, sólo se ejecutarán las pruebas CPU y se saltarán limpiamente las de GPU si no hay tarjeta de video disponible):
```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace nbody-cuda-test ctest --test-dir build --output-on-failure
```

---

### 5. Repetición de Experimentos y Benchmarks (Clúster DIINF)

Las mediciones finales de performance no se aceptan desde CI, sino que deben ejecutarse únicamente en el clúster DIINF utilizando nodos con GPU dedicada.

#### 5.1 Parámetros de Simulación Obligatorios
La matriz de pruebas de rendimiento a cubrir en el nodo GPU del clúster DIINF incluye:
*   **Tamaño de problema (N):** 256, 512, 1024, 2000 cuerpos.
*   **Variante de Kernel:** Básica (0) y Memoria Compartida (1).
*   **Tamaño de bloque (blockDim.x):** 64, 128, 256, 512, 1024.
*   **Repeticiones por punto:** $\ge 10$ ejecuciones (reportando promedio $\bar{T} \pm \sigma_T$).
*   **Pasos temporales:** $\ge 100$ pasos por corrida.

#### 5.2 Generación de Gráficos
Una vez ejecutados los benchmarks en el clúster y descargados los archivos `.dat` (`benchmark_results.dat`, `blockdim_study.dat` y `trajectories.dat`) a tu máquina local, puedes generar los gráficos requeridos ejecutando el script de Python dentro del contenedor Docker:
```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace nbody-cuda-test python3 plot.py
```

Esto generará las figuras del informe (incluyendo análisis de Speedup, Amdahl, blockDim y trayectorias físicas).

### 5.4 Lab 2: Soporte GPU (CUDA)

El Lab 2 porta el cálculo de aceleraciones a GPU. El build CUDA es **opcional** y se autodetecta: si no hay CUDA Toolkit, el proyecto compila igual que en el Lab 1.

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

**Esquema de transferencias por paso** (Euler se integra en host según el enunciado):
- Masas: H2D **una sola vez** (no cambian durante la simulación).
- Posiciones (x, y): H2D en cada paso (el host las actualiza con drift).
- Aceleraciones (ax, ay): D2H en cada paso (las produce el kernel).
- Velocidades: **nunca** tocan el device (solo las usa el host en kick).

El layout en device es **SoA** (`d_mass`, `d_x`, `d_y`, `d_ax`, `d_ay`) para favorecer accesos coalesced. GPU usada en desarrollo local: NVIDIA GeForce GTX 1660 SUPER (sm_75, driver 596.36); las mediciones finales se ejecutan en el nodo GPU del clúster DIINF (documentar nodo, GPU, driver y versión de CUDA en cada corrida).

---

### 6. Consideraciones Técnicas y Físicas

#### 6.1 Justificación de G = 1
Se ha fijado la constante gravitacional $G = 1$ por las siguientes razones:
*   **Técnica:** Previene problemas de *underflow* o *overflow* al trabajar con variables `double` y mejora la eficiencia computacional al evitar multiplicaciones por constantes extremadamente pequeñas en el kernel de CUDA.
*   **Física:** Representa un sistema de unidades N-cuerpos adimensional.

#### 6.2 Definición del Sistema de Unidades Físicas (Adimensionales)
Para asegurar la coherencia física de la simulación con $G = 1$, definimos el sistema de unidades adimensionales del simulador en función de tres unidades fundamentales del sistema:
*   **Masa ($[M]$):** Unidad de masa referencial, definida tal que la masa de una partícula típica o la masa total del sistema sea $1$ unidad de masa.
*   **Longitud ($[L]$):** Unidad de longitud referencial, que define la escala del plano bidimensional (por ejemplo, el radio inicial de distribución de los cuerpos).
*   **Tiempo ($[T]$):** Unidad de tiempo derivada del sistema, calculada de tal forma que la constante gravitatoria sea unitaria. La relación física es:
    $$[T] = \sqrt{\frac{[L]^3}{G \cdot [M]}}$$
    Con $G = 1$, un intervalo de tiempo simulado de $\Delta t = 0.01$ equivale a $0.01 [T]$.
*   **Velocidad ($[V]$) y Aceleración ($[A]$):** Unidades derivadas del movimiento, expresadas como $[V] = [L]/[T]$ y $[A] = [L]/[T]^2$ respectivamente.

#### 6.3 Criterio de Tolerancia CPU vs. GPU
Debido a las diferencias de redondeo y acumulación en aritmética de punto flotante en paralelo dentro de la GPU, se define una tolerancia mixta aceptable para las aceleraciones:
*   **Tolerancia Relativa (`rtol`):** $1 \times 10^{-4}$
*   **Tolerancia Absoluta (`atol`):** $1 \times 10^{-8}$
*   Fórmula de validación: $|a_{cpu} - a_{gpu}| \le \text{atol} + \text{rtol} \times |a_{cpu}|$
