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

#### Requisitos

- Compilador compatible con **C++17**
- **CMake 3.16** o superior
- **OpenMP**
- *(Opcional)* **CUDA Toolkit** para habilitar la aceleración por GPU
- GoogleTest se descarga automáticamente durante la configuración mediante CMake.

#### Compilación

bash
mkdir build
cd build
cmake ..
make


#### Compilación solo CPU

Si no se desea compilar el soporte CUDA:

bash
mkdir build
cd build
cmake -DENABLE_CUDA=OFF ..
make


#### Compilación con CUDA

Si el CUDA Toolkit está instalado, el soporte GPU se habilita automáticamente:

bash
mkdir build
cd build
cmake -DENABLE_CUDA=ON ..
make


Si es necesario especificar la arquitectura de la GPU:

bash
cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ..
make


(Reemplace `75` por la arquitectura correspondiente a su GPU.)

#### Ejecutar la aplicación

bash
./nbody_app


#### Ejecutar las pruebas

Ejecutar todas las pruebas registradas por CTest:

bash
ctest


O ejecutar un ejecutable específico:

bash
./nbody_tests
./test_benchmark


Si CUDA está habilitado, también se generará:

bash
./nbody_gpu_tests


#### Ejecutables generados

Dependiendo de la configuración, se generarán los siguientes ejecutables:

- `nbody_app`: aplicación principal del simulador.
- `nbody_tests`: pruebas unitarias e integración.
- `test_benchmark`: pruebas del módulo de benchmarking.
- `nbody_gpu_tests`: pruebas de CUDA (solo si CUDA está habilitado).

## 4. Ejecución de Pruebas

El proyecto incorpora una suite de pruebas automáticas basada en **GoogleTest**, integrada con **CTest** mediante CMake. Estas pruebas verifican tanto la implementación secuencial como la paralela (OpenMP) y, cuando está disponible, la implementación sobre GPU mediante CUDA.

### Ejecutar todas las pruebas

Una vez compilado el proyecto, desde el directorio `build` pueden ejecutarse todas las pruebas registradas:

bash
ctest


También es posible utilizar el objetivo generado por CMake:

bash
make test


### Ejecutar pruebas individuales

Cada conjunto de pruebas puede ejecutarse de manera independiente:

bash
./nbody_tests
./test_benchmark


Si el proyecto fue compilado con soporte CUDA (`ENABLE_CUDA=ON`) también estará disponible:

bash
./nbody_gpu_tests


### Cobertura de las pruebas

La suite de pruebas verifica los principales componentes del simulador:

- Correctitud de la clase `Particle`.
- Inicialización y generación reproducible de sistemas mediante una semilla fija.
- Correctitud del integrador de Euler.
- Validación del cálculo de aceleraciones gravitacionales.
- Verificación de las métricas físicas del sistema:
  - Energía cinética.
  - Energía potencial.
  - Energía total.
  - Centro de masa.
  - Momento lineal.
  - Radio RMS.
  - Distancia mínima entre partículas.
- Consistencia entre implementaciones seriales y paralelas (OpenMP).
- Funcionamiento del módulo de benchmarking.
- Correctitud de la administración de memoria y transferencia Host–Device en la versión CUDA.

Cuando CUDA está habilitado, las pruebas GPU comparan los resultados obtenidos por los kernels contra la implementación CPU serial, utilizada como referencia, verificando que las diferencias permanezcan dentro de la tolerancia numérica definida para el proyecto.

---

## 5. Repetición de Experimentos

Los experimentos presentados en el informe pueden reproducirse completamente utilizando el ejecutable principal del proyecto. Durante la ejecución se realizan automáticamente los benchmarks de rendimiento y la generación de los archivos de salida necesarios para el análisis físico del sistema.

## 5.1 Parámetros utilizados

Los parámetros por defecto se encuentran definidos en `src/main.cpp`.

| Parámetro | Valor |
|-----------|------:|
| Número de partículas (N) | 1000 |
| Semilla aleatoria | 42 |
| Constante gravitacional (G) | 1.0 |
| Softening | 0.1 |
| Paso temporal (Δt) | 0.01 |
| Repeticiones benchmark | 10 |
| Pasos simulados | 500 |
| Intervalo de muestreo | 10 |

La utilización de una semilla fija (`seed = 42`) garantiza la reproducibilidad de los experimentos y permite comparar directamente los resultados obtenidos entre distintas implementaciones (serial, OpenMP y CUDA).

---

## 5.2 Ejecución del simulador

Desde el directorio `build` ejecutar:

bash
./nbody_app


Durante la ejecución el programa realiza automáticamente dos etapas:

1. **Benchmark de rendimiento**, donde se evalúan las distintas implementaciones disponibles y se generan los archivos utilizados para el análisis de escalabilidad.

2. **Visualización física**, donde se simula la evolución temporal del sistema almacenando periódicamente el estado completo de las partículas y las principales magnitudes físicas.

Al finalizar se generan los siguientes archivos:

| Archivo | Descripción |
|----------|-------------|
| `benchmark_results.dat` | Resultados de tiempos de ejecución de los distintos algoritmos evaluados. |
| `scaling_analysis.dat` | Datos utilizados para el análisis de escalabilidad. |
| `snapshots.dat` | Posición y masa de todas las partículas para cada instante muestreado de la simulación. |
| `energy_timeseries.dat` | Evolución temporal de las métricas físicas del sistema. |

El archivo `snapshots.dat` posee el siguiente formato:

text
Step ID X Y Mass


donde cada fila representa una partícula en un instante de tiempo determinado.

Por su parte, `energy_timeseries.dat` almacena para cada muestra:

text
Step
KineticEnergy
PotentialEnergy
TotalEnergy
CenterOfMassX
CenterOfMassY
RMSRadius
MomentumX
MomentumY
MomentumMagnitude
MinDistance


Estas magnitudes permiten evaluar la estabilidad numérica de la simulación y verificar la conservación aproximada de las cantidades físicas.

---

## 5.3 Generación de gráficos

El repositorio incluye el script `plot.py`, encargado de procesar los archivos `.dat` generados por el simulador.

**Importante:** el script debe ejecutarse desde la carpeta `build`, ya que allí se copian automáticamente tanto el script como los archivos de salida durante la compilación.

bash
cd build
python3 plot.py


Se requiere tener instaladas las siguientes bibliotecas:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Pillow

El script genera automáticamente las figuras utilizadas para el análisis experimental, incluyendo:

- Trayectorias de las partículas.
- Evolución temporal de la energía.
- Conservación del momento lineal.
- Evolución del centro de masa.
- Análisis de escalabilidad.
- Comparación de tiempos de ejecución entre implementaciones.

---

## 5.4 Soporte GPU (CUDA)

El proyecto incorpora una implementación opcional mediante CUDA para acelerar el cálculo de las aceleraciones gravitacionales.

La compilación GPU se habilita automáticamente cuando existe un CUDA Toolkit instalado. En caso contrario, el proyecto continúa compilando únicamente la versión CPU, manteniendo compatibilidad con entornos sin GPU (por ejemplo, la integración continua).

### Compilar con CUDA

bash
cmake -B build -DENABLE_CUDA=ON
cmake --build build --parallel


### Ejecutar las pruebas GPU

bash
./build/nbody_gpu_tests


### Compilar únicamente la versión CPU

bash
cmake -B build -DENABLE_CUDA=OFF
cmake --build build


En el clúster DIINF puede especificarse manualmente la arquitectura CUDA:

bash
cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75


---

## 5.5 Transferencias entre CPU y GPU

Siguiendo el enunciado del laboratorio, la integración temporal mediante Euler permanece ejecutándose en CPU, mientras que únicamente el cálculo de aceleraciones gravitacionales se realiza sobre la GPU.

En cada iteración ocurre el siguiente flujo de datos:

- Las masas se copian al dispositivo una única vez al inicio de la simulación.
- Las posiciones (`x`,`y`) se transfieren desde CPU hacia GPU en cada paso temporal.
- El kernel CUDA calcula las aceleraciones.
- Las aceleraciones (`ax`,`ay`) son copiadas nuevamente al host.
- La actualización de velocidades y posiciones continúa ejecutándose en CPU.

Las velocidades nunca son transferidas al dispositivo, ya que únicamente intervienen durante la integración realizada por el host.

La memoria del dispositivo utiliza un esquema **Structure of Arrays (SoA)**:

- `d_mass`
- `d_x`
- `d_y`
- `d_ax`
- `d_ay`

Esta organización favorece accesos coalescentes a memoria global y mejora el rendimiento del kernel respecto a una representación basada en estructuras (AoS).

---

## 5.6 Metodología de medición (kernel-only vs end-to-end)

Ambas mediciones usan `std::chrono::steady_clock` alrededor de `cudaDeviceSynchronize()`, sin `cudaEvent_t`, tal como exige el enunciado. Ambas promedian el **mismo** número de lanzamientos (`steps`, por defecto 100) y devuelven el tiempo promedio de uno solo. La diferencia está únicamente en qué encierra el cronómetro:

| | H2D | kernel | sync | D2H | Euler en host |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `benchmarkKernelOnly` | fuera del cronómetro | ✔ | ✔ | no se ejecuta | no se ejecuta |
| `benchmarkEndToEnd` | ✔ | ✔ | ✔ | ✔ | ✔ |

En la medición kernel-only el host no integra, de modo que las posiciones en el dispositivo no cambian entre lanzamientos y no hace falta volver a copiarlas. La sincronización se realiza **en cada iteración**, no una sola vez al final: sin ella los lanzamientos se encolarían y se mediría el *throughput* del kernel en lugar del costo que este aporta realmente a un paso temporal.

Gracias a esta simetría se cumple por construcción la identidad


EndToEndMean_s − KernelOnlyMean_s  =  transferencias H2D/D2H + sincronización + Euler en host


que es la fracción serial mínima exigida para el análisis de la ley de Amdahl. Cronometrar un único lanzamiento en el lado kernel rompía esta identidad: la latencia de lanzamiento y sincronización no se amortizaba, el ruido relativo de esa medición subía a ~48 % (mediana) frente al ~6.5 % del end-to-end, y la resta podía dar valores **negativos**, saturando la fracción serial a cero y degradando la curva de Amdahl. Por eso ambas mediciones deben ejecutarse siempre con el mismo `steps`.

---

## 6. Consideraciones Técnicas

### 6.1 Constante gravitacional

Se adopta una constante gravitacional normalizada:

$$
G = 1
$$

Esta decisión posee dos ventajas principales:

- Evita problemas numéricos asociados a constantes extremadamente pequeñas cuando se utilizan variables de doble precisión (`double`).
- Permite trabajar en unidades adimensionales típicas de simulaciones N-Body, facilitando la comparación entre experimentos sin alterar el comportamiento físico relativo del sistema.

---

### 6.2 Parámetro de suavizado (Softening)

El parámetro

$$
\varepsilon = 0.1
$$

se incorpora al denominador de la ley gravitacional para evitar singularidades cuando dos partículas se encuentran extremadamente próximas.

Su utilización mejora la estabilidad numérica de la simulación y evita aceleraciones excesivamente grandes producto de distancias cercanas a cero.

---

### 6.3 Tolerancias numéricas

Las comparaciones entre implementaciones CPU y GPU utilizan el criterio

$$
|GPU - CPU| \le atol + rtol \cdot |CPU|
$$

con:

- `rtol = 1 × 10⁻⁴`
- `atol = 1 × 10⁻⁸`

Estas tolerancias compensan las pequeñas diferencias producidas por:

- distinto orden de acumulación de las sumas,
- paralelismo masivo de la GPU,
- utilización de instrucciones FMA (Fused Multiply-Add),
- comportamiento normal de la aritmética de punto flotante.

Para las pruebas puramente CPU se emplean tolerancias más estrictas (`1×10⁻¹⁰`), ya que el orden de las operaciones permanece prácticamente inalterado.

---

### 6.4 Implementación GPU (CUDA)

La versión GPU del simulador acelera el cálculo de aceleraciones gravitacionales y el cálculo de la energía del sistema, manteniendo la integración temporal de Euler en la CPU, tal como establece el enunciado del laboratorio. De esta forma, únicamente los cálculos con mayor costo computacional son ejecutados sobre la GPU, mientras que la actualización de posiciones y velocidades continúa realizándose en el host.

#### Organización de memoria

El estado del sistema en el dispositivo se administra mediante la clase `DeviceNBodyState`, la cual encapsula todos los buffers CUDA utilizando la clase `CudaBuffer`. Esta última implementa el patrón RAII (Resource Acquisition Is Initialization), garantizando la correcta reserva (`cudaMalloc`) y liberación (`cudaFree`) automática de la memoria del dispositivo.

En la GPU los datos se almacenan utilizando un esquema **Structure of Arrays (SoA)**, compuesto por los siguientes arreglos independientes:

- `d_mass`
- `d_x`
- `d_y`
- `d_ax`
- `d_ay`
- `d_vx`
- `d_vy`

A diferencia de una representación basada en estructuras (AoS), este diseño permite que hilos consecutivos accedan a posiciones contiguas de memoria global, favoreciendo accesos coalescentes y mejorando el ancho de banda efectivo de la GPU.

#### Transferencias Host–Device

Con el objetivo de minimizar el costo de comunicación entre CPU y GPU, únicamente se transfieren los datos estrictamente necesarios en cada paso temporal.

Durante la simulación se sigue el siguiente esquema:

- **Masas:** se copian una única vez al dispositivo, ya que permanecen constantes durante toda la simulación.
- **Posiciones (x,y):** se transfieren desde el host al dispositivo en cada iteración, debido a que la integración de Euler actualiza estas variables en CPU.
- **Aceleraciones (ax, ay):** una vez finalizado el kernel, son copiadas nuevamente al host para continuar la integración temporal.
- **Velocidades:** solamente se transfieren cuando se calcula la energía del sistema en GPU; durante el cálculo de aceleraciones nunca son necesarias.

Este esquema reduce significativamente el volumen de datos transferidos respecto a copiar el estado completo del sistema en cada iteración.

---

### Cálculo de aceleraciones en GPU

El cálculo de aceleraciones mantiene el algoritmo directo del problema N-Body, cuya complejidad computacional es $O(N^2)$.

Cada hilo CUDA es responsable de calcular la aceleración correspondiente a una única partícula.

El índice global del hilo se obtiene mediante

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

y determina la partícula sobre la cual trabajará dicho hilo.

Posteriormente, cada hilo recorre secuencialmente todas las partículas del sistema acumulando la contribución gravitacional producida por cada una de ellas.

La configuración de lanzamiento utiliza una grilla unidimensional con tamaño

$$gridSize = \left\lceil \frac{N}{blockSize} \right\rceil$$

de modo que exista al menos un hilo disponible para cada cuerpo del sistema.

El tamaño del bloque (`block_size`) es configurable por el usuario, validándose previamente que pertenezca al rango permitido por CUDA (1–1024 hilos por bloque).

---

### Variante básica

La primera implementación utiliza exclusivamente memoria global.

Cada hilo:

1. Lee su posición.
2. Recorre todas las partículas.
3. Calcula las fuerzas gravitacionales.
4. Acumula la aceleración total.
5. Escribe el resultado final en `d_ax` y `d_ay`.

Esta implementación es sencilla y sirve como referencia para validar la correctitud del algoritmo.

---

### Variante optimizada mediante memoria compartida

La segunda implementación incorpora **Shared Memory** para reducir el número de accesos a memoria global.

El procedimiento consiste en dividir el conjunto de partículas en **tiles** de tamaño igual al número de hilos por bloque.

Para cada tile:

1. Todos los hilos cooperan cargando posiciones y masas desde memoria global hacia memoria compartida.
2. Se sincroniza el bloque mediante `__syncthreads()`.
3. Cada hilo calcula las interacciones utilizando los datos almacenados en Shared Memory.
4. Finalizado el procesamiento del tile, los hilos vuelven a sincronizarse antes de cargar el siguiente bloque de partículas.

La memoria compartida almacena tres arreglos:

- posiciones X,
- posiciones Y,
- masas.

Cada uno posee un tamaño igual a `blockDim.x`, por lo que la memoria dinámica reservada durante el lanzamiento del kernel corresponde a

$$3 \times blockDim \times sizeof(double)$$

Este enfoque disminuye considerablemente la cantidad de accesos repetidos a memoria global, ya que todas las partículas pertenecientes a un tile son reutilizadas por los hilos del mismo bloque.

---

### Manejo del parámetro de suavizado (Softening)

Para evitar singularidades cuando dos partículas se encuentran muy próximas, ambas implementaciones incorporan el parámetro de suavizado directamente en el cálculo de la distancia:

$$r^2 = dx^2 + dy^2 + \varepsilon^2$$

donde $\varepsilon$ corresponde al parámetro `softening`.

La inclusión de este término evita divisiones por cero y reduce aceleraciones extremadamente grandes que podrían afectar la estabilidad numérica de la simulación.

---

### Cálculo de energía en GPU

El cálculo de la energía también se ejecuta completamente sobre la GPU mediante dos estrategias independientes.

#### Reducción paralela

La primera implementación utiliza una reducción jerárquica en memoria compartida.

Cada hilo calcula:

- la energía cinética de una partícula;
- la energía potencial considerando únicamente pares $j > i$, evitando contabilizar dos veces la misma interacción.

Las contribuciones se reducen primero dentro de cada bloque mediante memoria compartida y posteriormente una segunda reducción combina los resultados parciales hasta obtener la energía total del sistema.

Esta estrategia reduce significativamente el número de accesos a memoria global y minimiza la sincronización entre bloques.

#### Acumulación mediante operaciones atómicas

La segunda implementación calcula las mismas contribuciones individuales, pero cada hilo realiza únicamente dos operaciones `atomicAdd` sobre variables globales:

- una para la energía cinética;
- una para la energía potencial.

Esta versión presenta una implementación más sencilla, aunque puede experimentar mayor contención cuando muchos hilos intentan actualizar simultáneamente las mismas posiciones de memoria.

---

### Decisiones de diseño

Las principales decisiones adoptadas en la implementación CUDA fueron:

- mantener la integración temporal en CPU para respetar la arquitectura propuesta por el laboratorio;
- minimizar las transferencias Host–Device copiando únicamente los datos estrictamente necesarios;
- utilizar una representación SoA para favorecer accesos coalescentes;
- ofrecer dos implementaciones del cálculo de aceleraciones (memoria global y memoria compartida) con fines comparativos;
- implementar dos estrategias distintas para el cálculo de energía (reducción paralela y operaciones atómicas), permitiendo evaluar el impacto de diferentes mecanismos de sincronización en GPU.

## 7. Agentes de Integración Continua (CI)

Como complemento al pipeline tradicional de integración continua, el proyecto incorpora tres agentes inteligentes que utilizan modelos de lenguaje (Google Gemini) para asistir el proceso de revisión del código y la documentación. Estos agentes no reemplazan la revisión humana, sino que automatizan tareas repetitivas y clasifican los cambios según su impacto.

Todos los agentes se ejecutan mediante **GitHub Actions**, utilizan la API de GitHub para interactuar con el repositorio y emplean la biblioteca **Google GenAI** para realizar el análisis del contenido.

### 7.1 Agente Revisor de Bugs

El agente de bugs analiza automáticamente los archivos CUDA (`.cu` y `.cuh`) ubicados en `src/kernels/`.

Su objetivo es detectar problemas relacionados con la implementación GPU, tales como:

- ausencia de verificaciones mediante `CUDA_CHECK`;
- errores en el manejo de memoria CUDA;
- posibles desincronizaciones entre host y device;
- errores evidentes en pruebas unitarias;
- problemas lógicos asociados al comportamiento del kernel.

Cada problema encontrado es clasificado en una de dos categorías:

**Errores mecánicos**

Corresponden a errores simples que pueden corregirse automáticamente sin modificar la lógica del programa, por ejemplo:

- omisión de una llamada a `CUDA_CHECK`;
- errores tipográficos;
- pequeñas inconsistencias de implementación.

En estos casos el agente:

1. crea automáticamente una nueva rama;
2. aplica la corrección propuesta por el modelo;
3. realiza el commit;
4. genera automáticamente un Pull Request hacia `main`.

**Errores complejos**

Cuando el problema puede afectar:

- la física de la simulación;
- la API pública;
- la lógica de los kernels CUDA;
- la estabilidad numérica;

el agente no modifica el código.

En su lugar genera automáticamente un **Issue**, describiendo el problema detectado para que sea revisado por un desarrollador.

---

### 7.2 Agente Documentador

Este agente analiza automáticamente la documentación del proyecto, actualmente:

- `README.md`
- `CHANGELOG.md`

Su objetivo es detectar:

- errores ortográficos;
- problemas de formato;
- enlaces rotos;
- documentación incompleta;
- ausencia de explicaciones técnicas relevantes.

Al igual que el agente de bugs, clasifica los problemas en dos grupos.

Para errores puramente mecánicos genera automáticamente una rama con la documentación corregida y crea un Pull Request.

Cuando identifica deficiencias técnicas que requieren criterio de ingeniería (por ejemplo, falta de documentación sobre el funcionamiento de un kernel CUDA o sobre decisiones de diseño), abre automáticamente un Issue solicitando intervención humana sin modificar el repositorio.

---

### 7.3 Agente Revisor de Pull Requests

Cada Pull Request abierto hacia la rama principal es analizado automáticamente por un tercer agente especializado.

Este agente inspecciona el **diff completo** del Pull Request y determina si los cambios corresponden a modificaciones mecánicas o si requieren revisión manual.

Para ello considera criterios como:

- modificación exclusiva de documentación;
- cambios de formato;
- refactorizaciones sin alterar el comportamiento;
- modificaciones sobre la lógica física;
- cambios en la API pública;
- alteraciones en los kernels CUDA.

Además, consulta el estado del pipeline de integración continua antes de emitir su recomendación.

Como resultado publica automáticamente un comentario en el Pull Request indicando:

- estado del pipeline de CI;
- clasificación del cambio (mecánico o complejo);
- explicación del análisis realizado por el modelo;
- recomendación de revisión.

Por razones de seguridad, este agente **nunca realiza merges automáticos** hacia la rama `main`; la decisión final permanece bajo responsabilidad de un desarrollador.

---

### 7.4 Arquitectura del flujo de trabajo

El flujo completo implementado por los agentes puede resumirse de la siguiente forma:

1. Un desarrollador realiza un commit o abre un Pull Request.
2. GitHub Actions ejecuta automáticamente el agente correspondiente.
3. El agente obtiene el contenido del repositorio mediante la API de GitHub.
4. El contenido es analizado utilizando un modelo Gemini.
5. El modelo clasifica el resultado como **mecánico** o **complejo**.
6. Dependiendo de la clasificación:
   - se crea automáticamente un Pull Request con la corrección propuesta; o
   - se genera un Issue para revisión humana.
7. Finalmente, el agente registra el resultado dentro del repositorio mediante comentarios, Pull Requests o Issues.