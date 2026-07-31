# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Capa host/device y memoria CUDA (Rol 2, Lab 2):
  - `CudaBuffer<T>`: buffer RAII de memoria device (cudaMalloc/cudaFree, move-only, transferencias H2D/D2H verificadas).
  - `CudaUtils.h`: macros `CUDA_CHECK` y `CUDA_CHECK_KERNEL` para verificación de errores CUDA.
  - `DeviceNBodyState`: estado SoA en device (d_mass, d_x, d_y, d_ax, d_ay) con esquema de copias mínimas por paso (masas una sola vez; velocidades nunca tocan el device).
  - Implementación de `NBodySystem::computeAccelerationsGpu()` (3 sobrecargas) y `NBodySimulator::stepEulerGpu()` con el orden fijo del enunciado (kernel → sync → Euler en host).
- Integración de los kernels CUDA del Rol 1 (PR #5): variante básica y variante con memoria compartida (tiles + `__syncthreads()`), conectados a los buffers SoA del Rol 2 en `src/kernels/`; API unificada a `computeAccelerationsGpu` (nomenclatura del enunciado §5.3).
  - Tests GPU (`nbody_gpu_tests`): round-trip de buffers, caso analítico de 2 cuerpos, equivalencia CPU vs GPU (rtol=1e-4, atol=1e-8), independencia del block size e integración de 10 pasos.
  - Soporte CUDA opcional en CMake (`ENABLE_CUDA`, autodetectado): sin CUDA Toolkit el build queda idéntico al Lab 1 y la CI sin GPU sigue funcionando.

### Changed

- `NBodySystem` ahora es no-copiable y con destructor propio (dueño del estado GPU opaco).
- CMake: flags para MSVC (`/openmp:llvm` para las tareas OpenMP del Lab 1, `/utf-8`).
- `computeAccelerationsGpuKernelOnly` y `benchmarkKernelOnly` aceptan un parámetro `steps` (por defecto 100) y devuelven el tiempo promedio por lanzamiento, en lugar del tiempo de un único lanzamiento. El parámetro tiene valor por defecto, así que las firmas `benchmarkKernelOnly()` y `computeAccelerationsGpuKernelOnly(variant)` que exige el enunciado (§5.3) siguen siendo válidas.

### Fixed

- Medición de benchmarks GPU: `benchmarkKernelOnly` cronometraba un único lanzamiento mientras `benchmarkEndToEnd` promediaba 100 pasos. Al no amortizarse la latencia de lanzamiento y sincronización, la medición kernel-only quedaba con ~48 % de ruido relativo (mediana; hasta 184 % en el peor punto) frente al ~6.5 % del end-to-end. Como consecuencia, 12 de 40 combinaciones locales (2 de 40 en el clúster) daban sobrecarga negativa `EndToEndMean_s - KernelOnlyMean_s < 0` —físicamente imposible, porque el paso completo incluye al kernel más las transferencias—, `SerialFraction` se saturaba a 0 y la predicción de Amdahl degeneraba en la curva kernel-only, quedando incluso por debajo de la medición en N=2000 pese a ser una cota superior.

  Ahora ambas mediciones promedian el mismo número de lanzamientos, con un `cudaDeviceSynchronize()` por iteración, de modo que la resta `e2e - kernel` aísla transferencias H2D/D2H más el Euler en host, que es la fracción serial mínima que exige §8.1.5 del enunciado. **No se modificó ningún kernel ni ninguna fórmula física**: los resultados numéricos de la simulación son idénticos.

  Tras este cambio, el ruido de la medición kernel-only bajó de ~48 % a ~2.1 % (mediana) y las filas con sobrecarga negativa cayeron de 12 a 2 de 40.

- Baseline CPU del speedup de kernel: `runGpuBenchmarks` obtenía `CpuKernelMean_s` de `benchmarkSerial(1)`, que cronometra **una única** llamada a `computeAccelerations()` por repetición, mientras `CpuStepMean_s` venía de `benchmarkCpuEndToEnd`, que promedia 100 pasos. Era la misma asimetría del lado GPU, pero en las columnas CPU: a N=1024 el baseline salía inflado ~28 % (20 513 µs frente a los 16 553 µs del paso completo, es decir `CpuStep < CpuKernel`, imposible porque el paso incluye al kernel) y el escalado dejaba de seguir la ley O(N²) esperada (3.39×, 4.91×, 3.08× medidos contra 4.00×, 4.00×, 3.81× teóricos, mientras `CpuStep` daba 3.65×, 3.52×, 3.83×).

  Como `KernelSpeedup = CpuKernelMean_s / KernelOnlyMean_s`, ese sesgo se propagaba al numerador del speedup y de ahí a la curva de Amdahl. Se añadió `Benchmark::benchmarkCpuKernelOnly(steps)`, espejo exacto de `benchmarkCpuEndToEnd` pero sin integrar, y `runGpuBenchmarks` ahora lo usa. Con esto las cuatro columnas de tiempo del `.dat` (`CpuKernel`, `CpuStep`, `KernelOnly`, `EndToEnd`) promedian el mismo número de iteraciones. `benchmarkSerial` se mantiene sin cambios porque `runScalingAnalysis` lo usa para los benchmarks OpenMP del Lab 1.

- Rampa de frecuencia de la GPU antes de medir: `runGpuBenchmarks` ejecuta los baselines CPU (`benchmarkSerial` y `benchmarkCpuEndToEnd`, segundos de trabajo en host) antes de la primera medición GPU de cada `N`, durante los cuales la GPU baja de reloj. Como el recorrido es `variante {0,1} × blockDim {64…1024}`, la combinación `variant=0, blockDim=64` era siempre la primera medida y absorbía la rampa de subida, con hasta 81 % de ruido relativo — y era justamente el `blockDim` que `plot.py` elegía para la figura de Amdahl. Se añadió `Benchmark::warmUpGpu`, un calentamiento por tiempo fijo (100 ms) que se ejecuta una vez por combinación, fuera del bucle de repeticiones, en `benchmarkKernelOnly` y en `benchmarkEndToEnd`. Es por tiempo y no por número de lanzamientos porque la duración de un lanzamiento depende de `N`.

## [1.0.0] - 06-07-2026

### Added

- Archivos básicos necesarios (`CudaBuffer`).
- Declaración de métodos mínimos necesarios (dentro de `NBodySystem`, `NBodySimulator`, `Benchmark`).

## [1.1.0] - 07-07-2026

### Added

- Agentes de Documentación, Bugs y Merge Request.

## [1.1.1] - 17-07-2026

### Fixed

- Error de tipeo dentro del agente MR.

## [1.1.2] - 17-07-2026

### Fixed

- Error de tipeo dentro del agente MR.

## [1.1.3] - 17-07-2026

### Changed

- Se cambió la versión de Python para evitar warnings.
- Se cambió el modelo de IA a latest.

## [1.1.4] - 17-07-2026

### Changed

- Cambio de requirements para no especificar versionado.

## [1.1.5] - 17-07-2026

### Changed

- Cambio de librería de Google AI.

## [1.1.6] - 28-07-2026

### Changed

- Agente de Bugs ahora revisa todo el código.
- Se cambió el gatillante del agente documentador.
