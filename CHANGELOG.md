# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0-lab2] - 2026-08-03

### Added

- **Capa host/device y memoria CUDA (Rol 2, Lab 2)**:
  - `CudaBuffer<T>`: buffer RAII de memoria device (`cudaMalloc`/`cudaFree`, move-only, transferencias H2D/D2H verificadas).
  - `CudaUtils.h`: macros `CUDA_CHECK` y `CUDA_CHECK_KERNEL` para verificación de errores CUDA.
  - `DeviceNBodyState`: estado SoA en device (`d_mass`, `d_x`, `d_y`, `d_ax`, `d_ay`) con esquema de copias mínimas por paso (masas una sola vez; velocidades nunca tocan el device).
  - Implementación de `NBodySystem::computeAccelerationsGpu()` (3 sobrecargas) y `NBodySimulator::stepEulerGpu()` con el orden fijo del enunciado (kernel → sync → Euler en host).
- **Kernels CUDA (Rol 1, Lab 2)**:
  - Integración de variante básica y variante con memoria compartida (tiles + `__syncthreads()`), conectados a los buffers SoA en `src/kernels/` con API unificada a `computeAccelerationsGpu` (PR #5, PR #29).
- **Infraestructura y Agentes de IA (Rol 4, Lab 2)**:
  - Imagen y contenedor Docker para desarrollo CUDA (`Dockerfile` / PR #28, PR #42).
  - Documentación, pruebas e integración de los tres agentes de IA (Documentador, Bugs y Merge Request) en scripts de CI (Issue #27, PR #31, PR #36, PR #39, PR #41).
  - Soporte CUDA opcional en CMake (`ENABLE_CUDA`, autodetectado) para mantener builds y CI funcionales sin GPU.
- **Pruebas y Benchmarking (Rol 5, Lab 2)**:
  - Suite `nbody_gpu_tests`: round-trip de buffers, caso analítico de 2 cuerpos, equivalencia CPU vs GPU ($rtol=10^{-4}$, $atol=10^{-8}$), independencia de `blockDim.x` e integración de 10 pasos.
  - Cronometraje con `std::chrono::steady_clock`, batería de benchmarks en clúster DIINF y generación automatizada de gráficos (PR #29, commit `d20eb8d`).

### Changed

- `NBodySystem` ahora es no-copiable y cuenta con destructor propio para el manejo del estado GPU opaco.
- Ajuste de flags de CMake para MSVC (`/openmp:llvm` para tareas OpenMP del Lab 1, `/utf-8`).
- Sobrecarga de `computeAccelerationsGpuKernelOnly` y `benchmarkKernelOnly` con parámetro `steps` (100 por defecto) para promediar iteraciones manteniendo compatibilidad con la API.
- Reconfiguración de los agentes de IA (Documentador y Bugs) ajustando sus alcances y detonantes para optimizar el consumo de tokens (PR #43, commits `eb39963`, `3426296`).
- Actualización de los flujos de generación de gráficos de rendimiento y análisis de Amdahl, preservando las métricas obtenidas en el Lab 1 (PR #45, commit `67587a1`).

### Fixed

- **Estabilización y Normalización de Benchmarks**:
  - `benchmarkKernelOnly` y `benchmarkEndToEnd` promedian ahora el mismo número de iteraciones con un `cudaDeviceSynchronize()` por paso, reduciendo el ruido relativo a ~2.1% y eliminando sobrecargas negativas ($EndToEnd - KernelOnly < 0$) (PR #35, PR #47).
  - Corrección de baseline CPU (`benchmarkCpuKernelOnly`) para evitar el inflado del tiempo de cómputo en `CpuKernelMean_s` y corregir la distorsión del speedup en la curva de Amdahl.
  - Implementación de calentamiento previo de GPU (`WarmUpGpu` por 100 ms) para prevenir la absorción de la rampa de subida de frecuencias en la primera medición de cada bloque.
- **Correcciones de Código y Entorno**:
  - Corrección en la suite de ejecuciones de pruebas para el clúster DIINF (PR #48).
  - Omisión condicional de pruebas de energía CUDA en entornos de CI carentes de GPU física (commit `08df09b`).
  - Correcciones mecánicas automáticas en `src/NBodySystemGpu.cu` y formato en `README.md` generadas por los agentes de IA (commits `c1500f7`, `bc11d9c`, `2f4005c`).

## [1.1.6] - 2026-07-28

### Changed

- Agente de Bugs ahora revisa todo el código.
- Se cambió el gatillante del agente documentador.

## [1.1.5] - 2026-07-17

### Changed

- Cambio de librería de Google AI.

## [1.1.4] - 2026-07-17

### Changed

- Cambio de requirements para no especificar versionado.

## [1.1.3] - 2026-07-17

### Changed

- Se cambió la versión de Python para evitar warnings.
- Se cambió el modelo de IA a latest.

## [1.1.2] - 2026-07-17

### Fixed

- Error de tipeo dentro del agente MR

## [1.1.1] - 2026-07-17

### Fixed

- Error de tipeo dentro del agente MR.

## [1.1.0] - 2026-07-07

### Added

- Agentes de Documentación, Bugs y Merge Request.

## [1.0.0] - 2026-07-06

### Added

- Archivos básicos necesarios.
- Declaración de métodos mínimos necesarios.