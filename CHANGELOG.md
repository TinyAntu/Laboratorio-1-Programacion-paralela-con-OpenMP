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

## [1.0.0] 06-07-2026

### Added


- Archivos básicos necesarios (`CudaBuffer`).
- Declaración de métodos mínimos necesarios (dentro de `NBodySystem`, `NBodySimulator`, `Benchmark`).

- Archivos básicos necesarios (CudaBuffer)
- Declaración de métodos mínimos necesarios (dentro de NBodySystem, NBodySimulator, Benchmark)


## [1.1.0] 07-07-2026

### Added


- Agentes de Documentación, Bugs y Merge Request.

- Agentes de Documentación, Bugs y Merge Request

## [1.1.1] 17-07-2026

### Fixed


- Error de tipeo dentro del agente MR.

- Error de tipeo dentro del agente MR

## [1.1.2] 17-07-2026

### Fixed


- Error de tipeo dentro del agente MR.

- Error de tipeo dentro del agente MR


## [1.1.3] 17-07-2026

### Changed


- Se cambió la versión de Python para evitar warning.
- Se cambió el modelo de IA a latest.

- Se cambió la versión de Python para evitar warnings
- Se cambió el modelo de IA a latest


## [1.1.4] 17-07-2026

### Changed


- Cambio de requirements para no especificar versionado.
- Cambio de requirements para no especificar versionado.

## [1.1.5] 17-07-2026

### Changed


- Cambio de librería de Google AI.

- Cambio de librería de Google AI
