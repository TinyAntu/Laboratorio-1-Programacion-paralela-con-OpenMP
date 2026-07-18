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
  - Kernel básico provisional de aceleraciones (`kernels/accelerations.cu`, un hilo por cuerpo) con lanzador host y división techo; interfaz `launchAccelerationsShared` pendiente (Rol 1).
  - Implementación de `NBodySystem::computeAccelerationsGPU()` (3 sobrecargas) y `NBodySimulator::stepEulerGpu()` con el orden fijo del enunciado (kernel → sync → Euler en host).
  - Tests GPU (`nbody_gpu_tests`): round-trip de buffers, caso analítico de 2 cuerpos, equivalencia CPU vs GPU (rtol=1e-4, atol=1e-8), independencia del block size e integración de 10 pasos.
  - Soporte CUDA opcional en CMake (`ENABLE_CUDA`, autodetectado): sin CUDA Toolkit el build queda idéntico al Lab 1 y la CI sin GPU sigue funcionando.

### Changed

- `NBodySystem` ahora es no-copiable y con destructor propio (dueño del estado GPU opaco).
- CMake: flags para MSVC (`/openmp:llvm` para las tareas OpenMP del Lab 1, `/utf-8`).

## [1.0.0] 06-07-2026

### Added

-Archivos basicos necesarios (CudaBuffer)
-Declaracion de metodos minimos necesarios (dentro de NBodySystem, NBodySimulator, Benchmark)

## [1.1.0] 07-07-2026

### Added

-Agentes de Documentacion, Bugs y Merge request

## [1.1.1] 17-07-2026

### Fixed

-Error de typeo dentro del agente MR

## [1.1.2] 17-07-2026

### Fixed

-Error de typeo dentro del agente MR

## [1.1.3] 17-07-2026

### Changed

-Se cambio la version de python para evitar warning
-Se cambio el modelo de IA a latest

## [1.1.4] 17-07-2026

### Changed

-Cambio de requirements para no especificar versionado

## [1.1.5] 17-07-2026

### Changed

-Cambio de libreria de googleIA