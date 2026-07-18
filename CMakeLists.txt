cmake_minimum_required(VERSION 3.16)

project(nbody_simulation LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # El Lab 1 usa #pragma omp task/taskwait: MSVC los soporta solo con el
  # runtime LLVM de OpenMP (-openmp:llvm). Fuentes en UTF-8 (comentarios con
  # tildes) requieren /utf-8. Ambas opciones son no-op fuera de MSVC.
  set(OpenMP_RUNTIME_MSVC "llvm")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

find_package(OpenMP REQUIRED)

# ===== Lab2: soporte CUDA opcional =====
# Se habilita solo si hay CUDA Toolkit disponible; sin él, el build queda
# idéntico al del Lab 1 (la CI sin GPU sigue funcionando) y los métodos GPU
# lanzan una excepción en runtime.
option(ENABLE_CUDA "Compilar el soporte GPU (kernels CUDA del Lab 2)" ON)

set(NBODY_CUDA_ENABLED OFF)
if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      # native = la GPU de la máquina actual; en el clúster DIINF se puede
      # fijar con -DCMAKE_CUDA_ARCHITECTURES=<sm de la GPU del nodo>
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 75)
      endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(NBODY_CUDA_ENABLED ON)
    message(STATUS "CUDA habilitado (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "CUDA Toolkit no encontrado: build solo CPU (métodos GPU deshabilitados)")
  endif()
endif()

add_library(nbody_core
  src/Benchmark.cpp
  src/Integrator.cpp
  src/MetricsCalculator.cpp
  src/NBodySimulator.cpp
  src/NBodySystem.cpp
  src/Particle.cpp
  src/Visualizer.cpp
)

target_include_directories(nbody_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(nbody_core PUBLIC OpenMP::OpenMP_CXX)

if(NBODY_CUDA_ENABLED)
  target_sources(nbody_core PRIVATE
    src/NBodySystemGpu.cu
    kernels/accelerations.cu
  )
  target_include_directories(nbody_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  )
  # PUBLIC: NBodySystem.cpp y los tests deciden con este define si compilar
  # los stubs sin CUDA o usar las definiciones reales de NBodySystemGpu.cu
  target_compile_definitions(nbody_core PUBLIC NBODY_HAS_CUDA)
endif()

add_executable(nbody_app src/main.cpp)

target_link_libraries(nbody_app PRIVATE nbody_core)

# Copiar el script de graficación al directorio de construcción
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/plot.py
    ${CMAKE_CURRENT_BINARY_DIR}/plot.py
    COPYONLY
)

include(CTest)

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(nbody_tests
    tests/test_main.cpp
    tests/test_NbodySimulator.cpp
    tests/test_NBodySystem.cpp
    tests/test_Particle.cpp
    tests/test_metrics.cpp
    tests/test_integrator.cpp
  )

  target_link_libraries(nbody_tests PRIVATE nbody_core GTest::gtest_main)

  add_executable(test_benchmark tests/test_Benchmark.cpp)
  target_link_libraries(test_benchmark PRIVATE nbody_core GTest::gtest_main)

  add_test(NAME test_benchmark COMMAND test_benchmark)

  # Tests GPU (Lab2): solo se compilan si hay CUDA; requieren una GPU NVIDIA
  if(NBODY_CUDA_ENABLED)
    add_executable(nbody_gpu_tests tests/test_CudaMemory.cu)
    target_link_libraries(nbody_gpu_tests PRIVATE nbody_core GTest::gtest_main)
    add_test(NAME nbody_gpu_tests COMMAND nbody_gpu_tests)
  endif()

  include(GoogleTest)
  gtest_discover_tests(nbody_tests)
endif()