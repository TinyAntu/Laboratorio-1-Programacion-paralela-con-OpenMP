cmake_minimum_required(VERSION 3.16)

project(nbody_simulation LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

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

  include(GoogleTest)
  gtest_discover_tests(nbody_tests)
endif()