#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Macro obligatoria del enunciado: envolver toda llamada a la API CUDA.
// Lanza std::runtime_error (en vez de abortar) para que los tests puedan reportar el fallo.
// ya que abort() termina el proceso y no permite que Catch2 informe del fallo.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t cuda_check_err_ = (call);                                     \
        if (cuda_check_err_ != cudaSuccess) {                                     \
            throw std::runtime_error(                                             \
                std::string("Error CUDA en ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + " -> " +                               \
                cudaGetErrorString(cuda_check_err_));                             \
        }                                                                         \
    } while (0)

// Verificación tras lanzar un kernel (los errores de lanzamiento son asíncronos).
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

#endif // CUDA_UTILS_H
