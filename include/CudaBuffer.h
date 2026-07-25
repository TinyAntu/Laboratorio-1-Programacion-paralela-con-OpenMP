// CudaBuffer: buffer RAII de memoria device (cudaMalloc/cudaFree).
// Dueño único del puntero device: no copiable, sí movible.
// Este header solo debe incluirse desde código compilado con nvcc (archivos .cu).
#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include "CudaUtils.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T>
class CudaBuffer {
private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0; // número de elementos T

public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        release();
    }

    // Dueño único del recurso device: prohibimos copia
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Reserva count elementos en device (libera lo anterior si existía)
    void allocate(std::size_t count) {
        release();
        if (count == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        size_ = count;
    }

    // Libera la memoria device. noexcept: en destrucción no se debe lanzar.
    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_); // no CUDA_CHECK: release se usa en el destructor
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    // Transferencias H2D / D2H (siempre con verificación de tamaño y CUDA_CHECK)
    void copyToDevice(const T* host_src, std::size_t count) {
        checkBounds(count);
        CUDA_CHECK(cudaMemcpy(ptr_, host_src, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    void copyToDevice(const std::vector<T>& host_src) {
        copyToDevice(host_src.data(), host_src.size());
    }

    void copyToHost(T* host_dst, std::size_t count) const {
        checkBounds(count);
        CUDA_CHECK(cudaMemcpy(host_dst, ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    void copyToHost(std::vector<T>& host_dst) const {
        host_dst.resize(size_);
        copyToHost(host_dst.data(), size_);
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    void checkBounds(std::size_t count) const {
        if (ptr_ == nullptr) {
            throw std::logic_error("CudaBuffer: buffer no reservado (allocate primero)");
        }
        if (count > size_) {
            throw std::out_of_range("CudaBuffer: transferencia excede el tamano reservado");
        }
    }
};

#endif // CUDA_BUFFER_H
