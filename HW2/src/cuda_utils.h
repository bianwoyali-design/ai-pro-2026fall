#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace hw2::detail {

inline void cuda_check(cudaError_t error, const char *call, const char *file,
                       int line) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             "CUDA error at " + call + ": " +
                             cudaGetErrorString(error));
  }
}

inline void cuda_check_noexcept(cudaError_t error, const char *call,
                                const char *file, int line) noexcept {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s:%d CUDA error at %s: %s\n", file, line, call,
                 cudaGetErrorString(error));
  }
}
} // namespace hw2::detail

#define CUDA_CHECK(call)                                                       \
  ::hw2::detail::cuda_check((call), #call, __FILE__, __LINE__)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

#define CUDA_CHECK_NOEXCEPT(call)                                              \
  ::hw2::detail::cuda_check_noexcept((call), #call, __FILE__, __LINE__)
