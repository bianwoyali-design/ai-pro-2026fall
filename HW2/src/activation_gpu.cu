#include "activation_backend.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace hw2::detail {

namespace {
template <typename T>
__global__ __forceinline__ void relu_kernel(const T *__restrict__ x,
                                            T *__restrict__ y, size_t n) {
  const size_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  for (size_t i = global_tid; i < n; i += stride) {
    y[i] = x[i] > T{0} ? x[i] : T{0};
  }
}

template <typename T>
__global__ __forceinline__ void
relu_backward_kernel(const T *__restrict__ x, const T *__restrict__ grad_output,
                     T *__restrict__ grad_input, size_t n) {
  const size_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  for (size_t i = global_tid; i < n; i += stride) {
    grad_input[i] = x[i] > T{0} ? grad_output[i] : T{0};
  }
}

template <typename T>
__global__ __forceinline__ void sigmoid_kernel(const T *__restrict__ x,
                                               T *__restrict__ y, size_t n) {
  const size_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  for (size_t i = global_tid; i < n; i += stride) {
    y[i] = T{1} / (T{1} + std::exp(-x[i]));
  }
}

template <typename T>
__global__ __forceinline__ void
sigmoid_backward_kernel(const T *__restrict__ y,
                        const T *__restrict__ grad_output,
                        T *__restrict__ grad_input, size_t n) {
  const size_t global_tid = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  for (size_t i = global_tid; i < n; i += stride) {
    grad_input[i] = grad_output[i] * y[i] * (1 - y[i]);
  }
}
} // namespace

template <typename T> void relu_cuda(const T *x, T *y, size_t n) {
  if (!n) {
    return;
  }

  constexpr int threads = 256;

  constexpr int devId = 0;
  int sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devId));

  const int blocks = std::min(static_cast<int>((n - 1) / threads + 1), 4 * sms);

  relu_kernel<<<blocks, threads>>>(x, y, n);
  CUDA_CHECK_KERNEL();
}

template <typename T>
void relu_backward_cuda(const T *x, const T *grad_output, T *grad_input,
                        size_t n) {
  if (!n) {
    return;
  }

  constexpr int threads = 256;

  constexpr int devId = 0;
  int sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devId));

  const int blocks = std::min(static_cast<int>((n - 1) / threads + 1), 4 * sms);

  relu_backward_kernel<<<blocks, threads>>>(x, grad_output, grad_input, n);
  CUDA_CHECK_KERNEL();
}

template <typename T> void sigmoid_cuda(const T *x, T *y, size_t n) {
  if (!n) {
    return;
  }

  constexpr int threads = 256;

  constexpr int devId = 0;
  int sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devId));

  const int blocks = std::min(static_cast<int>((n - 1) / threads + 1), 4 * sms);

  sigmoid_kernel<<<blocks, threads>>>(x, y, n);
  CUDA_CHECK_KERNEL();
}

template <typename T>
void sigmoid_backward_cuda(const T *y, const T *grad_output, T *grad_input,
                           size_t n) {
  if (!n) {
    return;
  }

  constexpr int threads = 256;

  constexpr int devId = 0;
  int sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, devId));

  const int blocks = std::min(static_cast<int>((n - 1) / threads + 1), 4 * sms);

  sigmoid_backward_kernel<<<blocks, threads>>>(y, grad_output, grad_input, n);
  CUDA_CHECK_KERNEL();
}

template void relu_cuda<float>(const float *, float *, size_t);
template void relu_cuda<double>(const double *, double *, size_t);

template void relu_backward_cuda<float>(const float *, const float *, float *,
                                        size_t);
template void relu_backward_cuda<double>(const double *, const double *,
                                         double *, size_t);

template void sigmoid_cuda<float>(const float *, float *, size_t);
template void sigmoid_cuda<double>(const double *, double *, size_t);

template void sigmoid_backward_cuda<float>(const float *, const float *,
                                           float *, size_t);
template void sigmoid_backward_cuda<double>(const double *, const double *,
                                            double *, size_t);

} // namespace hw2::detail
