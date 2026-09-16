#include "activation_backend.h"

#include <cmath>
#include <cstddef>

namespace hw2::detail {

template <typename T> void relu_cpu(const T *x, T *y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    y[i] = x[i] > T{0} ? x[i] : T{0};
  }
}

template <typename T>
void relu_backward_cpu(const T *x, const T *grad_output, T *grad_input,
                       size_t n) {
  for (size_t i = 0; i < n; ++i) {
    grad_input[i] = x[i] > T{0} ? grad_output[i] : T{0};
  }
}

template <typename T> void sigmoid_cpu(const T *x, T *y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    y[i] = T{1} / (T{1} + std::exp(-x[i]));
  }
}

template <typename T>
void sigmoid_backward_cpu(const T *y, const T *grad_output, T *grad_input,
                          size_t n) {
  for (int i = 0; i < n; ++i) {
    grad_input[i] = grad_output[i] * y[i] * (1 - y[i]);
  }
}

template void relu_cpu<float>(const float *, float *, size_t);
template void relu_cpu<double>(const double *, double *, size_t);

template void relu_backward_cpu<float>(const float *, const float *, float *,
                                       size_t);
template void relu_backward_cpu<double>(const double *, const double *,
                                        double *, size_t);

template void sigmoid_cpu<float>(const float *, float *, size_t n);
template void sigmoid_cpu<double>(const double *, double *, size_t n);

template void sigmoid_backward_cpu<float>(const float *, const float *, float *,
                                          size_t);
template void sigmoid_backward_cpu<double>(const double *, const double *,
                                           double *, size_t);

} // namespace hw2::detail