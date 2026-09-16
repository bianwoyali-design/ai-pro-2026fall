#pragma once

#include <cstddef>

namespace hw2::detail {

template <typename T> void relu_cpu(const T *x, T *y, size_t n);
template <typename T>
void relu_backward_cpu(const T *x, const T *grad_output, T *grad_input,
                       size_t n);

template <typename T> void sigmoid_cpu(const T *x, T *y, size_t n);
template <typename T>
void sigmoid_backward_cpu(const T *y, const T *grad_output, T *grad_input,
                          size_t n);

template <typename T> void relu_cuda(const T *x, T *y, size_t n);
template <typename T>
void relu_backward_cuda(const T *x, const T *grad_output, T *grad_input,
                        size_t n);

template <typename T> void sigmoid_cuda(const T *x, T *y, size_t n);
template <typename T>
void sigmoid_backward_cuda(const T *y, const T *grad_output, T *grad_input,
                           size_t n);

} // namespace hw2::detail