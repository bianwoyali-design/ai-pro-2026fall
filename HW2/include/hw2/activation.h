#pragma once

#include "hw2/tensor.h"

namespace hw2 {

template <typename T> Tensor<T> relu(const Tensor<T> &x);
template <typename T>
Tensor<T> relu_backward(const Tensor<T> &x, const Tensor<T> &grad_output);

template <typename T> Tensor<T> sigmoid(const Tensor<T> &x);
template <typename T>
Tensor<T> sigmoid_backward(const Tensor<T> &y, const Tensor<T> &grad_output);

} // namespace hw2