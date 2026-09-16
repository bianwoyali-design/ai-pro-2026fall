#include "hw2/activation.h"
#include "activation_backend.h"

#include <stdexcept>

namespace hw2 {

template <typename T> Tensor<T> relu(const Tensor<T> &x) {
  Tensor<T> y(x.shape(), x.device());

  if (x.device() == Device::CPU) {
    detail::relu_cpu(x.data(), y.data(), x.numel());
  } else {
    detail::relu_cuda(x.data(), y.data(), x.numel());
  }

  return y;
}

template <typename T>
Tensor<T> relu_backward(const Tensor<T> &x, const Tensor<T> &grad_output) {
  if (x.shape() != grad_output.shape()) {
    throw std::invalid_argument("Gradient shape mismatch");
  }
  if (x.device() != grad_output.device()) {
    throw std::invalid_argument("Gradient device mismatch");
  }

  Tensor<T> grad_input(x.shape(), x.device());

  if (x.device() == Device::CPU) {
    detail::relu_backward_cpu(x.data(), grad_output.data(), grad_input.data(),
                              x.numel());
  } else {
    detail::relu_backward_cuda(x.data(), grad_output.data(), grad_input.data(),
                               x.numel());
  }

  return grad_input;
}

template <typename T> Tensor<T> sigmoid(const Tensor<T> &x) {
  Tensor<T> y(x.shape(), x.device());

  if (x.device() == Device::CPU) {
    detail::sigmoid_cpu(x.data(), y.data(), x.numel());
  } else {
    detail::sigmoid_cuda(x.data(), y.data(), x.numel());
  }

  return y;
}

template <typename T>
Tensor<T> sigmoid_backward(const Tensor<T> &y, const Tensor<T> &grad_output) {
  if (y.shape() != grad_output.shape()) {
    throw std::invalid_argument("Gradient shape mismatch");
  }
  if (y.device() != grad_output.device()) {
    throw std::invalid_argument("Gradient device mismatch");
  }

  Tensor<T> grad_input(y.shape(), y.device());

  if (y.device() == Device::CPU) {
    detail::sigmoid_backward_cpu(y.data(), grad_output.data(),
                                 grad_input.data(), y.numel());
  } else {
    detail::sigmoid_backward_cuda(y.data(), grad_output.data(),
                                  grad_input.data(), y.numel());
  }

  return grad_input;
}

template Tensor<float> relu<float>(const Tensor<float> &);
template Tensor<double> relu<double>(const Tensor<double> &);

template Tensor<float> relu_backward<float>(const Tensor<float> &,
                                            const Tensor<float> &);
template Tensor<double> relu_backward<double>(const Tensor<double> &,
                                              const Tensor<double> &);

template Tensor<float> sigmoid<float>(const Tensor<float> &);
template Tensor<double> sigmoid<double>(const Tensor<double> &);

template Tensor<float> sigmoid_backward<float>(const Tensor<float> &,
                                               const Tensor<float> &);
template Tensor<double> sigmoid_backward<double>(const Tensor<double> &,
                                                 const Tensor<double> &);

} // namespace hw2
