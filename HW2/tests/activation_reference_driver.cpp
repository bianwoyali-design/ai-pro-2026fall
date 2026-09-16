#include "hw2/activation.h"
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Text protocol: rank, dimensions, input values, upstream gradient values.
// Output: flattened ReLU, ReLU backward, Sigmoid, Sigmoid backward.
template <class T> void run(hw2::Device device) {
  size_t rank;
  if (!(std::cin >> rank) || rank > 8)
    throw std::runtime_error("Invalid rank");
  std::vector<uint64_t> shape(rank);
  size_t n = 1;
  for (auto &dim : shape) {
    if (!(std::cin >> dim) || dim > 1000000 || (dim && n > 1000000 / dim))
      throw std::runtime_error("Invalid test shape");
    n *= dim;
  }
  hw2::Tensor<T> x(shape, hw2::Device::CPU), g(shape, hw2::Device::CPU);
  for (size_t i = 0; i < n; ++i)
    if (!(std::cin >> x.data()[i]))
      throw std::runtime_error("Missing input");
  for (size_t i = 0; i < n; ++i)
    if (!(std::cin >> g.data()[i]))
      throw std::runtime_error("Missing gradient");
  if (device == hw2::Device::GPU) {
    x = x.gpu();
    g = g.gpu();
  }
  auto r = hw2::relu(x);
  auto dr = hw2::relu_backward(x, g);
  auto s = hw2::sigmoid(x);
  auto ds = hw2::sigmoid_backward(s, g);
  std::cout << std::setprecision(std::numeric_limits<T>::max_digits10);
  for (const auto *output : {&r, &dr, &s, &ds}) {
    if (output->shape() != shape || output->device() != device)
      throw std::runtime_error("Output metadata mismatch");
    auto host = output->cpu();
    for (size_t i = 0; i < n; ++i)
      std::cout << host.data()[i] << ' ';
    std::cout << '\n';
  }
}

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Expected cpu|gpu float|double");
    std::string backend = argv[1], dtype = argv[2];
    if ((backend != "cpu" && backend != "gpu") ||
        (dtype != "float" && dtype != "double"))
      throw std::runtime_error("Invalid backend or dtype");
    if (backend == "gpu") {
      int count = 0;
      auto error = cudaGetDeviceCount(&count);
      if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
          (error == cudaSuccess && count == 0)) {
        std::cerr << "CUDA unavailable: " << cudaGetErrorString(error) << '\n';
        return 77;
      }
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
      error = cudaSetDevice(0);
      if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    }
    auto device = backend == "gpu" ? hw2::Device::GPU : hw2::Device::CPU;
    if (dtype == "float")
      run<float>(device);
    else
      run<double>(device);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
