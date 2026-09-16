#include "hw2/activation.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using hw2::Device;
using hw2::Tensor;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class F> void expect_invalid(F function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("Expected std::invalid_argument");
}

void close(double actual, double expected, double tolerance,
           const std::string &context) {
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <=
                  tolerance * (1 + std::abs(expected)),
          context + ": actual=" + std::to_string(actual) +
              ", expected=" + std::to_string(expected));
}

template <class T> Tensor<T> on_device(const Tensor<T> &tensor, Device device) {
  return device == Device::CPU ? tensor : tensor.gpu();
}

template <class T>
void values_and_shapes(Device device, const std::vector<uint64_t> &shape) {
  Tensor<T> host_x(shape, Device::CPU), host_g(shape, Device::CPU);
  const double inputs[] = {-100, -8, -2, -0.25, 0, 0.25, 2, 8, 100};
  for (size_t i = 0; i < host_x.numel(); ++i) {
    host_x.data()[i] = static_cast<T>(inputs[i % 9]);
    host_g.data()[i] = static_cast<T>((static_cast<int>(i % 7) - 3) * 0.5);
  }

  auto x = on_device(host_x, device);
  auto g = on_device(host_g, device);
  auto r = hw2::relu(x);
  auto s = hw2::sigmoid(x);
  auto dr = hw2::relu_backward(x, g);
  auto ds = hw2::sigmoid_backward(s, g);

  for (const auto *output : {&r, &s, &dr, &ds}) {
    require(output->shape() == shape, "Output shape changed");
    require(output->device() == device, "Output device changed");
    require(output->numel() == x.numel(), "Output size changed");
    if (x.numel()) {
      require(output->data() != x.data() && output->data() != g.data(),
              "Output aliases an input");
    }
  }

  auto hr = r.cpu(), hs = s.cpu(), hdr = dr.cpu(), hds = ds.cpu();
  auto unchanged_x = x.cpu(), unchanged_g = g.cpu();
  const double tolerance = std::is_same_v<T, float> ? 2e-6 : 1e-12;
  for (size_t i = 0; i < host_x.numel(); ++i) {
    double value = host_x.data()[i], gradient = host_g.data()[i];
    // Independent reference identity, evaluated in double precision.
    double sigmoid = 0.5 * (1 + std::tanh(value / 2));
    close(hr.data()[i], std::max(value, 0.0), 0, "ReLU forward");
    close(hdr.data()[i], value > 0 ? gradient : 0, 0, "ReLU backward");
    close(hs.data()[i], sigmoid, tolerance, "Sigmoid forward");
    close(hds.data()[i], gradient * sigmoid * (1 - sigmoid), tolerance,
          "Sigmoid backward");
    require(hs.data()[i] >= T{0} && hs.data()[i] <= T{1},
            "Sigmoid outside [0, 1]");
    require(unchanged_x.data()[i] == host_x.data()[i] &&
                unchanged_g.data()[i] == host_g.data()[i],
            "Input modified");
  }

  if (device == Device::GPU) {
    auto cr = hw2::relu(host_x), cs = hw2::sigmoid(host_x);
    auto cdr = hw2::relu_backward(host_x, host_g);
    auto cds = hw2::sigmoid_backward(cs, host_g);
    for (size_t i = 0; i < host_x.numel(); ++i) {
      close(hr.data()[i], cr.data()[i], tolerance, "CPU/GPU ReLU");
      close(hs.data()[i], cs.data()[i], tolerance, "CPU/GPU Sigmoid");
      close(hdr.data()[i], cdr.data()[i], tolerance, "CPU/GPU ReLU gradient");
      close(hds.data()[i], cds.data()[i], tolerance,
            "CPU/GPU Sigmoid gradient");
    }
  }
}

template <class T> void numerical_gradients(Device device) {
  Tensor<T> host_x({4}, Device::CPU), host_g({4}, Device::CPU);
  const double values[] = {-2, -0.5, 0.5, 2}; // Avoid ReLU's kink at zero.
  const double gradients[] = {2, -3, 0.5, 1.25};
  for (size_t i = 0; i < 4; ++i) {
    host_x.data()[i] = static_cast<T>(values[i]);
    host_g.data()[i] = static_cast<T>(gradients[i]);
  }
  const T epsilon = std::is_same_v<T, float> ? T(1e-3) : T(1e-5);
  const double tolerance = std::is_same_v<T, float> ? 5e-4 : 1e-8;
  auto x = on_device(host_x, device), g = on_device(host_g, device);

  for (bool sigmoid : {false, true}) {
    auto analytical = (sigmoid ? hw2::sigmoid_backward(hw2::sigmoid(x), g)
                               : hw2::relu_backward(x, g))
                          .cpu();
    auto loss = [&](const Tensor<T> &input) {
      auto d = on_device(input, device);
      auto output = (sigmoid ? hw2::sigmoid(d) : hw2::relu(d)).cpu();
      double result = 0;
      for (size_t j = 0; j < 4; ++j)
        result += double(output.data()[j]) * host_g.data()[j];
      return result;
    };
    for (size_t i = 0; i < 4; ++i) {
      auto plus = host_x.clone(), minus = host_x.clone();
      plus.data()[i] += epsilon;
      minus.data()[i] -= epsilon;
      double numerical = (loss(plus) - loss(minus)) /
                         (double(plus.data()[i]) - double(minus.data()[i]));
      close(analytical.data()[i], numerical, tolerance,
            sigmoid ? "Sigmoid finite difference" : "ReLU finite difference");
    }
  }
}

template <class T> void invalid_inputs(Device device) {
  Tensor<T> x({2, 3}, device);
  // Same element count is not sufficient: full shapes must match.
  Tensor<T> wrong_shape({3, 2}, device);
  expect_invalid([&] { hw2::relu_backward(x, wrong_shape); });
  expect_invalid([&] { hw2::sigmoid_backward(x, wrong_shape); });
  Tensor<T> wrong_count({5}, device);
  expect_invalid([&] { hw2::relu_backward(x, wrong_count); });
  expect_invalid([&] { hw2::sigmoid_backward(x, wrong_count); });
  // Empty tensors carry device metadata without requiring a GPU allocation.
  Tensor<T> cpu_empty({0}, Device::CPU), gpu_empty({0}, Device::GPU);
  expect_invalid([&] { hw2::relu_backward(cpu_empty, gpu_empty); });
  expect_invalid([&] { hw2::sigmoid_backward(cpu_empty, gpu_empty); });
  expect_invalid([&] { hw2::relu_backward(gpu_empty, cpu_empty); });
  expect_invalid([&] { hw2::sigmoid_backward(gpu_empty, cpu_empty); });
}

template <class T> void run(Device device) {
  for (const std::vector<uint64_t> &shape :
       std::vector<std::vector<uint64_t>>{{}, {0, 3}, {2, 3}, {9}, {257}})
    values_and_shapes<T>(device, shape);
  if (device == Device::GPU) {
    int sms = 0;
    auto error =
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    require(error == cudaSuccess, cudaGetErrorString(error));
    // Exceeds the current 4*SM block cap: exercise repeated grid-stride
    // iterations.
    values_and_shapes<T>(device, {uint64_t(sms) * 4 * 256 * 2 + 17});
  }
  numerical_gradients<T>(device);
  invalid_inputs<T>(device);
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "Usage: test_activation cpu|gpu float|double");
    std::string backend = argv[1], dtype = argv[2];
    require(backend == "cpu" || backend == "gpu", "Unknown backend");
    require(dtype == "float" || dtype == "double", "Unknown dtype");
    Device device = backend == "gpu" ? Device::GPU : Device::CPU;
    if (device == Device::GPU) {
      int count = 0;
      auto error = cudaGetDeviceCount(&count);
      if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
          (error == cudaSuccess && count == 0)) {
        std::cout << "[SKIP] No usable CUDA GPU: " << cudaGetErrorString(error)
                  << '\n';
        return 77;
      }
      require(error == cudaSuccess, cudaGetErrorString(error));
      error = cudaSetDevice(0);
      require(error == cudaSuccess, cudaGetErrorString(error));
    }
    if (dtype == "float")
      run<float>(device);
    else
      run<double>(device);
    std::cout << "[PASS] " << backend << ' ' << dtype
              << ": values, shapes, gradients, input validation\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
