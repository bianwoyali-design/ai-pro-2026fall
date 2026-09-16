#include "hw2/activation.h"
#include "hw2/device.h"
#include "hw2/tensor.h"

#include <iostream>

using hw2::Device;
using hw2::Tensor;

int main() {
  std::vector<float> x = {-3, -2, 0.0034, 1.234, 8, 10};
  std::vector<float> g = {2, -3, 0.5, 1.25, -4.32, 9};
  Tensor tensor({2, 3}, Device::CPU);
  Tensor grad({2, 3}, Device::CPU);
  for (int i = 0; i < 6; ++i) {
    tensor.data()[i] = x[i];
    grad.data()[i] = g[i];
  }

  auto ht = tensor.cpu();
  auto dt = tensor.gpu();
  auto hg = grad.cpu();
  auto dg = grad.gpu();

  auto hr = hw2::relu(ht);
  auto dr = hw2::relu(dt);
  auto hrb = hw2::relu_backward(ht, hg);
  auto drb = hw2::relu_backward(dt, dg);
  auto hs = hw2::sigmoid(ht);
  auto ds = hw2::sigmoid(dt);
  auto hsb = hw2::sigmoid_backward(hs, hg);
  auto dsb = hw2::sigmoid_backward(ds, dg);

  for (const auto *output : {&hr, &dr, &hrb, &drb, &hs, &ds, &hsb, &dsb}) {
    auto host = output->cpu();
    for (int i = 0; i < 6; ++i) {
      std::cout << host.data()[i] << " \n"[i == 5];
    }
  }
}