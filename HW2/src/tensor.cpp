#include "hw2/tensor.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace hw2 {

template <typename T>
Tensor<T>::Tensor(std::vector<uint64_t> shape, Device device)
    : shape_(std::move(shape)) {
  stride_.resize(shape_.size());

  const uint64_t limit =
      std::min(std::numeric_limits<uint64_t>::max(),
               static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  uint64_t count = 1;
  for (size_t i = shape_.size(); i > 0; --i) {
    const size_t axis = i - 1;
    stride_[axis] = count;
    if (shape_[axis] && count > limit / shape_[axis]) {
      throw std::overflow_error("Tensor shape overflow");
    }
    count *= shape_[axis];
  }

  storage_ = std::make_shared<Storage<T>>(static_cast<size_t>(count), device);
}

template <typename T> T *Tensor<T>::data() noexcept {
  T *base = storage_->data();
  return base == nullptr ? nullptr : base + offset_;
}

template <typename T> const T *Tensor<T>::data() const noexcept {
  const T *base = storage_->data();
  return base == nullptr ? nullptr : base + offset_;
}

template <typename T> size_t Tensor<T>::numel() const noexcept {
  // Change when Tensor supports slice.
  return storage_->capacity();
}

template <typename T> Device Tensor<T>::device() const noexcept {
  return storage_->device();
}

template <typename T>
const std::vector<uint64_t> Tensor<T>::shape() const noexcept {
  return shape_;
}

template <typename T> Tensor<T> Tensor<T>::cpu() const {
  if (device() == Device::CPU) {
    return *this;
  }

  Tensor<T> hT(shape_, Device::CPU);

  const size_t count = numel();
  if (count) {
    CUDA_CHECK(cudaMemcpy(hT.data(), data(), count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  return hT;
}

template <typename T> Tensor<T> Tensor<T>::gpu() const {
  if (device() == Device::GPU) {
    return *this;
  }

  Tensor<T> dT(shape_, Device::GPU);

  const size_t count = numel();
  if (count) {
    CUDA_CHECK(cudaMemcpy(dT.data(), data(), count * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  return dT;
}

template <typename T> Tensor<T> Tensor<T>::clone() const {
  Tensor<T> cpy(shape_, device());

  const size_t count = numel();
  if (!count) {
    return cpy;
  }

  if (device() == Device::CPU) {
    std::copy_n(data(), count, cpy.data());
  } else {
    CUDA_CHECK(cudaMemcpy(cpy.data(), data(), count * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }

  return cpy;
}

template class Tensor<float>;
template class Tensor<double>;

} // namespace hw2