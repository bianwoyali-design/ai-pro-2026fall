#include "hw2/storage.h"
#include "cuda_utils.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hw2 {

template <typename T>
Storage<T>::Storage(size_t capacity, Device device)
    : capacity_(capacity), device_(device) {
  if (device_ != Device::CPU && device_ != Device::GPU) {
    throw std::invalid_argument("Invalid storage device");
  }

  if (capacity_ > std::numeric_limits<size_t>::max() / sizeof(T)) {
    throw std::overflow_error("Storage size overflow");
  }

  if (!capacity_) {
    return;
  }

  if (device_ == Device::CPU) {
    data_ = new T[capacity_];
  } else {
    CUDA_CHECK(cudaMalloc(&data_, capacity_ * sizeof(T)));
  }
}

template <typename T> Storage<T>::~Storage() noexcept {
  if (data_ == nullptr) {
    return;
  }

  if (device_ == Device::CPU) {
    delete[] data_;
  } else {
    CUDA_CHECK_NOEXCEPT(cudaFree(data_));
  }
}

template <typename T> T *Storage<T>::data() noexcept { return data_; }

template <typename T> const T *Storage<T>::data() const noexcept {
  return data_;
}

template <typename T> size_t Storage<T>::capacity() const noexcept {
  return capacity_;
}

template <typename T> Device Storage<T>::device() const noexcept {
  return device_;
}

template class Storage<float>;
template class Storage<double>;

} // namespace hw2