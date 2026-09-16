#pragma once

#include "hw2/device.h"
#include <cstddef>
#include <type_traits>

namespace hw2 {

template <typename T> class Storage {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Storage currently supports float and double only");

public:
  Storage(size_t capacity, Device device);
  ~Storage() noexcept;

  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;

  Storage(Storage &&) = delete;
  Storage &operator=(Storage &&) = delete;

  T *data() noexcept;
  const T *data() const noexcept;

  size_t capacity() const noexcept;
  Device device() const noexcept;

private:
  T *data_ = nullptr;
  size_t capacity_;
  Device device_;
};

} // namespace hw2