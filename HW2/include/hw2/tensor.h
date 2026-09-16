#pragma once

#include "hw2/device.h"
#include "hw2/storage.h"

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace hw2 {

template <typename T = float> class Tensor {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Tensor currently supports float and double only");

public:
  Tensor(std::vector<uint64_t> shape, Device device);
  ~Tensor() noexcept = default;

  Tensor<T> cpu() const;
  Tensor<T> gpu() const;
  Tensor<T> clone() const;

  T *data() noexcept;
  const T *data() const noexcept;

  size_t numel() const noexcept;
  Device device() const noexcept;
  const std::vector<uint64_t> shape() const noexcept;

private:
  std::shared_ptr<Storage<T>> storage_;
  std::vector<uint64_t> shape_;
  std::vector<uint64_t> stride_;
  size_t offset_ = 0;
};

} // namespace hw2