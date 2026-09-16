# HW2：Tensor 与激活函数

## 项目结构

```text
HW2/
├── CMakeLists.txt                  # 构建库、示例和 CTest 测试
├── HW2.md                         # 作业要求
├── REPORT.md                      # 实现与实验结果
├── README.md                      # 环境与复现说明
├── include/hw2/                   # 对外接口
│   ├── device.h                   # CPU/GPU 设备枚举
│   ├── storage.h                  # 泛型内存管理
│   ├── tensor.h                   # 形状、布局、设备迁移
│   └── activation.h               # 激活函数正向与反向接口
├── src/
│   ├── storage.cpp                # 内存申请与释放
│   ├── tensor.cpp                 # Tensor 实现
│   ├── activation.cpp             # 参数检查与后端分派
│   ├── activation_backend.h       # 内部后端接口
│   ├── activation_cpu.cpp         # CPU 逐元素计算
│   ├── activation_gpu.cu          # CUDA kernel 与启动函数
│   └── cuda_utils.h               # CUDA 错误检查
├── examples/demo.cpp              # CPU/GPU 激活函数示例
└── tests/
    ├── test_activation.cpp        # 数值、梯度与边界测试
    ├── activation_reference_driver.cpp # PyTorch 测试调用的 C++ 程序
    └── test_pytorch.py            # PyTorch 正向与 autograd 对照
```

## 环境

- CMake 4.4 或更新版本（已验证 4.4.3）。
- 支持 C++23 的 C++ 编译器；本项目使用 Clang 19。
- 支持 CUDA C++23 的 CUDA Toolkit / nvcc（13.3 或更新版本；已验证 13.4）。
- 与 Toolkit 和显卡兼容的 NVIDIA 驱动。
- 默认配置时需要能访问 NVIDIA GPU，以检测本机架构。

## 构建和测试

在仓库根目录运行：

```bash
cmake -S HW2 -B HW2/build -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build HW2/build -j
ctest --test-dir HW2/build --output-on-failure
```

如果 Clang 的程序名带版本号，将 `clang++` 换成例如 `clang++-19`。
如果 nvcc 不在 PATH 中，配置时额外传入
`-DCMAKE_CUDA_COMPILER=/实际路径/cuda/bin/nvcc`。

默认 `CMAKE_CUDA_ARCHITECTURES=native`，针对配置机器检测到的 GPU 构建。
这是在各自硬件上重新编译，不保证不同机器得到相同的二进制文件。

无 GPU 的构建机器或交叉编译场景，应显式指定部署目标，例如：

```bash
cmake -S HW2 -B HW2/build -DCMAKE_CUDA_ARCHITECTURES=120-real
```

`120-real` 适用于计算能力 12.0 的目标；其他显卡需选择对应架构。
显式配置参数及 `CUDAARCHS` 环境变量优先于项目默认值。
已有构建缓存中的架构会保留；要改用本机检测，重新配置时传入
`-DCMAKE_CUDA_ARCHITECTURES=native`。

GPU 不可用或驱动不足时，GPU 测试显示 Skipped，不能视为 GPU 验证通过。
单独运行 GPU 测试并显示详细输出：

```bash
ctest --test-dir HW2/build -L GPU -V
```

## PyTorch 对照测试

测试通过独立 C++ 可执行程序运行本项目的四个激活运算，Python 使用
PyTorch 正向函数和 `torch.autograd.grad` 生成参考值；不是用 Python 重写本项目实现。
覆盖 float/double、CPU/GPU、标量、空张量、零点、极端值和固定种子的随机输入。
GPU 对照要求本项目和 PyTorch 均能使用 CUDA。

选择已安装 PyTorch 的 Python 环境：

```bash
cmake -S HW2 -B HW2/build -DPython3_EXECUTABLE=/实际路径/环境/bin/python
cmake --build HW2/build -j
ctest --test-dir HW2/build -L PyTorch -V
```

报告会显示 PyTorch 版本、比较容差和各运算最大绝对误差。
缺少 PyTorch 或 GPU 不可用时对应测试标记为 Skipped；数值错误或 kernel
加载失败则标记为 Failed。可用 `-DHW2_TEST_PYTORCH=OFF` 关闭这些可选测试。

## Demo

可以修改 `examples/demo.cpp` 以测试 Tensor 类和激活函数的可用性。

编译并运行：

```bash
cmake --build HW2/build --target hw2_demo -j
./HW2/build/hw2_demo
```

`demo.cpp` 创建 Tensor 并展示各项激活函数的 CPU/GPU 正反向结果，运行需要可用的 CUDA GPU。
