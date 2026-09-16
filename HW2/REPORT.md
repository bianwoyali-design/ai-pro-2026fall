# HW2 实验报告

## 1. 实现方法

使用 C++23 与 CUDA，实现支持 `float`、`double` 的 Tensor 和激活函数。

- **内存管理**：`Storage<T>` 使用 `new[]/delete[]` 管理 CPU 内存，使用 `cudaMalloc/cudaFree` 管理 GPU 显存，析构时自动释放。`Tensor<T>` 通过 `shared_ptr` 共享 Storage，以 `vector` 保存 shape、stride，并保留 offset，便于后续扩展视图；本次仅支持连续布局。
- **设备迁移**：`.cpu()`、`.gpu()` 在跨设备时创建新存储并调用 `cudaMemcpy`；同设备时共享原存储。`.clone()` 在当前设备上深拷贝。
- **激活函数**：入口检查反向输入的形状和设备，分派到 CPU 循环或 CUDA grid-stride kernel。GPU 每个 block 使用 256 个线程，block 数取所需数量与 `4 × SM数` 的较小值。

令上游梯度为 $g=\partial L/\partial y$，逐元素计算：

| 运算 | 正向 | 反向 |
| --- | --- | --- |
| ReLU | $y=\max(x,0)$ | $x>0$ 时为 $g$，否则为 $0$ |
| Sigmoid | $y=1/(1+e^{-x})$ | $g\,y(1-y)$ |

ReLU 在零点的梯度设为 0；Sigmoid 反向复用前向输出。上游梯度由调用者提供，不实现自动求导系统。

## 2. 正确性验证

测试覆盖标量、空张量、零点、极端输入、非整块长度和随机数据，并检查输出形状、设备及输入不被修改。反向使用非全 1 的上游梯度；另外对 $L(x)=\sum_i f(x_i)g_i$ 做中心差分检查，ReLU 避开零点。CPU 的 float/double 数值测试均通过。

PyTorch 对照测试由 Python 调用实际 C++ 程序，以 PyTorch 正向函数及 `torch.autograd.grad` 为参考。每组覆盖 7 种形状，随机种子固定。测试环境为 Debian/WSL、Clang 19、CMake 4.4.3、CUDA 13.4、RTX 5070 Ti Laptop（`sm_120`），参考版本为 PyTorch 2.14.0+cu132。

根据本机最终测试日志，CPU/GPU × float/double 四组对照测试全部通过，最大绝对误差如下：

| 设备 / 类型 | ReLU 正向 / 反向 | Sigmoid 正向 | Sigmoid 反向 |
| --- | --- | --- | --- |
| CPU / float | 0 / 0 | 1.19e-7 | 4.77e-7 |
| CPU / double | 0 / 0 | 2.22e-16 | 9.44e-16 |
| GPU / float | 0 / 0 | 0 | 1.19e-7 |
| GPU / double | 0 / 0 | 0 | 2.22e-16 |

比较条件为 $|a-b|\le atol+rtol\,|b|$；float 使用 `(2e-6, 2e-5)`，double 使用 `(1e-12, 1e-10)`。结果均满足容差，验证了测试范围内的正反向计算及 GPU 数据迁移。构建与复现命令见 [README](README.md)。
