# AI Programming HW 1: LeNet on CIFAR-10

《人工智能中的编程》第一次作业：使用 PyTorch 实现 LeNet，并在 CIFAR-10 数据集上完成图像分类。

## 1. Environment

本项目开发与测试环境如下：

| Component | Version |
| --- | --- |
| OS | `Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.41` |
| Python | 3.11.15 |
| CUDA Toolkit | 13.4 |
| Pytorch | 2.14.0+cu132 |
| torchvision | 0.29.0+cu132 |
| Package Manager | uv |
| Experiment Tracking | Weights & Biases (W&B) |

可以使用以下命令检查环境：

```bash
nvcc --version
uv --version
python3 --version
```

检查 PyTorch、torchvision 和 CUDA：

```bash
uv run python -c \
'import torch, torchvision; \
print("PyTorch:", torch.__version__); \
print("torchvision:", torchvision.__version__); \
print("PyTorch CUDA:", torch.version.cuda); \
print("CUDA available:", torch.cuda.is_available()); \
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'
```

本项目要求训练和测试均在 CUDA GPU 上运行。如果 CUDA 不可用，程序会直接终止。

## 2. Installation

本项目使用 `uv` 管理 Python 环境和依赖。

克隆项目后，在项目根目录运行：

```bash
uv sync
```

`uv` 会根据 `pyproject.toml` 和 `uv.lock` 创建虚拟环境并安装依赖。

如果需要手动添加普通 Python 依赖，可以使用：

```bash
uv add <package-name>
```

例如安装 W&B：

```bash
uv add wandb
```

对于 PyTorch，本项目使用 CUDA 13.2 对应的官方 wheel，因此应从 PyTorch 官方 cu132 源安装：

```bash
uv add "torch==2.14.0+cu132" "torchvision==0.29.0+cu132" \
  --index pytorch-cu132=https://download.pytorch.org/whl/cu132
```

安装完成后，`uv` 会自动更新：

```text
pyproject.toml
uv.lock
```

之后再次恢复环境时，只需要运行：

```bash
uv sync
```

如果尚未登录 W&B：

```bash
uv run wandb login
```

## 3. Project Structure

```text
HW1/
├── src/
│   └── hw1/
│       ├── __init__.py
│       └── main.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── REPORT.md
└── HW1.md
```

各文件和目录用途：

- `src/hw1/main.py`：LeNet、CIFAR-10 数据加载、训练及测试代码；
- `HW1.md`：实验要求；
- `REPORT.md`：实验报告；
- `.python-version`：指定项目使用 Python 3.11.15；
- `pyproject.toml`：项目及直接依赖配置；
- `uv.lock`：记录完整且可复现的依赖版本。

## 4. Dataset

实验使用 CIFAR-10 数据集，通过

```python
torchvision.datasets.CIFAR10
```

自动下载。

CIFAR-10 包含 10 个类别：

```text
plane
car
bird
cat
deer
dog
frog
horse
ship
truck
```

数据集规模：

| Dataset | Number of Images |
| --- | ---: |
| Training Set | 50,000 |
| Test Set | 10,000 |

每张图像为 RGB 图像，形状为：

```text
3 × 32 × 32
```

数据预处理使用 torchvision transforms v2：

```python
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)
```

训练集使用：

```python
shuffle=True
```

测试集使用：

```python
shuffle=False
```

Batch Size 设置为：

```text
64
```

## 5. LeNet

使用 `torch.nn.Module` 实现适用于 CIFAR-10 的 LeNet：

| Layer | Parameters | Output Shape |
| --- | --- | --- |
| Input | RGB Image | `3 × 32 × 32` |
| Conv1 | `3 → 6`, kernel `5 × 5` | `6 × 28 × 28` |
| ReLU + MaxPool | kernel `2 × 2`, stride `2` | `6 × 14 × 14` |
| Conv2 | `6 → 16`, kernel `5 × 5` | `16 × 10 × 10` |
| ReLU + MaxPool | kernel `2 × 2`, stride `2` | `16 × 5 × 5` |
| Flatten | - | `400` |
| Linear1 + ReLU | `400 → 120` | `120` |
| Linear2 + ReLU | `120 → 84` | `84` |
| Linear3 | `84 → 10` | `10` |

损失函数：

```python
nn.CrossEntropyLoss()
```

优化器：

```python
optim.SGD(...)
```

## 6. Training Configuration

主要训练参数：

| Hyperparameter | Value |
| --- | ---: |
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | SGD |
| Loss Function | CrossEntropyLoss |
| Device | CUDA |

为了研究 Momentum 参数对 SGD 训练过程的影响，程序依次运行以下三组实验：

```text
momentum = 0.50
momentum = 0.90
momentum = 0.98
```

三组实验除 Momentum 外采用相同的模型结构、数据集、Batch Size、Learning Rate 和 Epoch 数。

训练和测试均在 CUDA GPU 上完成。

在第一次 forward 之前，程序会输出模型参数、输入图像和标签所在的 Device：

```text
Model device: cuda:0
Images device: cuda:0
Labels device: cuda:0
```

## 7. Run

在项目根目录运行：

```bash
uv run python -m hw1.main
```

程序将依次完成：

```text
Momentum = 0.50
    ├── Train for 10 epochs
    └── Test on CIFAR-10

Momentum = 0.90
    ├── Train for 10 epochs
    └── Test on CIFAR-10

Momentum = 0.98
    ├── Train for 10 epochs
    └── Test on CIFAR-10
```

每组实验使用独立的 W&B Run。

## 8. W&B

实验使用 Weights & Biases 记录训练过程。

W&B Project：

```text
aipro-hw1-cifar10
```

每个 Batch 记录：

```text
loss
```

每个 Epoch 记录：

```text
epoch_loss
```

同时分别使用 `batch` 和 `epoch` 作为两个指标的横坐标，从而可以直观比较不同 Momentum 下的 Loss Curve。

正式实验包含以下三个 Run：

```text
momentum-0.5
momentum-0.9
momentum-0.98
```

W&B Project / Run 链接：

https://wandb.ai/bianwoyali-peking-university/aipro-hw1-cifar10

## 9. Output

### Training Output

每个 Epoch 结束后，程序会在终端输出平均训练 Loss：

```text
Epoch [1/10] loss: ...
Epoch [2/10] loss: ...
...
Epoch [10/10] loss: ...
```

完整的 Batch Loss 和 Epoch Loss 曲线保存在对应的 W&B Run 中。

### Test Output

每组实验训练完成后，程序在 CIFAR-10 测试集上计算：

1. 整体测试准确率；
2. 10 个类别各自的测试准确率。

终端输出格式如下：

```text
Accuracy of the network on the 10000 test images: XX.XX %

Accuracy for class: plane is XX.X %
Accuracy for class: car   is XX.X %
Accuracy for class: bird  is XX.X %
Accuracy for class: cat   is XX.X %
Accuracy for class: deer  is XX.X %
Accuracy for class: dog   is XX.X %
Accuracy for class: frog  is XX.X %
Accuracy for class: horse is XX.X %
Accuracy for class: ship  is XX.X %
Accuracy for class: truck is XX.X %
```

实验结果、W&B Loss Curve 以及不同 Momentum 的对比分析见：

```text
HW1.md
```

## 10. Reproducibility

项目使用 `uv.lock` 锁定 Python 依赖。

在兼容的平台上执行：

```bash
uv sync
```

即可根据 lock file 恢复本实验使用的 Python 依赖环境。

实际安装的软件包版本可以通过以下命令查看：

```bash
uv tree
```