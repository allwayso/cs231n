---
title: "PyTorch : 3 abstraction levels"
publish: true
---
> 基于 CS231n Assignment 2 — PyTorch.ipynb  
> 以三层卷积网络为例，展示三种构建神经网络的方式：**Barebones → Module → Sequential**，抽象级别从低到高

---

## Level 1：Barebones PyTorch

> 直接操作 Tensor，手写前向传播函数，手动管理参数和梯度更新。没有 `nn.Module`，没有 `optim`，一切靠自己。

### 涉及的 PyTorch 函数

| 函数                                                                                                    | 简述                             |
| ----------------------------------------------------------------------------------------------------- | ------------------------------ |
| [`torch.zeros`](https://pytorch.org/docs/stable/generated/torch.zeros.html)                           | 创建全零 Tensor                    |
| [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html)                           | 创建服从标准正态分布 𝒩(0,1) 的随机 Tensor  |
| [`Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)                     | 重塑 Tensor 形状（类似 numpy reshape） |
| [`Tensor.mm`](https://pytorch.org/docs/stable/generated/torch.Tensor.mm.html)                         | 矩阵乘法 `(N, D) @ (D, H)`         |
| [`Tensor.requires_grad`](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html)   | 标记该 Tensor 需要计算梯度              |
| [`F.relu`](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)                   | ReLU 激活函数                      |
| [`F.conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)               | 2D 卷积（函数式，需手动传 weight/bias）    |
| [`F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) | 交叉熵损失（内置 softmax）              |
| [`Tensor.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html)             | 反向传播，计算 `.grad`                |
| [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html)                       | 上下文管理器，禁用梯度计算（推理/参数更新时用）       |
| [`torch.flatten`](https://pytorch.org/docs/stable/generated/torch.flatten.html)                       | 展平 Tensor 指定维度                 |

### 核心模式

**1. 辅助函数：flatten**

```python
def flatten(x):
    N = x.shape[0]            # 读出 batch size
    return x.view(N, -1)      # 把 C×H×W 压成单个向量
```

**2. 定义模型 = 一个纯函数**

```python
def two_layer_fc(x, params):
    w1, w2 = params
    x = flatten(x)           # (N, 3, 32, 32) → (N, 3072)
    x = F.relu(x.mm(w1))     # (N, 3072) @ (3072, H) → (N, H)
    x = x.mm(w2)             # (N, H) @ (H, 10) → (N, 10)
    return x
```

**3. 参数初始化 = 手动创建带梯度的 Tensor**

```python
def random_weight(shape):
    # fan_in: 全连接层=shape[0], 卷积层=in_ch × kH × kW
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    w = torch.randn(shape) * np.sqrt(2. / fan_in)   # Kaiming 初始化
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, requires_grad=True)
```

**4. 训练循环 = 手写 SGD**

```python
for x, y in loader_train:
    scores = model_fn(x, params)
    loss = F.cross_entropy(scores, y)

    loss.backward()               # 自动求所有 param.grad

    with torch.no_grad():
        for w in params:
            w -= learning_rate * w.grad
            w.grad.zero_()        # 梯度清零！否则会累加
```

**5. 三层卷积网络完整示例**

```python
# --- 参数初始化 ---
channel_1, channel_2 = 32, 16
learning_rate = 3e-3

conv_w1 = random_weight((channel_1, 3, 5, 5))       # [out=32, in=3, kH=5, kW=5]
conv_b1 = zero_weight((channel_1,))                 # [32]
conv_w2 = random_weight((channel_2, channel_1, 3, 3))# [out=16, in=32, kH=3, kW=3]
conv_b2 = zero_weight((channel_2,))                 # [16]
flat_feat_dim = channel_2 * 32 * 32                 # 16 × 32 × 32 = 16384
fc_w = random_weight((flat_feat_dim, 10))           # [16384, 10]  ← 注意 (in, out)!
fc_b = zero_weight((10,))

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]

# --- 前向传播 ---
def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    out = F.conv2d(x, conv_w1, conv_b1, padding=2)   # (N, 3, 32, 32) → (N, 32, 32, 32)
    out = F.relu(out)
    out = F.conv2d(out, conv_w2, conv_b2, padding=1)  # → (N, 16, 32, 32)
    out = F.relu(out)
    out = torch.flatten(out, start_dim=1)             # → (N, 16384)
    scores = out.mm(fc_w) + fc_b                      # → (N, 10)
    return scores

# --- 训练 ---
train_part2(three_layer_convnet, params, learning_rate)
```

### 缺点

- **参数管理繁琐**：需要手动创建每个 Tensor，存入 list，在训练循环中逐个遍历更新。网络越深越容易出错（拼错变量名、list 顺序不对等）。
- **梯度清零容易忘**：`w.grad.zero_()` 必须手动调用，遗忘会导致梯度累加，训练发散。
- **没有复用性**：模型是一个函数，无法像 `nn.Module` 那样 `model.to(device)`、`model.eval()`、`model.parameters()` 等。
- **与 PyTorch 生态割裂**：不能直接使用 `torch.optim`、`lr_scheduler`、`save/load state_dict` 等高级功能，因为 optimizer 期望 `model.parameters()`。
- **初始化代码冗长**：每个 weight/bias 都要单独调用 `random_weight` 或 `zero_weight`，容易漏写或写错形状。

---

## Level 2 ：PyTorch Module API 

> 继承 `nn.Module`，在 `__init__` 中定义层，在 `forward` 中定义连接关系。参数自动追踪，配合 `torch.optim` 使用。

### 涉及的 PyTorch 函数

| 函数 | 简述 |
|------|------|
| [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) | 所有神经网络模块的基类 |
| [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 2D 卷积层（内置 weight 和 bias） |
| [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) | 全连接层（内置 weight 和 bias） |
| [`nn.init.kaiming_normal_`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) | Kaiming 正态初始化（in-place） |
| [`Module.parameters()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) | 返回模块中所有可学习参数的迭代器 |
| [`Module.to()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) | 将模型参数移动到指定设备 |
| [`Module.train()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) | 设置模型为训练模式 |
| [`Module.eval()`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) | 设置模型为评估模式 |
| [`optim.SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) | SGD 优化器 |
| [`optimizer.zero_grad()`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) | 清零所有参数的梯度 |
| [`optimizer.step()`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) | 更新参数 |

### 核心模式

**1. 定义模型 = 继承 `nn.Module`**

```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        # 层定义（只需声明，参数自动追踪）
        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        # 连接关系（只定义计算流，不创建新层）
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x)
        scores = self.fc(x)
        return scores
```

**2. 实例化模型 + 优化器**

```python
model = ThreeLayerConvNet(in_channel=3, channel_1=32, channel_2=16, num_classes=10)
optimizer = optim.SGD(model.parameters(), lr=3e-3)
```

**3. 训练循环（对比 Part II）**

```python
# Part II 手动版：
#   loss.backward()
#   with torch.no_grad():
#       for w in params:
#           w -= lr * w.grad
#           w.grad.zero_()

# Part III optim 版：
optimizer.zero_grad()   # 替代手动的 for w in params: w.grad.zero_()
loss.backward()
optimizer.step()        # 替代手动的 for w in params: w -= lr * w.grad
```

完整训练循环：

```python
for x, y in loader_train:
    x, y = x.to(device), y.to(device)
    scores = model(x)
    loss = F.cross_entropy(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 缺点

- **仍有一定样板代码**：每个简单的前馈网络都要写 `class`、`__init__`、`forward` 三部曲，对于直筒型网络略嫌啰嗦。
- **`forward` 灵活性带来风险**：可以在 `forward` 中写任意逻辑（如创建新的 `nn.Linear`），但这会导致参数不在 `model.parameters()` 中追踪，optimizer 找不到它们。
- **初始化需要手动调用**：`nn.init.kaiming_normal_` 等需要显式调用，`nn.Sequential` 配合 `apply` 时会更方便。

---

## Level 3 ： PyTorch Sequential API

> 用 `nn.Sequential` 一行串起所有层，代码最简洁。适合直筒型前馈网络（每一层的输出恰好是下一层的输入）。

### 涉及的 PyTorch 函数

| 函数 | 简述 |
|------|------|
| [`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) | 顺序容器，按顺序执行传入的模块 |
| [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) | ReLU 激活层（`nn.Module` 形式） |
| [`optim.SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) | SGD 优化器，支持 momentum / Nesterov |

### 核心模式

**1. 自定义 `Flatten` 模块**

`nn.Sequential` 只能容纳 `nn.Module` 子类，普通函数（如 `flatten`）需要包装成 Module：

```python
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)          # 即 x.view(x.shape[0], -1)
```

**2. 定义模型 = 一行 `nn.Sequential`**

```python
channel_1, channel_2 = 32, 16
learning_rate = 1e-2

model = nn.Sequential(
    nn.Conv2d(3, channel_1, 5, padding=2),       # (N, 3, 32, 32) → (N, 32, 32, 32)
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, 3, padding=1),# (N, 32, 32, 32) → (N, 16, 32, 32)
    nn.ReLU(),
    Flatten(),                                    # (N, 16, 32, 32) → (N, 16384)
    nn.Linear(channel_2 * 32 * 32, 10),           # (N, 16384) → (N, 10)
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, nesterov=True)

train_part34(model, optimizer)
```

**3. 与 Part III 的完整对比**

```python
# Part III（Module）：
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_ch, ch1, ch2, num_cls):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, ch1, 5, padding=2)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.fc = nn.Linear(ch2 * 32 * 32, num_cls)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(flatten(x))

# Part IV（Sequential—等价功能）：
model = nn.Sequential(
    nn.Conv2d(3, 32, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 16, 3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(16 * 32 * 32, 10),
)
```

### 缺点

- **灵活性最低**：只能表达"前一层的输出直连后一层"的拓扑，无法处理分支、跳跃连接（如 ResNet）、多输入/多输出等情况。
- **调试困难**：层之间没有命名（除非用 `OrderedDict`），中间结果无法方便地取出检查。
- **自定义操作需额外包装**：像 `flatten` 这样的非参数操作，需要单独写一个 `nn.Module` 包装类才能放入 `Sequential`。
- **初始化不灵活**：默认使用 PyTorch 的内置初始化，要对不同层使用不同初始化策略需要额外的遍历逻辑（`model.apply(...)`）。

---

## 总结：三种抽象级别对比

| | Part II — Barebones | Part III — Module | Part IV — Sequential |
|---|---|---|---|
| **定义方式** | 纯函数 `def model(x, params)` | `class Net(nn.Module)` | `nn.Sequential(...)` |
| **参数管理** | 手动 list `[w1, w2, ...]` | 自动 `model.parameters()` | 自动 |
| **梯度清零** | `w.grad.zero_()` 手动遍历 | `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| **参数更新** | `w -= lr * w.grad` 手动 | `optimizer.step()` | `optimizer.step()` |
| **设备移动** | 每个 Tensor 单独 `.to(device)` | `model.to(device)` 一键 | `model.to(device)` 一键 |
| **灵活性** | 最高（任何计算图都行） | 高（`forward` 中任意逻辑） | 低（仅限顺序堆叠） |
| **代码量** | 最多 | 中等 | 最少 |
| **适用场景** | 学习原理、调试底层 | 通用场景、研究实验 | 简单直筒网络、快速原型 |
