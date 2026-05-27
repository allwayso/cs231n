---
title: RNN Concrete Example
publish: true
---

### 任务

用RNN完成目标序列中连续出现的1的检测

### 权重矩阵

为了方便起见，激活函数使用ReLU

$W_{hh} = \begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$，$h_t = \begin{pmatrix} x_t \\ x_{t-1} \\ 1 \end{pmatrix}$

### 运算流程

 1. 隐藏状态更新式 $h_t = \begin{pmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix} h_{t-1} + \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} x_t = \begin{pmatrix} x_t \\ x_{t-1} \\ 1 \end{pmatrix}$

 2. 输出生成式 $y_t = \text{ReLU}\left( \begin{pmatrix} 1 & 1 & -1 \end{pmatrix} h_t \right) = \text{ReLU}(x_t + x_{t-1} - 1)$

### 实现

```python
import numpy as np

# w_xh: 将当前输入 x 映射到 h_t 的第一个元素 (Current)
w_xh = np.array([[1], [0], [0]])

# w_hh: 转移矩阵。第二行 [1, 0, 0] 负责把上一步的 Current 复制为这一步的 Previous
w_hh = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 1]])

# w_yh: 输出权重。计算 Current + Previous - 1，用于检测是否两者都为 1
w_yh = np.array([1, 1, -1])

# 初始化隐藏状态 h_0 = [[Current], [Previous], [Bias]]
h_t_prev = np.array([[0], [0], [1]])

# 测试输入序列
x_seq = [0, 1, 0, 1, 1, 1, 0, 1, 1]

def relu(x):
    return np.maximum(0, x)

# 前向传播循环
for t, x in enumerate(x_seq):
    # 1. 更新当前隐藏状态：通过 w_hh 完成记忆右移，并加上当前输入
    h_t = relu(w_hh @ h_t_prev + (w_xh @ np.array([[x]])))
    
    # 2. 计算当前输出：满足 (1 + 1 - 1) = 1 触发输出，其余 <=0 的情况被 ReLU 截断为 0
    y_t = relu(w_yh @ h_t)
    
    print(f"输入: {x} -> 输出: {int(y_t[0])}")
    
    # 3. 记忆传递，进入下一轮循环
    h_t_prev = h_t
```

> 这个示例展示了RNN架构中隐藏状态$h_t$，输入值$x_t$和输出值$y_t$之间的关系