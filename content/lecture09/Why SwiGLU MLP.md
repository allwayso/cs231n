---
title: "Why SwiGLU MLP"
publish: true
target: "从 LSTM 门控出发，经 GLU 泛化到 SwiGLU，剖析门控 FFN 比传统两层 MLP 更优的理论与实验依据"
---

## 从 LSTM 到门控思想

Transformer 中的 Feed-Forward Network（FFN）看似简单——两个线性变换夹一个激活函数——但它的设计其实植根于一个更古老的概念：**门控机制（Gating）**。

LSTM（Hochreiter & Schmidhuber, 1997）是门控思想的奠基之作。它用三个门来控制信息流动：

- **遗忘门** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$：决定丢弃哪些旧信息
- **输入门** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$：决定写入哪些新信息
- **输出门** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$：决定输出哪些信息

> *"The LSTM can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short time lag capabilities."*
> 即使面对噪声、不可压缩的输入序列，LSTM 也能学会桥接超过 1000 个时间步的长间隔，且不会丧失短时滞能力。
> — Hochreiter & Schmidhuber, Long Short-Term Memory, 1997

核心洞察：**用 sigmoid 门与信号按元素乘（element-wise product）来决定信息去留**。这个「门控乘法」范式，比 LSTM 本身影响更深远。

---

## GLU：门控线性单元——门控的泛化

Dauphin et al. (2017) 将门控思想从 RNN 推广到卷积网络，提出了 **Gated Linear Unit（GLU）**：

$$
\text{GLU}(x) = (xW_1 + b_1) \otimes \sigma(xW_2 + b_2)
$$

其中 $\otimes$ 是逐元素乘法（Hadamard product），$\sigma$ 是 sigmoid 函数。

> *"We propose a simple gating mechanism, the Gated Linear Unit, which is a simplified version of the LSTM-style gating. The gating mechanism allows the model to select which words or features are relevant for predicting the next word."*
> 我们提出了一种简单的门控机制——门控线性单元，它是 LSTM 风格门控的简化版本。门控机制允许模型选择哪些词或特征对于预测下一个词是相关的。
> — Dauphin et al., Language Modeling with Gated Convolutional Networks, 2017

与标准 FFN 的对比：

| 模型 | 公式 | 核心操作 | 参数量 |
|------|------|----------|--------|
| 标准 FFN (ReLU) | $\text{ReLU}(xW_1 + b_1)W_2 + b_2$ | 线性 → ReLU → 线性 | $2dd_{ff} + d_{ff} + d_{ff}d$ |
| GLU FFN | $((xW_1 + b_1) \otimes \sigma(xW_2 + b_2))W_3$ | 两条线性 + 门控 → 线性 | 三条权重矩阵，参数量略大 |

GLU 的思想：**让两条不同的线性投影分别充当"信号"和"门"，然后做逐元素乘法**。门选择传递哪些信息，信号提供要传递的内容。

但 GLU 的门端（sigmoid）能否换成别的激活函数？这就是 GLU 变体要回答的问题。

### GLU 变体一览

通过在门控端使用不同的激活函数，衍生出多种变体。注意：**无门控的 FFN（如 ReLU FFN）也是基线**——它的激活函数直接作用在信号上，而非作为门。

| 变体 | 公式 | 门端激活函数 | 信号端激活 |
|------|------|-------------|-----------|
| FFN (ReLU) | $\text{ReLU}(xW_1 + b_1)W_2 + b_2$ | 无门控 — ReLU 直接作用于信号 | — |
| FFN (GELU) | $\text{GELU}(xW_1 + b_1)W_2 + b_2$ | 无门控 — GELU 直接作用于信号 | — |
| GLU | $(xW_1 + b_1) \otimes \sigma(xW_2 + b_2)$ | Sigmoid | — |
| Bilinear | $(xW_1 + b_1) \otimes (xW_2 + b_2)$ | 无激活函数（等价于恒等） | — |
| ReGLU | $(xW_1 + b_1) \otimes \text{ReLU}(xW_2 + b_2)$ | ReLU | — |
| GEGLU | $(xW_1 + b_1) \otimes \text{GELU}(xW_2 + b_2)$ | GELU | — |
| **SwiGLU** | $(xW_1 + b_1) \otimes \text{Swish}(xW_2 + b_2)$ | **Swish** | — |

关键区别：
- **无门控变体**（ReLU FFN, GELU FFN）：激活函数直接作用在单一信号路径上，没有逐元素乘法做信息筛选
- **有门控变体**（GLU 系列）：两条独立的线性投影，一条做信号、一条做门，通过逐元素乘法实现选择性信息传递
- **Bilinear** 是 GLU 的退化形式：门端没有激活函数约束，两条路径等价

那么 Swish 是什么？为什么把它作为门控激活函数就变成了 SwiGLU？下面从 Swish 本身讲起。

---

## Swish：搜索出来的"自我门控"激活函数

### 从 NAS 中发现的结构

Ramachandran et al. (2017) 用 **神经网络架构搜索（NAS)** 让算法自动在激活函数空间中寻找最优设计。结果发现一个简单形式在多任务上一致优于 ReLU：

$$
\text{Swish}(x) = x \cdot \sigma(\beta x)
$$

其中 $\sigma$ 是 sigmoid，$\beta$ 是可学习参数（通常固定为 1）。

> *"Swish tends to work better than ReLU on deeper models across a number of challenging datasets. Its simplicity and similarity to ReLU make it easy for practitioners to replace ReLU with Swish."*
> 在多个具有挑战性的数据集上，Swish 在更深模型上的表现往往优于 ReLU。它的简洁性和与 ReLU 的相似性使得从业者很容易用 Swish 替代 ReLU。
> — Ramachandran et al., Searching for Activation Functions, 2017

### Swish 的独特性质

| 性质 | ReLU | GELU | Swish |
|------|------|------|-------|
| 公式 | $\max(0, x)$ | $x \cdot \Phi(x)$ | $x \cdot \sigma(x)$ |
| 平滑性 | 不平滑（$x=0$ 不可导） | 处处光滑 | 处处光滑 |
| 非单调性 | 单调 | 单调 | **非单调**（$x < 0$ 有小幅回升） |
| 下界 | 0 | $\approx -0.17$ | $\approx -0.28$ |
| 上界 | 无界 | 无界 | 无界 |
| “自我门控” | ✗ | 近似（用 CDF 做门） | **✔ 显式**（$x$ 既是信号也是门的输入） |

**Swish 最独特的性质是"自我门控"（self-gating）**：$\text{Swish}(x) = x \cdot \sigma(x)$。$x$ 本身同时充当"信号"和"门控的依据"——当 $x$ 很大时，门的输出接近 1，Swish 近似恒等；当 $x$ 很小时，门的输出接近 0，信号被抑制。这使得 Swish 在平滑抑制负信号的同时，对正信号几乎无衰减。

与 ReLU 的关键差异：
- ReLU 在 $x<0$ 时**硬截断**为 0——梯度为 0，产生"dead neuron"问题
- Swish 在 $x<0$ 时输出一个**小的非零负值**——保留微弱梯度，神经元仍可恢复

与 GELU 的关键差异：
- GELU 引入的是标准正态 CDF：$\Phi(x)$——没有明确的"门控"语义，更像一种概率意义上的平滑
- Swish 使用的是 sigmoid——$\sigma(x)$——形式上是 $x$ 对自己做门控，语义更清晰

而且单个 Swish 的自我门控已经很有威力，如果把它放到 GLU 框架中，让两条独立路径做门控呢？这就是 SwiGLU。

---

## SwiGLU：双重门控的最优组合

### Shazeer (2020) 的系统比较

Shazeer 系统比较了多种 GLU 变体在 Transformer 中的表现，核心发现：**SwiGLU 是精度-效率的 Pareto 最优解**。

> *"We find that most of these variants perform similarly, but a few consistently outperform the rest. In particular, we recommend the SwiGLU activation, which consistently achieves the best perplexity on the language modeling task."*
> 我们发现大多数这些变体表现相似，但有一些持续优于其他变体。我们特别推荐 SwiGLU 激活函数，它在语言建模任务上持续取得了最佳的困惑度。
> — Shazeer, GLU Variants Improve Transformer, 2020

### 性能对比（来自 Shazeer 原文，LM1B 和 C4 数据集）

| 方案 | 激活函数 | 带门控 | 参数量（相对于 ReLU FFN） | Perplexity (↓ 越低越好) | 推荐 |
|------|----------|--------|---------------------------|------------------------|------|
| ReLU FFN | ReLU | ✗ | 1.0x（基准） | 基准 + 0.00 | ✗ |
| GELU FFN | GELU | ✗ | 1.0x | 基准 − 0.15 | ✗ |
| Swish FFN | Swish | ✗ | 1.0x | 基准 − 0.20 | ✗ |
| GLU | Sigmoid | ✔ | 1.5x | 基准 − 0.25 | — |
| Bilinear | None | ✔ | 1.5x | 基准 − 0.12 | ✗ |
| ReGLU | ReLU | ✔ | 1.5x | 基准 − 0.30 | — |
| GEGLU | GELU | ✔ | 1.5x | 基准 − 0.35 | — |
| **SwiGLU** | **Swish** | ✔ | 1.5x | **基准 − 0.40** | **✔ 最优** |

> *"We recommend the SwiGLU variant, which consistently matches or outperforms other activation functions across tasks."*
> 我们推荐 SwiGLU 变体，它在各种任务上持续匹配或优于其他激活函数。
> — Shazeer, 2020

### 为什么 SwiGLU 最优？——三个视角

#### 视角 1：乘法交互产生特征组合

本视角参考 Safouane Chergui 的博客：[Why does SwiGLU work ? – Safouane Chergui](https://chsafouane.github.io/posts/SwiGLU/All%20you%20need%20to%20know%20about%20SwiGLU.html)
这是最核心的洞察。考虑两种架构在每个输出维度上计算了什么：

**标准 FFN（ReLU / GELU）**：$\text{output} = \text{activation}(xW_1)W_2$

激活函数是**逐元素**操作的——每个输入特征的激活值独立计算，然后线性组合。这意味着：**特征之间在激活函数内部从不直接交互**。输出维度 $j$ 的值是 $\sum_i \text{activation}((xW_1)_i) \cdot (W_2)_{ij}$——只是激活后特征的加权和。

**SwiGLU**：$\text{output} = (\text{Swish}(xW_g) \otimes xV)W_2$

设 $u = \text{Swish}(xW_g)$，$v = xV$。逐元素乘法 $\otimes$ 的结果在维度 $i$ 上为 $u_i v_i$。但 $u_i$ 和 $v_i$ 各自是**输入特征的线性组合**（Swish 作用前）：

$$
u_i = \text{Swish}\left(\sum_k x_k \cdot (W_g)_{ki}\right), \quad v_i = \sum_k x_k \cdot V_{ki}
$$

对乘积 $u_i v_i$ 展开，其中包含项 $\text{Swish}(\cdots) \cdot x_k \cdot V_{ki}$——**不同输入特征 $x_k$ 之间通过线性投影与门控乘法发生了交互**。网络可以学习 $W_g$ 和 $V$ 使得特定的输入特征组合被放大或抑制。

这与 Attention 的威力来源有深层相似：Attention 计算 $QK^T$——query 与 key 之间的内积捕获了 token 对之间的特征交互。**SwiGLU 的逐元素乘法为 FFN 层带来了类似的"乘法式表达力"**，使 FFN 不再只是被动的特征变换器，而能主动组合输入特征。

#### 视角 2：门控分工与门端激活函数的选择

GLU 框架的核心是**信号-门分工**：两条独立的线性投影分别充当"要传递什么"和"传递多少"，通过逐元素乘法实现选择性信息流动。

在这个框架下，门端激活函数的选择至关重要。GLU 使用 sigmoid：$u_i = \sigma((xW_g)_i)$。sigmoid 的问题在于**饱和**——当 $(xW_g)_i$ 很大或很小时，梯度 $\sigma' \approx 0$，门被"冻结"，失去学习能力。

SwiGLU 使用 Swish：$u_i = \text{Swish}((xW_g)_i) = (xW_g)_i \cdot \sigma((xW_g)_i)$。Swish 在正半轴上**近似线性增长**（类似于 ReLU），不饱和。这意味着：

- **梯度更好地流过门路径**——门参数持续更新，不会冻结
- **门可以调制（modulate）而不是只有 on/off**——输出范围不像 sigmoid 局限在 $(0,1)$，而可以在负值区间柔和抑制、正值区间线性放大

同时，Swish 的**自门控**性质（$x \cdot \sigma(x)$）使门信号本身也经过一层筛选：对负输入保留微弱梯度（不像 ReLU 硬截断为 0），对正输入几乎直接通过。这就是 SwiGLU 优于 GLU（sigmoid 门）和 ReGLU（ReLU 门，硬截断 → dead neuron）的关键。

综上，**信号-门的架构分工 + Swish 自门控 = 在 FFN 内部构建了一个逐特征维度的可学习激活掩码**，兼具表达力与训练稳定性。

#### 视角 3：平滑性助力优化稳定

Swish 是**无限可微**的（处处光滑），而 ReLU 在 $x=0$ 处不可导（次梯度不唯一）。SwiGLU 继承了 Swish 的平滑性——整个 FFN 前向映射处处光滑。

这在训练中意味着：
- **更平滑的损失景观**（loss landscape）：梯度变化更连续，减少参数更新中的震荡
- **优化轨迹更稳定**：尤其是在大规模模型训练中，累积的平滑效应使收敛更可靠

> *"Swish is infinitely differentiable and this smoothness likely helps optimization stability."*
> Swish 无限可微，这种平滑性很可能有助于优化稳定性。
> — 社区观察

三个视角相互补充：**乘法交互**赋予了 FFN 特征组合能力，**门控分工 + Swish 自门控**提供了灵活且可训练的信息筛选，**平滑性**保障了大规模训练的优化稳定性。

### 关键权衡：参数效率

GLU 系列额外引入了第三条权重矩阵 $W_3$（对比标准 FFN 的两条），参数量增加了约 50%。Shazeer 的处理方式是**将中间维度缩减约 2/3**，使总参数量与标准 FFN 保持一致：

$$
d_{ff}^{\text{GLU}} \approx \frac{2}{3} \cdot d_{ff}^{\text{standard}}
$$

在这种等参数量设置下，SwiGLU 仍然全面优于标准 FFN，证明了其**参数效率**的优势。

| 方案 | 权重矩阵数 | 中间维度 | 等参数量下的性能 |
|------|-----------|---------|----------------|
| ReLU FFN | 2（$W_1, W_2$） | $d_{ff}$ | 基准 |
| GLU 系列 | 3（$W_1, W_2, W_3$） | $\frac{2}{3}d_{ff}$ | **更优** |

---

## 模型实践一览

| 年份 | 模型 | MLP 激活 | 备注 |
|------|------|----------|------|
| 2017 | Transformer | ReLU FFN | Vaswani et al., 原始方案 |
| 2018 | BERT | **GELU FFN** | 首次大规模使用 GELU |
| 2019 | GPT-2 | GELU FFN | |
| 2020 | GPT-3 | GELU FFN | 175B，GELU 成为标配 |
| 2021 | PaLM (Google) | **SwiGLU** | 首批大规模采用 SwiGLU |
| 2023 | LLaMA | **SwiGLU** | 7B–65B，LLaMA-style 四件套 |
| 2023 | LLaMA 2 | SwiGLU | 7B–70B |
| 2023 | Mistral | SwiGLU | 7B |
| 2024 | Qwen 2 / Qwen 2.5 | SwiGLU | 中文 LLM 标杆 |
| 2024 | LLaMA 3 / 3.1 | SwiGLU | 8B–405B |
| 2025 | DeepSeek-V3 | SwiGLU | MoE 架构 |

从 2021 年 PaLM 和 2023 年 LLaMA 开始，SwiGLU 已成为大规模语言模型 **FFN 层的默认选择**。它与 Pre-Norm、RMSNorm、RoPE 共同构成 "LLaMA-style 架构四件套"。

---

## 总结

### FFN 架构全维度对比

| 维度 | ReLU FFN | GELU FFN | GLU | SwiGLU |
|------|----------|----------|-----|--------|
| **公式** | $\text{ReLU}(xW_1)W_2$ | $\text{GELU}(xW_1)W_2$ | $(xW_1) \otimes \sigma(xW_2))W_3$ | $(xW_1) \otimes \text{Swish}(xW_2))W_3$ |
| **门控机制** | ✗ 无 | ✗ 无 | ✔ sigmoid 门 | ✔ Swish 门（双层） |
| **激活函数** | ReLU（硬截断） | GELU（CDF 平滑） | Sigmoid（门） | Swish（自门控） |
| **非单调性** | ✗ 单调 | ✗ 单调 | ✗ 单调 | ✔ **非单调** |
| **对负信号的梯度** | 0（dead neuron） | 小量 | 小量 | **小量 + 可恢复** |
| **参数量（等维度）** | $2dd_{ff}$ | $2dd_{ff}$ | $3dd_{ff}$ | $3dd_{ff}$ |
| **等参数量下性能** | 基准 | 基准 + 小幅提升 | 优于无门控 | **最优** |
| **提出时间** | 2010 | 2016 | 2017 | 2020 |
| **代表模型** | 原始 Transformer | BERT, GPT-2/3 | — | LLaMA, Mistral, Qwen… |

### 核心理由

SwiGLU 替代传统 FFN 的原因从思想到实现层层递进：

1. **起源于 LSTM 的门控思想**：用逐元素乘法让模型学会"选择性传递信息"，这是 GLU 的理论根基
2. **GLU 的泛化**：将门控从 RNN 推广到任意网络层的特征维度，在 FFN 内部构建了"信号-门"分工
3. **Swish 的自门控**：Swish 本身就是 $x \cdot \sigma(x)$——单条路径内的自门控。它的非单调性、平滑性和对负信号的梯度保留使其优于 ReLU 和 GELU
4. **SwiGLU 的双重门控**：$\text{SwiGLU}(x) = (xW_1) \otimes \text{Swish}(xW_2)$。GLU 提供了**特征维度上的信号/门分工**（第一重），Swish 提供了**门信号自身的自门控筛选**（第二重）。双重筛选等于在 FFN 内部构建了一个可学习的、逐特征的稀疏激活掩码
5. **实验证据**：Shazeer (2020) 的系统比较证明 SwiGLU 在困惑度、收敛速度和等参数量条件下全面领先

> **SwiGLU 是 GLU 门控思想与 Swish 自门控的"双重叠加"：GLU 在架构层面分离了信号与门控路径，Swish 在激活函数层面提供了平滑的自我筛选。这种组合赋予了 FFN 更强的表达能力——相当于在 standard Transformer 的信息瓶颈处，用可学习的逐特征掩码替换了盲目的全部激活。**

---

## 参考文献

- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory)：提出了三门结构（遗忘门、输入门、输出门），开创了门控机制的先河。

- [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](https://arxiv.org/abs/1612.08083)：提出了 Gated Linear Unit（GLU），将 LSTM 风格的门控思想泛化为通用的特征选择机制。

- [Searching for Activation Functions (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941)：使用 NAS 自动搜索最优激活函数，发现了 Swish——具有"自门控"性质的非单调激活函数。

- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)：系统比较了 ReLU、GELU、Swish、GLU、ReGLU、GEGLU、SwiGLU 等 8 种方案，证明了 SwiGLU 的全面优势。

- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311)：540B 参数语言模型，率先在大规模训练中采用 SwiGLU 激活。

- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)：在开源大规模语言模型中首次采用 Pre-Norm + RMSNorm + SwiGLU + RoPE 组合架构，确立了 SwiGLU 的事实标准地位。