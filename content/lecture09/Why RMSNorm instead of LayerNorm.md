---
title: "Why RMSNorm Instead of LayerNorm"
publish: true
target: "分析 RMSNorm 去掉均值中心化后计算更快、性能相当甚至更优的原因，以及在现代 LLM 中的广泛采用"
---

# RMSNorm vs LayerNorm

## 基本形式

**LayerNorm**（Ba et al., 2016）对每个样本的特征维度进行归一化：

$$
y = \frac{x - \mu}{\sigma} \odot \gamma + \beta
$$

其中 $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$（均值），$\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2}$（标准差），$\gamma, \beta \in \mathbb{R}^d$ 是可学习的仿射参数。

> *"Layer normalization is a method to reduce the training time of various neural network models. Unlike batch normalization, layer normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer."*
> 层归一化是一种减少各种神经网络模型训练时间的方法。与批量归一化不同，层归一化直接从隐藏层内神经元的汇总输入中估计归一化统计数据。
> — Ba et al., Layer Normalization, 2016

**RMSNorm**（Zhang & Sennrich, 2019）去掉了均值中心化，仅保留缩放：

$$
y = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

注意 RMSNorm 只有 $\gamma$ 参数，没有 $\beta$（偏置），因为去掉均值后不再需要偏移补偿。

> *"RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability."*
> RMSNorm 根据均方根（RMS）对一层中神经元的汇总输入进行正则化，赋予模型重缩放不变性属性和隐式学习率适应能力。
> — Zhang & Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 公式 | $y = \frac{x - \mu}{\sigma} \odot \gamma + \beta$ | $y = \frac{x}{\text{RMS}(x)} \odot \gamma$ |
| 均值中心化（re-center） | ✔ 减均值 | ✗ 无 |
| 缩放（re-scale） | ✔ 除标准差 | ✔ 除 RMS |
| 可学习参数 | $\gamma, \beta$（2d 个） | $\gamma$（d 个） |
| 提出时间 | 2016 | 2019 |
| 代表模型 | 原始 Transformer, BERT, GPT-2 | LLaMA, LLaMA 2/3, Mistral, Qwen, 多数 2023+ 开源 LLM |

直观理解：LayerNorm 同时做"平移 + 缩放"，RMSNorm 只做"缩放"。表面看只是省了一步均值计算，但这背后的理论与工程意义值得深入探讨。

下面我们将从三个角度深入分析 RMSNorm 替代 LayerNorm 的原因：**计算效率**（训练与推理的加速）、**为什么去掉均值中心化**（理论依据）、**梯度与训练影响**（收敛与性能对比）。

---

## 计算效率

### 计算步骤对比

LayerNorm 一次前向传播需要计算的步骤（假设特征维度为 $d$）：

1. **求均值**：$d$ 次加法 + $1$ 次除法
2. **求方差**：$d$ 次减法 + $d$ 次平方 + $d$ 次加法 + $1$ 次除法
3. **归一化**：$d$ 次减法 + $d$ 次除法
4. **仿射变换**：$d$ 次乘法 + $d$ 次加法

RMSNorm 一次前向传播：

1. **求 RMS**：$d$ 次平方 + $d$ 次加法 + $1$ 次除法 + $1$ 次开方
2. **归一化**：$d$ 次除法
3. **缩放**：$d$ 次乘法（无加法，无 $\beta$）

| 操作 | LayerNorm | RMSNorm | 节省 |
|------|-----------|---------|------|
| 均值计算 | $O(d)$ | — | 省去 |
| 方差 / RMS 计算 | $O(d)$ | $O(d)$ | 相近 |
| 归一化（减法） | $O(d)$ | — | 省去 |
| 仿射参数 | $2d$ | $d$ | 省去 $\beta$ |
| 总体时间 | 基准 | 快约 10-20% (论文报告) | 减少约 1-2 次逐元素运算 |

> *"RMSNorm is computationally simpler and thus more efficient than LayerNorm. Experiments on several tasks show that RMSNorm achieves comparable performance to LayerNorm but with less running time."*
> RMSNorm 计算上更简单，因此比 LayerNorm 更高效。在多个任务上的实验表明，RMSNorm 取得了与 LayerNorm 相当的性能，但运行时间更短。
> — Zhang & Sennrich, 2019

### 在 LLM 推理中的实际意义

对于大规模语言模型（如 LLaMA-70B），推理时每一层都需要执行归一化操作。RMSNorm 省去的均值和偏置计算在大批量推理中累积为可观的时间节省。此外，$\beta$ 参数的消除减少了内存占用：每层节省 $d$ 个 float16，对于 8192 维 × 80 层的模型约节省 1.3 MB——虽然相对于总参数不大，但在显存紧张的场景下仍有价值。

计算效率的提升直接来源于去掉均值中心化，那么为什么去掉它不会损害模型性能？下一节将从理论上回答这个问题。

---

## 为什么去掉均值中心化？

### LayerNorm 的成功归因

Ba et al. (2016) 认为 LayerNorm 的有效性来自两个操作：

> *"We compute the layer normalization statistics over all the hidden units in the same layer... The normalization terms make it invariant to the scaling and shifting of the summed inputs."*
> 我们在同一层中的所有隐藏单元上计算层归一化统计数据……归一化项使得它对汇总输入的缩放和平移具有不变性。
> — Ba et al., 2016

传统观点：re-center（减均值）和 re-scale（除标准差）都对训练稳定性和收敛速度有贡献。

### RMSNorm 的核心洞察：re-centering 并不必要

Zhang & Sennrich (2019) 的核心论点是：**re-scaling 已经足够**，re-centering 对 LayerNorm 的成功并非必要。

> *"We hypothesize that the re-scaling invariance is the reason for the success of LayerNorm, rather than the re-centering invariance. We propose RMSNorm which only retains the re-scaling invariance."*
> 我们假设重缩放不变性是 LayerNorm 成功的原因，而非重新定中心不变性。我们提出 RMSNorm，仅保留重缩放不变性。
> — Zhang & Sennrich, 2019

### 不变性分析

LayerNorm 提供两种不变性：

- **re-center 不变性**：对输入 $x$ 加上任意常数 $c$ 不改变输出（$\text{LN}(x + c) = \text{LN}(x)$）
- **re-scale 不变性**：对输入 $x$ 乘以任意常数 $c$ 不改变输出（$\text{LN}(c x) = \text{LN}(x)$）

RMSNorm 仅提供一种：

- **re-scale 不变性**：$\text{RMSNorm}(c x) = \text{RMSNorm}(x)$

关键问题：丢失 re-center 不变性是否会影响性能？

| 不变性 | LayerNorm | RMSNorm |
|--------|-----------|---------|
| re-center（平移）不变性 | ✔ | ✗ |
| re-scale（缩放）不变性 | ✔ | ✔ |
| 梯度流中的缩放不变性 | ✔ | ✔ |
| 隐式学习率适应 | ✔ | ✔ |

### 实验证据

Zhang & Sennrich 在机器翻译和语言建模任务上的实验表明，RMSNorm 和 LayerNorm 达到了**相当的 BLEU 分数和困惑度**，而 RMSNorm 计算开销更小。

> *"Empirically, RMSNorm achieves comparable performance to LayerNorm across a range of tasks including machine translation and language modeling, while being simpler and faster."*
> 从经验上看，RMSNorm 在包括机器翻译和语言建模在内的一系列任务上取得了与 LayerNorm 相当的性能，同时更简单、更快。
> — Zhang & Sennrich, 2019

既然纯缩放就能达到相同的效果，那么去掉均值中心化对梯度流有什么影响？下一节将从梯度与收敛角度展开。

---

## 梯度与训练影响

### RMSNorm 的梯度更简洁

LayerNorm 的梯度需要同时经过均值减法和标准差除法的反向传播，而 RMSNorm 的梯度只需经过 RMS 除法的反向传播，计算图更短：

$$
\frac{\partial \text{RMSNorm}(x)}{\partial x} = \frac{\gamma}{\text{RMS}(x)}\left(I - \frac{x x^T}{d \cdot \text{RMS}(x)^2}\right)
$$

相比 LayerNorm 少了均值相关的求导链，梯度计算更高效，且数值上更稳定（尤其是在 $x$ 分布偏移较大时，均值估计的方差可能引入额外噪声）。

### 训练收敛对比

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 梯度计算复杂度 | 较高（均值 + 方差两条链） | 较低（仅 RMS 一条链） |
| 梯度数值稳定性 | 一般（均值估计噪声） | 较好 |
| 收敛速度 | 基准 | 相当或略快 |
| 最终性能（机器翻译 BLEU） | 基准 | 相当（差异 $< 0.1$） |
| 最终性能（语言模型 PPL） | 基准 | 相当或略优 |
| 对学习率敏感度 | 一般 | 较低（论文报告更好的鲁棒性） |

### 实验数据摘要

Zhang & Sennrich 论文的核心实验结果：

- **WMT 机器翻译**（Transformer base / Transformer big）：RMSNorm 与 LayerNorm BLEU 基本持平
- **语言建模**（WikiText-2 / PTB）：RMSNorm 取得相似或略低的困惑度
- **收敛曲线**：RMSNorm 的训练 loss 下降曲线与 LayerNorm 高度重叠

这解释了为什么业界在大规模模型训练中乐意采用 RMSNorm——在保证性能的前提下，它以更低的计算开销和更简洁的实现完成了同样的工作。

既然 RMSNorm 在性能和效率上都具有竞争力，业界是如何采纳这一方案的？下一节将梳理实际模型的 Norm 方案演变。

---

## 模型实践一览

| 年份   | 模型                       | Norm 方案                | 备注                        |
| ---- | ------------------------ | ---------------------- | ------------------------- |
| 2016 | ResNet / RNN 类模型         | LayerNorm              | Ba et al. 提出 LayerNorm    |
| 2017 | Transformer (Vaswani)    | Post-Norm + LayerNorm  | 原始 Transformer            |
| 2018 | BERT                     | Post-Norm + LayerNorm  |                           |
| 2019 | RMSNorm 论文               | RMSNorm (首次提出)         | Zhang & Sennrich, NeurIPS |
| 2019 | GPT-2                    | Pre-Norm + LayerNorm   | 仍使用 LayerNorm             |
| 2020 | GPT-3                    | Pre-Norm + LayerNorm   | 96 层，仍用 LayerNorm         |
| 2023 | LLaMA                    | Pre-Norm + **RMSNorm** | 7B-65B，大规模模型中首次广泛采用       |
| 2023 | LLaMA 2                  | Pre-Norm + RMSNorm     | 7B-70B                    |
| 2023 | Mistral                  | Pre-Norm + RMSNorm     | 7B                        |
| 2024 | Qwen / Qwen 2 / Qwen 2.5 | Pre-Norm + RMSNorm     | 开源中文 LLM 标杆               |
| 2024 | LLaMA 3 / 3.1            | Pre-Norm + RMSNorm     | 8B-405B                   |
| 2025 | DeepSeek-V3              | Pre-Norm + RMSNorm     | MoE 架构                    |

从 2023 年 LLaMA 开始，RMSNorm 已成为开源大规模语言模型的**事实标准**。它通常是 "LLaMA-style" 架构（Pre-Norm + RMSNorm + SwiGLU + RoPE）四件套的一部分。

综合以上理论与实践的全面分析，下面给出全文总结。

---

## 总结

### 全维度对比表

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **公式** | $y = \frac{x - \mu}{\sigma} \odot \gamma + \beta$ | $y = \frac{x}{\text{RMS}(x)} \odot \gamma$ |
| **re-center（平移）** | ✔ 减均值 | ✗ 无 |
| **re-scale（缩放）** | ✔ 除标准差 | ✔ 除 RMS |
| **可学习参数** | $\gamma + \beta$（$2d$） | $\gamma$（$d$） |
| **前向计算步骤** | 均值 + 方差 + 归一化 + 仿射 | RMS + 归一化 + 缩放 |
| **计算开销** | 基准 | 减少约 10-20% |
| **梯度复杂度** | 两条求导链（均值 + 方差） | 一条求导链（RMS） |
| **性能（MT / LM）** | 基准 | 相当 |
| **提出时间** | 2016 | 2019 |
| **代表模型** | Transformer, BERT, GPT-2/3 | LLaMA 全系列, Mistral, Qwen, 多数 2023+ LLM |

### 核心理由

RMSNorm 替代 LayerNorm 的原因可以归结为三点：

1. **计算效率更高**：省去了均值计算和 $\beta$ 参数，减少了逐元素运算，在大规模模型中累积为可观的加速
2. **均值中心化（re-centering）并不必要**：论文从理论上论证了 re-scaling 不变性已经足够维持训练稳定性，re-centering 对最终性能贡献有限
3. **梯度更简洁、实现更简单**：求导链更短，在大规模分布式训练中减少了通信和同步开销

> **RMSNorm 是 LayerNorm 的"精简版"：它在保留核心缩放能力的前提下，移除了不必要的均值计算和偏置参数，从而在大规模语言模型时代成为更优的选择。两者的性能差异在统计上不显著，但 RMSNorm 的计算和实现优势使其自 2023 年后成为开源 LLM 架构的事实标准。**

---

## 参考文献

- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)：提出了 LayerNorm 范式（re-center + re-scale），为深度网络训练提供了稳定且批次无关的归一化方法。

- [Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019)](https://arxiv.org/abs/1910.07467)：提出了 RMSNorm，论证 re-centering 对 LayerNorm 的成功并非必要，仅保留 re-scaling 即可保持性能并减少计算开销。

- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)：在大规模语言模型中率先采用 Pre-Norm + RMSNorm + SwiGLU + RoPE 的组合架构，标志着 RMSNorm 进入主流。