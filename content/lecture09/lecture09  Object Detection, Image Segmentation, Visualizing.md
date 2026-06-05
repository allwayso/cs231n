---
title: "lecture09 Object Detection, Image Segmentation, Visualizing"
publish: true
---

>[!SUMMARY] Table of Contents
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Recap of Tweaking Transformers|Recap of Tweaking Transformers]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Pre-Norm Transformer|Pre-Norm Transformer]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#RMSNorm|RMSNorm]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#SwiGLU MLP|SwiGLU MLP]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Mixture of Experts (MoE)|Mixture of Experts (MoE)]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Tweaking Transformers 小结|Tweaking Transformers 小结]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Materials|Materials]]

## Recap of Tweaking Transformers

由于时长和篇幅限制，[[lecture08 Attention and Transformers#The Transformer|the Transformer part of lecture08]] 只是匆匆介绍了 [[lecture08 Attention and Transformers#Transformers for Language Modeling|LLM]] 和 [[lecture08 Attention and Transformers#Vision Transformers|ViT]] ，没有对近年来 Transformer 架构中越来越常见的改动展开介绍 ，在此做补充

自 2017 年 Attention Is All You Need 提出以来，Transformer 的核心架构变化不大。但近年来有几种改动变得越来越普遍：

1. **Pre-Norm**：将 Layer Normalization 放在残差连接内部，训练更稳定
2. **RMSNorm**：用 Root Mean Square Normalization 替代 LayerNorm
3. **SwiGLU MLP**：用带门控的 Swish 激活替代传统 ReLU/GeLU 两层 MLP
4. **Mixture of Experts (MoE)**：用多个专家 MLP 替代单一 MLP，大幅增加参数量但仅适度增加计算量

### Pre-Norm Transformer

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260605114043.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：Post-Norm：归一化层在残差块外</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260605114244.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：Pre-Norm：归一化层在残差块内</div>
    </div>
</div>

Post-Norm 表达式：$X_{l+1} = \text{LayerNorm}\big(X_l + \text{Sublayer}(X_l)\big)$

Pre-Norm 表达式：$X_{l+1} = X_l + \text{Sublayer}\big(\text{LayerNorm}(X_l)\big)$

Post-Norm 先做残差加和再做归一化，Pre-Norm 先归一化再做残差加和。表面看只是顺序调整，但这微小的差异在深层网络中会被急剧放大。

Pre-Norm 的优势在于同时保留了恒等映射与恒等梯度路径，从而显著提升了深层 Transformer 的训练稳定性；而 Post-Norm 在表达能力上更强，但需要额外设计才能稳定训练。对于不同场景，应在稳定性与表达能力之间权衡：现代大规模模型首选 Pre-Norm，追求极致表达力的中小规模任务可考虑 Post-Norm + warmup，超深层场景可采用 DeepNorm 等折中方案

> 详细分析参考 [[Why pre-norm better than post-norm]]：从恒等变换、梯度流动和训练稳定性对比了 Pre-Norm 和 Post-Norm
### RMSNorm

<div style="text-align: center;">
    <img src="Pasted image 20260605122415.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：RMSNorm 替代 LayerNorm</div>
</div>

传统 LayerNorm 对输入做均值和方差的归一化：

$$
y_i=\frac{x_i-\mu}{\sigma}\cdot\gamma_i+\beta_i
$$

其中 $\mu=\frac{1}{N}\sum_{j=1}^N x_j$，$\sigma=\sqrt{\frac{1}{N}\sum_{j=1}^N(x_j-\mu)^2}$。

**RMSNorm**（Root Mean Square Normalization）去掉了均值中心化（re-centering），只保留缩放（scaling）：

$$
y_i=\frac{x_i}{\text{RMS}(\mathbf{x})}\cdot\gamma_i,\quad\text{RMS}(\mathbf{x})=\sqrt{\varepsilon+\frac{1}{N}\sum_{j=1}^N x_j^2}
$$

其中 $\gamma_i$ 是可学习的缩放参数（无偏置 $\beta$），$\varepsilon$ 是防止除零的小常数。

RMSNorm 相比 LayerNorm 的优势：
- 计算更简单，省去了均值计算和减法操作，速度更快
- 在许多实验中训练稳定性相当甚至更优

> [[Why RMSNorm instead of LayerNorm]] 笔记大致解释了为什么现在主流 LLM 使用 RMSNorm 及其变体而不是LN，但是正如很多大模型的机制一样，仅为工程上的实验，缺乏数学上的证明
### SwiGLU MLP

<div style="text-align: center;">
    <img src="Pasted image 20260605153523.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：SwiGLU MLP 结构</div>
</div>

经典 Transformer 的 MLP 是简单的两层全连接网络，通常使用 ReLU 或 GeLU 激活：

**Classic MLP：**

$$
Y=\sigma(XW_1)W_2
$$

其中 $X\in\mathbb{R}^{N\times D}$，$W_1\in\mathbb{R}^{D\times 4D}$，$W_2\in\mathbb{R}^{4D\times D}$，$\sigma$ 为激活函数。

**SwiGLU MLP** 引入了门控机制（Gated Linear Unit, GLU）并使用 Swish 激活函数：

$$
Y=(\sigma(XW_1)\odot XW_2)W_3
$$

其中 $W_1,W_2\in\mathbb{R}^{D\times H}$，$W_3\in\mathbb{R}^{H\times D}$，$\sigma$ 为 Swish 激活函数（$\text{Swish}(x)=x\cdot\text{sigmoid}(x)$），$\odot$ 表示逐元素乘法。

设 $H=\frac{8D}{3}$ 可以保持与传统 MLP（$D\rightarrow 4D\rightarrow D$）大致相同的总参数量。SwiGLU 通过门控机制让网络能够有选择地传递信息，在许多 NLP 任务中表现优于传统 MLP，被 PaLM、LLaMA 等模型采用。

> 对 Swish 激活函数，GLU 门控机制以及为什么 SwiGLU 有效的猜想，可以参考笔记 [[Why SwiGLU MLP]]

### Mixture of Experts (MoE)

<div style="text-align: center;">
    <img src="Pasted image 20260602115336.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 20：Mixture of Experts 结构</div>
</div>

标准 Transformer 中，每个 block 只有一组 MLP 权重。**Mixture of Experts (MoE)** 在每个 block 中学习 $E$ 组独立的 MLP 权重，每组 MLP 称为一个"专家"（expert）：

$$
W_1: [D\times 4D] \rightarrow [E\times D\times 4D],\quad W_2: [4D\times D] \rightarrow [E\times 4D\times D]
$$

在计算时，每个 token 通过一个路由器（router/gate）被分配给 $A$ 个活跃专家（$A<E$），只有这些被选中的专家参与计算。这意味着：

- **参数量增加 $E$ 倍**：因为有 $E$ 组 MLP 权重
- **计算量仅增加 $A$ 倍**：每个 token 只经过 $A$ 个专家（而非全部 $E$ 个）

例如，假设 $E=8$，$A=2$，那么参数量是标准 Transformer 的 $8$ 倍，但 FLOPs 只有 $2$ 倍，实现了"用更少的计算支撑更大的模型"。

MoE 的优势：
- 大幅增加模型容量而不等比增加计算开销
- 不同专家可以专注于不同类型的输入模式
- 是现代最大规模 LLM 的核心技术

> 几乎所有当今最大的 LLM（如 GPT-4o、GPT-4.5、Claude 3.7、Gemini 2.5 Pro 等）几乎肯定使用了 MoE 架构，总参数量超过万亿（> 1T params），但它们不再公开具体架构细节。

### Tweaking Transformers 小结

Transformer 自 2017 年以来核心架构未发生根本变化，但以下改动已成为现代 Transformer 的标配：

| 改动 | 说明 | 效果 |
|------|------|------|
| **Pre-Norm** | 将 LayerNorm 移到残差连接内部 | 训练更稳定，模型可学习恒等函数 |
| **RMSNorm** | 去掉均值中心化，仅保留 RMS 缩放 | 计算更快，训练稳定性相当或更优 |
| **SwiGLU** | MLP 引入门控 + Swish 激活 | 选择性传递信息，性能优于传统 MLP |
| **MoE** | 多组 MLP 专家 + 稀疏路由 | 大幅增加参数量（$E$ 倍），计算量仅适度增加（$A$ 倍） |

这些改动并非互相排斥——现代大模型通常同时使用 Pre-Norm + RMSNorm + SwiGLU + MoE 的组合，形成大参数量、高计算效率、训练稳定的 Transformer 架构。

### Materials

- [Adaptive Input Representations for Neural Language Modeling (Baevski & Auli, 2018)](https://arxiv.org/abs/1809.10853)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019)](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformers (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)





