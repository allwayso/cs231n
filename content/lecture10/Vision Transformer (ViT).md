---
title: "Vision Transformer (ViT) 详解"
publish: true
target: "系统介绍 ViT 架构：patch embedding、CLS token、positional encoding 及与 CNN 的本质差异"
---

## CNN 的局限：为什么需要 ViT？

CNN 统治计算机视觉多年，但它有一个根本性的结构偏置：**局部感受野**。卷积核每次只看 $K \times K$ 的邻域，靠堆叠层数逐步扩大感受野。

这种设计在捕捉**全局关系**时天然低效——两个相距遥远的像素需要经过很多层才能"看到"彼此。此外，CNN 的卷积核权重是固定的（与输入内容无关），而理想情况下，不同图像可能需要模型关注不同的区域关系。

Transformer 在 NLP 的成功给出了另一种范式：**让每个 token 直接关注所有其他 token**，不预设任何局部结构。ViT 把这个思想原封不动地搬到了视觉领域。

> *"We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks."*
> 我们证明，直接将纯 Transformer 应用于图像 patch 序列，可以在图像分类任务上表现非常好。
> — Dosovitskiy et al., An Image is Worth 16x16 Words, ICLR 2021

---

## ViT 架构：三步将图像变成分类结果

<div style="text-align: center;">
    <img src="Pasted image 20260615170103.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：ViT 总体架构</div>
</div>

ViT 的前向传播可以概括为三步：

### Step 1：图像 → Patch 序列

输入图像 $X \in \mathbb{R}^{H \times W \times C}$（例如 $224 \times 224 \times 3$）被切分为 $N$ 个不重叠的 patch，每个 patch 的尺寸为 $P \times P$：

$$
N = \frac{HW}{P^2}
$$

以 $P=16$ 为例，$N = 14 \times 14 = 196$ 个 patch。

每个 patch 被**展平**为一个向量 $\mathbf{x}_p \in \mathbb{R}^{P^2 \cdot C}$（$16 \times 16 \times 3 = 768$ 维），再通过一个可学习的**线性投影** $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 映射到 Transformer 的隐藏维度 $D$：

$$
\mathbf{z}_i = \mathbf{x}_p^i E, \quad i = 1, \dots, N
$$

这一步可以等价地用一个卷积实现：kernel size = stride = $16$，输入通道 3，输出通道 $D$。

### Step 2：加 CLS Token 和 Positional Encoding

与 BERT 类似，ViT 在 patch 序列最前面**拼接**一个可学习的 $\texttt{[CLS]}$ token embedding $\mathbf{z}_{\text{cls}} \in \mathbb{R}^D$：

$$
[\mathbf{z}_{\text{cls}}; \mathbf{z}_1; \mathbf{z}_2; \dots; \mathbf{z}_N]
$$

与此同时，必须加入**位置编码** $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$，告诉 Transformer 每个 patch 的二维空间位置：

$$
\mathbf{z}_0 = [\mathbf{z}_{\text{cls}}; \mathbf{z}_1 E; \dots; \mathbf{z}_N E] + E_{\text{pos}}
$$

位置编码可以是**可学习的 1D embedding**（标准做法），也可以是 2D 正弦编码。实验表明两者效果相近。

> $\texttt{[CLS]}$ token 有什么用？
> 这里  $\texttt{[CLS]}$ token 是一个监督者。Self-attention 让每个 token 都能聚合全局信息，所以经过 $L$ 层 Transformer 后，$\texttt{[CLS]}$ token 已经通过注意力机制"看到了"所有 patch 的信息，因此它的最终表示可以作为整张图像的**全局特征**，直接送入分类头。
> 其作用类似于 GAP ，也是对信息进行汇聚，效果也差不太多，更像是自然语言处理任务中 BERT 遗留的思想。

### Step 3：Transformer Encoder → MLP Head

序列 $\mathbf{z}_0$ 送入 $L$ 层标准 Transformer Encoder（Multi-Head Self-Attention + MLP + Residual + LayerNorm）：

$$
\mathbf{z}_{\ell}' = \text{MSA}\big(\text{LN}(\mathbf{z}_{\ell-1})\big) + \mathbf{z}_{\ell-1}
$$

$$
\mathbf{z}_{\ell} = \text{MLP}\big(\text{LN}(\mathbf{z}_{\ell}')\big) + \mathbf{z}_{\ell}'
$$

最终取出 $\texttt{[CLS]}$ token 的输出表示 $\mathbf{z}_L^0$，通过一个简单的 MLP Head 得到类别 logits：

$$
\mathbf{y} = \text{LayerNorm}(\mathbf{z}_L^0) W_{\text{head}}
$$

---

## ViT vs CNN：四种本质差异

| | CNN | ViT |
|---|---|---|
| **感受野** | 局部 → 逐层扩大 | 第一层就是全局 |
| **归纳偏置** | 强（局部性、平移等变性） | 弱（几乎只有 patch 切分） |
| **权重计算** | 固定卷积核（与输入无关） | 注意力权重由输入动态计算 |
| **数据需求** | 中等（ImageNet-1K 即可） | 大（需 ImageNet-21K 或 JFT-300M 预训练） |

最关键的差异在于**归纳偏置（inductive bias）**：

- CNN 内置了"邻近像素相关性高"和"平移等变性"这两个强先验，这让它在小数据上泛化好，但也限制了灵活建模远距离关系的能力
- ViT 几乎不预设任何空间结构（只有 patch 切分这一步引入了极弱的局部偏置），让模型从数据中自行学习空间关系。代价是**极度依赖大数据**——在 ImageNet-1K（1.2M 图像）上从头训练时 ViT 不如 ResNet，但在 JFT-300M（300M 图像）上预训练后全面超越

这也解释了为什么 ViT 在 ImageNet-21K 预训练 + ImageNet-1K 微调的标准范式下效果最好：大规模预训练弥补了 ViT 缺乏归纳偏置的弱点。

---

## 一个直观理解：ViT 每一层在做什么？

ViT 各层的行为呈现出与 CNN 不同的模式：

- **浅层**：attention 距离混合——有的 head 关注很近的邻域（类似卷积），有的 head 已经跨越整张图
- **中层**：逐渐形成语义上有意义的关注模式
- **深层**：$\texttt{[CLS]}$ token 的 attention 集中在最具判别力的区域（与 Grad-CAM 热力图高度吻合）

这与 CNN "浅层边缘 → 中层纹理 → 深层语义"的逐层抽象模式本质不同。ViT 从一开始就具备全局视野，只是各层对"看什么"的选择不同。

> 关于 ViT 特征可视化的更多内容，可以参考主笔记中的 [[lecture09  Object Detection, Image Segmentation, Visualizing#Visualizing ViT Features|Visualizing ViT Features]] 一节。

---

## 关键变体

自 ViT（Dosovitskiy et al., 2021）提出以来，涌现了大量改进工作，这里列举几个重要方向：

- **DeiT**（Touvron et al., 2021）：通过知识蒸馏（用 CNN 作为 teacher），让 ViT 在 ImageNet-1K 上也能高效训练，降低数据门槛
- **Swin Transformer**（Liu et al., 2021）：引入局部窗口注意力和层级化结构（patch merging 逐层减半分辨率），让 Transformer 也具备 CNN 式的多尺度特征金字塔
- **MAE**（He et al., 2022）：基于掩码自编码器的自监督预训练——随机遮住 75% 的 patch，用 ViT 从可见 patch 重建缺失 patch，极大提升了训练效率
- **ViT-22B**（Dehghani et al., 2023）：将 ViT 扩展到 220 亿参数，展示了视觉模型的 scaling law

---

## 小结

ViT 的核心创新不在于提出新的 Transformer 组件，而在于证明了：**只要把图像切成 patch 序列，纯 Transformer（不加任何视觉特定的归纳偏置）也能在视觉任务上超越 CNN**。

前提条件是有足够大的预训练数据集——这恰好是 2020 年代算力和数据规模快速增长的背景下自然满足的条件。ViT 的成功标志着计算机视觉从"设计更好的卷积结构"转向了"用更大数据训练更通用的 Transformer"。

## Materials

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers & distillation through attention (Touvron et al., ICML 2021)](https://arxiv.org/abs/2012.12877)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., ICCV 2021)](https://arxiv.org/abs/2103.14030)
- [Masked Autoencoders Are Scalable Vision Learners (He et al., CVPR 2022)](https://arxiv.org/abs/2111.06377)
- [Scaling Vision Transformers to 22 Billion Parameters (Dehghani et al., ICML 2023)](https://arxiv.org/abs/2302.05442)
