---
title: "Upsample+Conv VS Transposed Conv"
publish: true
target: "从上采样的数学原理出发，对比 Transposed Convolution 与 Upsample+Conv 两种方案，剖析 Checkerboard 伪影的根源及现代实践中的取舍"
---
## 从 FCN 到上采样困境

语义分割要求输出与输入同尺寸的密集预测——输入一张 $H \times W$ 的图像，输出 $C \times H \times W$ 的逐像素类别概率。问题在于：CNN 的下采样（pooling / strided conv）不可避免地压缩了空间分辨率。如何把压缩后的特征图"放大"回去？

Long et al. (2015) 在提出 Fully Convolutional Network 时给出了一个在当时看来自然而然的答案——**转置卷积（Transposed Convolution）**，论文中称为 "deconvolution layer"：

> *"We append a deconvolution layer with 2× upsampling that takes the coarse outputs and bilinear interpolation to produce dense per-pixel predictions. ... The deconvolution layer is initialized to bilinear interpolation but can be learned."*
> 我们附加了一个 2 倍上采样的反卷积层，它接收粗糙输出和双线性插值来产生密集的逐像素预测。……反卷积层初始化为双线性插值，但可以被学习。
> — Long et al., Fully Convolutional Networks for Semantic Segmentation, CVPR 2015

注意这里已经埋下了伏笔：FCN 的"反卷积层"被**初始化为双线性插值**，说明 Long 等人意识到完全从头学习上采样核可能不稳定。这暗示了一个更根本的问题——上采样的两种范式之间的张力："纯可学习"和"几何插值 + 特征精炼"。

---

## Transposed Convolution：可学习的上采样核

### 数学定义

Dumoulin & Visin (2016) 给出了完整的卷积算术指南。理解转置卷积的关键是把它看作**普通卷积的反向传播形式**。

考虑一个普通卷积：输入 $i$，卷积核 $k$，stride $s$，输出 $o$。其正向计算是：

$$
o = \text{conv}(i, k, s)
$$

转置卷积等价于**交换前向与反向传播**的操作——给定输出梯度 $\nabla o$，反向传播计算输入梯度 $\nabla i$ 的过程，恰好和转置卷积的前向计算形式一致。更直观地说：

- **普通卷积**（stride $s > 1$）：输入 → 卷积核滑动 → 下采样输出
- **转置卷积**（stride $s > 1$）：对稀疏输入插 0 → 卷积核滑动 → 上采样输出

> *"A transposed convolution is essentially the gradient of a convolution with respect to its input. While a convolution reduces spatial dimensions, its transposed counterpart increases them."*
> 转置卷积本质上是对输入求卷积梯度的操作。普通卷积减小空间维度，而它的转置操作增大空间维度。
> — Dumoulin & Visin, A Guide to Convolution Arithmetic for Deep Learning, 2016

### 谁在学什么？

转置卷积有自己的卷积核权重 $k$，这些权重**完全可学习**。对于一个 $s=2$ 的上采样，每个输出像素是多个输入像素通过**不同权重的加权组合**——这意味着转置卷积同时做了两件事：

1. **空间放大**：通过 strided 操作把低分辨率特征图映射到高分辨率
2. **特征精炼**：通过可学习卷积核把上采样后的特征进行滤波

这两件事被耦合在同一个可学习操作中。耦合带来了表达力，但也带来了训练难度。

---

## Upsample + Conv：解耦上采样与特征精炼

另一种思路是**解耦**：先把空间放大这件事交给不可学习的插值算法，再让卷积专注于特征精炼。

$$
\text{Upsample+Conv}(x) = \text{Conv}(\text{Interpolate}(x, \text{scale}=s))
$$

其中 $\text{Interpolate}$ 可以是：
- **Nearest Neighbor**：最近邻插值，速度最快但质量最粗糙
- **Bilinear Interpolation**：双线性插值，在图像空间上产生平滑过渡

上采样之后通常接一个普通卷积（有时带激活函数），这个卷积的职责纯粹是**修正插值引入的模糊、恢复细节信息**。

> 这种"先插值再卷积"模式没有单一的"创始论文"——它更像是社区在实践中的经验总结。但 Odena et al. (2016) 的工作在推动这种替代方案方面起到了关键作用。

---

## Checkerboard 伪影——转置卷积的阿喀琉斯之踵

### 问题的发现

<div style="text-align: center;">
    <img src="Pasted image 20260607010456.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：棋盘格伪影示例</div>
</div>

Odena et al. (2016) 在 Distill 上的经典文章系统诊断了转置卷积的一个致命问题：**棋盘格伪影（Checkerboard Artifacts）**。

> *"When we have a deconvolution with kernel size that is not divisible by the stride, we get uneven overlap—some parts of the output get 'painted' more times than others. This uneven overlap creates checkerboard-like patterns."*
> 当反卷积的核大小不能被 stride 整除时，我们会得到不均匀的重叠——输出的某些部分比其他部分被"绘制"了更多次。这种不均匀重叠产生了棋盘格状的图案。
> — Odena et al., Deconvolution and Checkerboard Artifacts, Distill, 2016

### 为什么会产生？

以 $s=2, k=3$ 的转置卷积为例：kernel 在输出平面上滑动时，相邻输出位置共享的输入像素数量不一致。具体来说：

<div style="text-align: center;">
    <img src="Pasted image 20260607005847.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：k=3, s=2 的转置卷积产生不均匀重叠</div>
</div>

> 题外话：这里[Odena 的网页](https://distill.pub/2016/deconv-checkerboard/)做的真的很好，用一系列可交互的动图展示了棋盘格伪影的产生逻辑

当 $k$ 不能被 $s$ 整除时（这是最常见的情况，因为 $k$ 通常是奇数如 3、5），卷积核在输出平面上的"足迹"会产生交替的深浅区域——就像国际象棋的棋盘格。更糟的是，**这种伪影会逐层累积**，在深层网络中变得非常明显。

> *"In principle, the model could learn to carefully avoid artifacts by learning weights that make the overlapping positions cancel out. But in practice, neural networks struggle to learn this cancellation perfectly."*
> 原则上，模型可以通过学习使重叠位置相互抵消的权重来小心避免伪影。但在实践中，神经网络很难完美地学到这种抵消。
> — Odena et al., 2016

Odena 等人提出的主要解决方案之一正是 **resize + convolution**——先做图像缩放（最近邻或双线性插值），再接普通卷积。这就是本节的主角 Upsample+Conv。

---

## 多视角对比

### 视角 1：Checkerboard 问题——Odena 的实验证据

这是最有说服力的视角，因为 Odena et al. (2016) 提供了清晰的视觉证据和可复现的实验。

他们训练了一个简单的 GAN 生成器，分别使用转置卷积和 Upsample+Conv 在 MNIST 和 CelebA 上生成图像：

| 方案 | MNIST 生成质量 | CelebA 生成质量 | 伪影 |
|------|---------------|-----------------|------|
| Transposed Conv ($k=5, s=2$) | 有明显棋盘格 | 人脸出现规则纹理 | **严重** |
| Transposed Conv ($k=4, s=2$) | 伪影减少但未消除 | 仍有可见伪影 | 中等 |
| **Upsample (NN) + Conv** | 几乎无伪影 | 几乎无伪影 | **无** |

> 关键发现：**$k$ 能被 $s$ 整除时能缓解但无法根除问题**，因为实际中多层堆叠和激活函数的非线性会使细微的不均匀被放大。而 Upsample+Conv 从结构上杜绝了不均匀重叠的可能性。

这也是为什么 StyleGAN（Karras et al., 2019）明确选择了 Upsample+Conv：

> *"We deviate from the current trend of using transposed convolutions for upsampling. Instead, we use bilinear upsampling followed by a convolution layer. This removes checkerboard artifacts that are commonly seen in generated images."*
> 我们偏离了当前使用转置卷积做上采样的趋势。取而代之，我们使用双线性上采样后再接一个卷积层。这消除了生成图像中常见的棋盘格伪影。
> — Karras et al., A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019

### 视角 2：表达能力——耦合 vs 解耦

来自 Dumoulin & Visin (2016) 对卷积算术的系统分析。

**转置卷积**：每个输出位置 $y[i,j]$ 是多个输入位置通过**各自独立的权重**加权求和得到的。不同输出位置之间共享卷积核，但核的滑动方式让每个输出像素有了独特的"感受野模式"。这种耦合赋予了转置卷积强大的表达能力——理论上，它可以通过学习消除不均匀重叠（但实践困难）。

**Upsample+Conv**：插值是**确定性的、不可学习的**几何操作，卷积只负责在插值后的密集网格上做特征精炼。表达能力被分解为两个正交部分——几何放大（固定规则）+ 局部滤波（可学习）。

| 维度 | Transposed Conv | Upsample + Conv |
|------|----------------|-----------------|
| 空间放大机制 | 可学习（隐式） | 固定规则（显式） |
| 特征精炼 | 与空间放大耦合 | 独立卷积层 |
| 每个输出像素的感受野 | 由 stride 和 kernel size 决定 | 由插值方法 + 后续卷积 kernel size 决定 |
| 表达能力上限 | 理论上更高 | 受限于插值质量 |

理论上看，转置卷积的表达能力应该更强——毕竟它多了一组可学习的"如何放大"的参数。但实践中的伪影问题和训练不稳定性很大程度上抵消了这个理论优势。

### 视角 3：参数效率与训练稳定性

这个视角综合了 Odena (2016) 的稳定性分析和社区实践经验。

**参数量**：
- 转置卷积：一个 $k \times k \times C_{in} \times C_{out}$ 的卷积核
- Upsample+Conv：插值无参数 + 一个 $k \times k \times C_{in} \times C_{out}$ 的普通卷积核

两者参数量相同，但转置卷积把这些参数同时用于"放大"和"精炼"两个任务，参数压力更大。Upsample+Conv 的参数完全专注于一个子任务：在已经插值好的密集网格上提取特征。

**训练稳定性**：
- 转置卷积：$k$ 不能被 $s$ 整除时，梯度流经不均匀重叠区域，某些参数更新信号过强、另一些过弱，导致优化困难。Odena 的发现是即使理论上可学习，实践中的梯度动力学（gradient dynamics）不利于消除伪影。
- Upsample+Conv：插值是确定性的无梯度操作，梯度仅通过后续卷积传播，路径均匀、稳定。

> 在实践中，初始化策略也会放大差异。FCN 论文将转置卷积初始化为双线性插值核——这本质上是在"假装"先做双线性上采样，再让训练过程中逐渐偏离。但如果初始化和训练不够精细，模型也容易学到坏的局部最优。

---

## 现代实践中的选择

### 两种方案的代表性使用者

| 架构方向 | 上采样方案 | 代表模型 |
|---------|-----------|---------|
| 语义分割 | Transposed Conv | **FCN** (Long 2015), **U-Net** (Ronneberger 2015) |
| 语义分割 | Bilinear Upsample + Conv | **DeepLab** (Chen et al., 2017+) |
| 图像生成 | Transposed Conv | **DCGAN** (Radford et al., 2016) |
| 图像生成 | Upsample + Conv | **StyleGAN / StyleGAN2** (Karras et al., 2019, 2020) |
| 超分辨率 | Sub-Pixel Conv (另类) | **ESPCN** (Shi et al., 2016) |
| 现代 LLM 多模态 | Conv 或 Upsample + Conv | 多数现代架构 |

### 为什么 U-Net 还在用转置卷积？

U-Net（Ronneberger et al., 2015）选择了转置卷积做上采样，但它的成功依赖两个关键设计：

1. **Skip Connection**：将编码器的浅层高分辨率特征直接拼接到解码器，弥补了转置卷积丢失的细节。这意味着转置卷积的"表达能力"压力被 skip connection 分担了。
2. **医学图像的稀疏性**：医学图像（U-Net 的目标场景）通常包含大面积的背景区域，checkerboard 伪影在均匀背景上更不明显——不像自然图像生成那样敏感。

### 为什么 StyleGAN 坚决淘汰了转置卷积？

StyleGAN（Karras et al., 2019）是一个从转置卷积切换到 Upsample+Conv 的典型案例。早期的 PGGAN (Karras et al., 2018) 仍在使用转置卷积，但 StyleGAN 明确指出了伪影问题并切换了方案。原因在于：**GAN 的判别器对生成图像的规律性伪影极其敏感**——任何周期性的纹理模式都可能被判别器捕捉为"假"的线索，从而破坏对抗训练的平衡。

---

## 总结

### 全维度对比

| 维度 | Transposed Convolution | Upsample (Bilinear/NN) + Conv |
|------|----------------------|------------------------------|
| **机制** | 可学习卷积核通过 strided 操作同时完成放大与滤波 | 几何插值放大 + 独立卷积滤波——解耦设计 |
| **空间放大** | 隐式，由 kernel size 和 stride 共同决定 | 显式，由插值算法决定（最近邻/双线性） |
| **特征精炼** | 与放大耦合在同一操作中 | 独立卷积层，职责单一 |
| **参数量** | $k^2 \cdot C_{in} \cdot C_{out}$ | $k^2 \cdot C_{in} \cdot C_{out}$（相同） |
| **Checkerboard 伪影** | **高危**（$k$ 不能整除 $s$ 时） | **无**（插值在空间上均匀分布） |
| **训练稳定性** | 梯度不均匀，需小心初始化 | 梯度路径均匀，训练稳定 |
| **表达能力上限** | 理论上更高（空间放大也可学习） | 受限于插值质量 |
| **表达能力实践** | 上限被伪影和优化困难侵蚀 | 在实践中通常持平或更优 |
| **典型使用门槛** | 需要精心选择 $k$ 与 $s$ 的整除关系 | 即插即用，几乎无门槛 |
| **代表论文** | FCN (2015), U-Net (2015), DCGAN (2016) | StyleGAN (2019), DeepLab (2017+) |

### 核心理由

Upsample+Conv 在实践中取代转置卷积的趋势，源于一条清晰的因果链：

1. **转置卷积的耦合设计**：将空间放大和特征精炼耦合在同一个可学习操作中——理论表达力强，但训练难度大
2. **Checkerboard 伪影的本质**：$k$ 不能被 $s$ 整除时产生不均匀重叠——Odena et al. (2016) 给出了清晰的诊断和视觉证据
3. **可学习性无法自救**：理论上转置卷积可以学会消除不均匀重叠，但实践中的梯度动力学使模型倾向于学到导致伪影的局部最优
4. **解耦方案的简洁性**：Upsample+Conv 将"放大"交给确定性算法、将"精炼"交给卷积——职责分离，各司其职，训练稳定且几乎不产生伪影
5. **StyleGAN 的关键验证**：在伪影最敏感的 GAN 生成任务中，StyleGAN 从转置卷积切换到 Upsample+Conv 后效果显著提升，成为了现代生成模型的事实标准

> **转置卷积并非"坏"的技术——它是当时（2015）解决"端到端可学习上采样"这一新问题的最直接方案。Upsample+Conv 也不是"更好"的万能答案——它的成功在于认识到"不是所有东西都需要端到端学习"。当插值这种成熟的信号处理技术已经很好地解决了空间放大问题时，把可学习的参数留给更重要的特征精炼任务，是一种务实的工程智慧。**

---

## 参考文献

- [Fully Convolutional Networks for Semantic Segmentation (Long et al., CVPR 2015)](https://arxiv.org/abs/1411.4038)：提出了 FCN 架构，首次在端到端语义分割网络中使用可学习的"反卷积层"做上采样，并将其初始化为双线性插值。

- [A Guide to Convolution Arithmetic for Deep Learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285)：系统定义了卷积、转置卷积、空洞卷积等操作的数学关系，给出了 $i$（输入尺寸）、$k$（核尺寸）、$s$（stride）、$p$（padding）、$o$（输出尺寸）之间的普适公式。

- [Deconvolution and Checkerboard Artifacts (Odena et al., Distill 2016)](https://distill.pub/2016/deconv-checkerboard/)：系统诊断了转置卷积产生棋盘格伪影的根本原因——不均匀重叠，并提出 resize + convolution 作为解决方案。这篇文章是 Upsample+Conv 替代转置卷积最重要的推动力。

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., MICCAI 2015)](https://arxiv.org/abs/1505.04597)：在转置卷积上采样路径中引入 skip connection，将编码器的浅层特征拼接到解码器，有效弥补了上采样中的细节丢失。

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al., ICLR 2016)](https://arxiv.org/abs/1511.06434)：DCGAN 在生成器中使用"fractionally-strided convolutions"（即转置卷积）做上采样，确立了转置卷积在图像生成领域的早期标准。

- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (Shi et al., CVPR 2016)](https://arxiv.org/abs/1609.05158)：提出了 Sub-Pixel Convolution（Pixel Shuffle），一种不同于转置卷积和 Upsample+Conv 的第三种上采样策略——通过通道重排实现高效的空间放大。

- [A Style-Based Generator Architecture for Generative Adversarial Networks (Karras et al., CVPR 2019)](https://arxiv.org/abs/1812.04948)：StyleGAN 明确放弃了转置卷积，改用双线性上采样 + 卷积的方案，以消除 checkerboard 伪影，成为现代图像生成模型的标准实践。