---
title: "lecture09 : Object Detection, Image Segmentation, Visualizing"
publish: true
target: CS231n Lecture 09 主线笔记：目标检测、图像分割与模型可视化，并补充 Pre-Norm、RMSNorm、SwiGLU、MoE 四种 Transformer 改进
---

>[!SUMMARY] Table of Contents
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Recap of Tweaking Transformers|Recap of Tweaking Transformers]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Pre-Norm Transformer|Pre-Norm Transformer]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#RMSNorm|RMSNorm]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#SwiGLU MLP|SwiGLU MLP]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Mixture of Experts (MoE)|Mixture of Experts (MoE)]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Tweaking Transformers 小结|Tweaking Transformers 小结]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Materials|Materials]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Computer Vision Tasks Overview|Computer Vision Tasks Overview]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#TASK1：Semantic Segmentation|TASK1：Semantic Segmentation]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Sliding Window|Sliding Window]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Fully Convolutional Networks|Fully Convolutional Networks]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Upsampling Methods|Upsampling Methods]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#U-Net|U-Net]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Semantic Segmentation 小结|Semantic Segmentation 小结]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#TASK2：Object Detection|TASK2：Object Detection]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Single Object: Classification + Localization|Single Object: Classification + Localization]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Multiple Objects: Sliding Window|Multiple Objects: Sliding Window]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Region Proposals: Selective Search，R-CNN，Fast R-CNN，Faster R-CNN|Region Proposals: Selective Search，R-CNN，Fast R-CNN，Faster R-CNN]]
>            - [[lecture09  Object Detection, Image Segmentation, Visualizing#Selective Search|Selective Search]]
>            - [[lecture09  Object Detection, Image Segmentation, Visualizing#R-CNN|R-CNN]]
>            - [[lecture09  Object Detection, Image Segmentation, Visualizing#Fast R-CNN|Fast R-CNN]]
>            - [[lecture09  Object Detection, Image Segmentation, Visualizing#Faster R-CNN: Region Proposal Network|Faster R-CNN: Region Proposal Network]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Single-Stage Object Detectors: YOLO / SSD / RetinaNet|Single-Stage Object Detectors: YOLO / SSD / RetinaNet]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#YOLO（You Only Look Once）|YOLO（You Only Look Once）]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#DETR（Detection Transformer）|DETR（Detection Transformer）]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#TASK3：Instance Segmentation|TASK3：Instance Segmentation]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Mask R-CNN|Mask R-CNN]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Visualization & Understanding|Visualization & Understanding]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#First Layer Filters|First Layer Filters]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Saliency Maps|Saliency Maps]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Class Activation Mapping (CAM)|Class Activation Mapping (CAM)]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Grad-CAM|Grad-CAM]]
>        - [[lecture09  Object Detection, Image Segmentation, Visualizing#Visualizing ViT Features|Visualizing ViT Features]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Summary|Summary]]
>    - [[lecture09  Object Detection, Image Segmentation, Visualizing#Materials|Materials]]

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
    <img src="Pasted image 20260606161843.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：Mixture of Experts 结构</div>
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

---

## Computer Vision Tasks Overview

计算机视觉领域的核心任务可以分为四类，按从粗粒度到细粒度的顺序：

<div style="text-align: center;">
    <img src="Pasted image 20260606163232.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 6：Four Major Tasks for CV</div>
</div>

1. **Classification（图像分类）**：判断图像"是什么"，输出整张图的类别标签，不包含空间信息
2. **Semantic Segmentation（语义分割）**：对图像中的每个像素进行分类，区分"猫、草地、树、天空"等不同语义类别，但不区分同一类别的不同实例
3. **Object Detection（目标检测）**：识别图像中每个物体的类别及其空间位置（边界框），区分同一类别的不同实例
4. **Instance Segmentation（实例分割）**：最细粒度的任务，既要检测出每个物体，又要为每个实例生成像素级别的 mask

其中图像分类已经在 CNN 等章节讨论过了，本章主要讨论后三种计算机视觉任务。

---

## TASK1：Semantic Segmentation

语义分割的目标是为图像的每一个像素预测一个类别标签。与图像分类只输出一个全局标签不同，语义分割需要输出与输入图像相同尺寸的密集预测（dense prediction）。也就是说，这是一个N2N的任务。

### Sliding Window

最朴素的方法是 **Sliding Window**：以每个像素为中心取一个小 patch（包含周围上下文），送入 CNN 对该中心像素进行分类。

<div style="text-align: center;">
    <img src="Pasted image 20260606163421.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：Sliding Window 方法：对每个像素取周围 patch，用 CNN 分类该中心像素</div>
</div>

> Sliding Window 语义分割有什么问题？
> 非常低效！每个像素都需要独立运行一次 CNN，计算量极大，而且相邻像素的 patch 高度重叠，大量计算被浪费。更重要的是，patch 大小的选择意味着感受野与效率之间的 trade-off——patch 太小则信息不足，太大则更加低效。

### Fully Convolutional Networks

**核心思想**：设计一个全卷积网络，输入任意尺寸图像，输出相同空间尺寸的预测。网络前半部分为下采样（标准卷积 + pooling），用于提取语义特征；后半部分为上采样，用于恢复空间分辨率。

<div style="text-align: center;">
    <img src="Pasted image 20260606164715.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：Fully Convolutional Network：Downsampling → Upsampling 架构</div>
</div>

典型设计：输入图像 $\mathbb{R}^{3\times H\times W}$ → 经过卷积和池化下采样 → 低分辨率高通道特征图 → 上采样层恢复至原始尺寸 → 输出 $\mathbb{R}^{C\times H\times W}$，其中 $C$ 为类别数。

为了提取语义信息，我们自然而然选择池化层和卷积层进行下采样。下采样可以增大感受野、减少计算量，但代价是丢失了空间细节信息。如果是图像识别，经过足够多的卷积层之后再加上一个全连接层即可，但是对于语义分割任务而言，还需要把图像恢复到原有尺寸，这就需要上采样的过程。

### Upsampling Methods

将低分辨率特征图恢复到高分辨率的几种常见方法：

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
    <img src="Pasted image 20260606165930.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：Nearest Neighbor and Bed of Nails</div>
</div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
    <img src="Pasted image 20260606170601.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 10：Max Unpooling 上采样</div>
</div>
</div>

1. **Nearest Neighbor（最近邻插值）**：最简单的上采样方法，填充值为最近的像素值。无需学习参数，但会产生块状伪影

2. **Bed of Nails**：又称为 zero insertion ，最暴力的上采样方式，将输入值放在对应位置，其余位置填充 0。简单但会产生稀疏的输出

3. **Max Unpooling**：记录下采样时 max pooling 的最大值位置，上采样时将值放回原位置，其余位置填 0。相比 Bed of Nails，能更好地保留空间结构信息

> Max Unpooling 相对于之前两种上采样的优势是什么？
> Max Unpooling 利用了下采样阶段记录的最大值位置信息，使得上采样后的特征图能够在空间位置上更加匹配原始图像的结构。但由于 pooling 必然丢失非最大值位置的信息，仅靠 unpooling 仍然无法完全恢复细节。

4. **Transposed Convolution ：可学习的上采样方法。给定一个输入，通过学习一个卷积核来"散布"输入值到更大的输出区域。

<div style="text-align: center;">
    <img src="Pasted image 20260606235741.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 11：Transposed Convolution：可学习的上采样</div>
</div>

与普通卷积的 stride 对应：如果把 stride 理解为输入上的移动步长，那么转置卷积的 stride 则对应输出上的移动步长。对于 stride $>1$，输出尺寸会大于输入尺寸，实现上采样。

> 什么是转置卷积？它与上采样后卷积的关系是什么？
> 1. 转置卷积是普通卷积关于输入的梯度运算——通过插 0 + 可学习卷积核同时完成空间放大和特征精炼，两者**耦合**在同一操作中。Upsample+Conv 则是**解耦**方案：插值做空间放大（确定性的、不可学习），卷积做特征精炼（可学习）。
> 2. 关键区别：转置卷积理论上表达力更强但容易产生 checkerboard 伪影（Odena 2016），Upsample+Conv 更稳定、实践中效果持平或更优。StyleGAN 的切换（Karras 2019）是现代实践的转折点。
> 更详细的解析比较可以参考[[Upsample+Conv VS Transposed Conv]]，原课程的PPT对此做了一些精简，只提及了转置卷积而并未提到上采样+卷积

### U-Net

<div style="text-align: center;">
    <img src="Pasted image 20260607012815.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 12：U-Net 架构</div>
</div>


U-Net 在 FCN 的基础上加入了 **skip connections（跳跃连接）**，将下采样路径的特征图直接 **拼接** (Concatenate)到 **对应分辨率** 的上采样路径中。这种设计让网络在上采样时能够同时利用：
- 深层的高语义特征（知道"是什么"）
- 浅层的空间细节（知道"在哪里"）

U-Net 最初用于生物医学图像分割，现已成为语义分割领域最经典的 backbone 之一，广泛应用于各种分割任务。

### Semantic Segmentation 小结

- Sliding window 虽直观但极其低效
- Fully Convolutional Networks 采用 Downsampling + Upsampling 的端到端架构
- 多种上采样方法（最近邻、Bed of Nails、Max Unpooling、转置卷积）各有优劣
- U-Net 通过 skip connections 融合多层特征，显著提升分割精度

> 语义分割的输出是一整张图，怎么设计损失函数呢？是不是对图像标注的要求很高？
> 如果你也有类似的疑问，可以看看我的笔记 [[The loss function for semanic segmentation]]
---

## TASK2：Object Detection

目标检测需要同时回答两个问题：**图中有什么物体**（分类）以及**它们分别在哪里**（定位）。这比单纯的图像分类或语义分割更复杂。以下从简单到复杂逐级演进。

### Single Object: Classification + Localization

<div style="text-align: center;">
    <img src="Pasted image 20260607213843.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 13：分类 + 定位：单目标场景</div>
</div>

对于只有单个物体的简单场景，我们可以将分类和定位合并为一个多任务学习（multitask learning）问题。网络共享 backbone 特征提取器，但有两个输出分支：

- **分类分支**：输出 softmax 类别概率
- **定位分支**：回归边界框的 4 个坐标 $(x, y, w, h)$

训练时使用 **Multitask Loss**：

$$
L = L_{\text{class}}(\hat{y}, y) + \lambda L_{\text{reg}}(\hat{b}, b)
$$

其中 $L_{\text{class}}$ 是分类损失（如 cross-entropy），$L_{\text{reg}}$ 是回归损失（如 L2 或 smooth L1），$\lambda$ 是平衡两个损失的权重超参数。

### Multiple Objects: Sliding Window

<div style="text-align: center;">
    <img src="Pasted image 20260607220448.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 14：多目标场景：图中有多个物体需要检测和定位</div>
</div>

当图像中包含多个物体时，单目标方法不再适用。一个朴素的想法是：用 Sliding Window 在图像上滑动不同尺度和宽高比的窗口，对每个窗口运行一次 CNN 。

 Sliding Window 多目标检测的问题主要有两点：
 
1. 需要尝试**所有位置、所有尺度、所有宽高比**，窗口数量庞大，计算极其低效
2. 输入图像中物体可能呈现各种不同的形状和姿势，固定大小的窗口难以完全匹配

### Region Proposals: Selective Search，R-CNN，Fast R-CNN，Faster R-CNN

与其盲目地在所有位置和尺度上滑动窗口，不如先用算法生成少量"可能包含物体"的候选区域（region proposals），再对这些候选区域进行分类。

R-CNN，Fast R-CNN，Faster R-CNN 都是采用 Region Proposals 思想，将目标识别任务分为找到候选区域和识别候选区域类别这两个阶段。

#### Selective Search

**Selective Search** 是一种经典的区域提议算法（无需学习）：

1. 基于像素的颜色/纹理/强度等低级特征进行过分割（over-segmentation）
2. 迭代合并最相似的相邻区域
3. 在多个尺度上生成不同大小的候选框
4. 最终输出约 2000 个候选区域（显著少于滑动窗口）

<div style="text-align: center;">
    <img src="Pasted image 20260607223519.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 15：Selective Search 生成候选区域</div>
</div>

#### R-CNN

<div style="text-align: center;">
    <img src="Pasted image 20260607230405.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 16：R-CNN 流程</div>
</div>

R-CNN 将 Region Proposals 与 CNN 结合：

1. 用 Selective Search 生成约 2000 个候选区域（RoI, Region of Interest）
2. 将每个候选区域 warp/crop 到固定尺寸（如 $224\times224$）
3. 每个区域独立输入 CNN 提取特征
4. 对每个区域的特征用 SVM 分类 + Bbox Regression 回归边界框修正

R-CNN 有什么问题？

1. **极慢**：对每张图要运行约 2000 次 CNN 前向传播（train 时更慢）
2. 训练是多阶段的：先训练 CNN，再训练 SVM，再训练 bbox regressor
3. Selective Search 本身也很慢，且不可学习

#### Fast R-CNN

<div style="text-align: center;">
    <img src="Pasted image 20260607232843.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 17：Fast R-CNN 架构</div>
</div>

R-CNN 对每个RoI都运行一次 CNN，而其中很大一部分是重叠的，如果能够提取一张全局特征图给所有RoI使用，将会大大减少 CNN 运算的次数。

这就是 Fast R-CNN 的核心改进：**共享 Feature Map**。不再对每个候选区域独立运行 CNN，而是：

1. 将整张图像一次性通过 CNN backbone，得到全局特征图
2. 将 RoI 投影到特征图上 
3. 使用 RoI Pooling（Region of Interest Pooling）从把投影的 RoI 缩放到固定尺寸
4. 所有 RoI 共享后续的全连接层
5. 训练分类和边界框回归

> 如何用 RoI 裁剪特征图？RoI Pooling 是如何操作的？
> 1. RoI 与特征图的匹配：ConvNet会保持空间特征，比如从 800×800 的原始图提取出 50×50 的特征图，这里的 Stride=800/50=16，只需要将RoI 的坐标值缩小相同的步长，即可找到与特征图相匹配
> 2. RoI pooling：对于任意大小的 RoI，将其分成 K×K 的方格，对每个方格采用 Max pooling 下采样，将其变为固定尺寸

#### Faster R-CNN: Region Proposal Network

<div style="text-align: center;">
    <img src="Pasted image 20260608010605.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 18：Faster R-CNN = RPN + Fast R-CNN</div>
</div>

Fast R-CNN 优化了 CNN 的运算次数，但是却遇到两个问题：
1. Selective Research 不可学习
2. 由于其为 CPU 算法，比 CNN 慢了一个数量级

所以引入 Faster R-CNN 的关键创新：把 Region Proposal 也融入神经网络，提出了 **Region Proposal Network (RPN)**，替代慢速的 Selective Search。

**整体架构**：

1. 输入图像经 CNN backbone 得到共享特征图（如 $512\times20\times15$）
2. RPN 在特征图上滑动，为每个位置生成 object proposals
3. RoI Pooling 从特征图上裁剪每个 proposal 的区域
4. 对每个 RoI 进行分类和边界框精修

Faster R-CNN 的关键在于 **RPN**：

<div style="text-align: center;">
    <img src="Pasted image 20260608014611.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 19：特征图每个位置放置不同尺度和宽高比的 K 个 Anchor boxes</div>
</div>

 RPN 如何生成最终的 object proposals?
 
1. 设置 Anchor : 对于每个中心，分别取 M 种尺寸和 N 种大小的矩形框，每个矩形框就是一个 Anchor
2. 目标检测：与最终的 K 类别得分不同，只是对目标/背景进行二分类，与 Ground Truth(GT)计算 IoU ，对 IoU>0.7的标记为 1，IoU < 0.3 的标记为 0 (背景)
3. 边界框回归：只对标记为1的 Anchor 做回归
4. 损失函数：对二分类和边界框分别计算损失，其中边界框回归与 GT 做交叉熵，得到得分 P(object)
5. 初步处理：筛除低分 Anchor，并根据边界框回归的结果调整边界框
6. Non-Maximum Suppression（NMS）去重：第一步按得分排名，第二步取最高分 Anchor，第三步把与高分重叠度 IoU>0.7 的 Anchor 删去，重复二三步直到处理完所有 Proposals
7. 最终筛选：根据预设的超参数，选择固定数量的 Proposal 进入下一阶段

> Anchor boxes 的设计思想是什么？
> Anchor boxes 本质上是一种"先验"：预定义一组常见尺度和宽高比的框，让网络只需要预测"锚框到真实框的偏移量"，而不是从零预测绝对坐标。这大大降低了回归的难度。

### Single-Stage Object Detectors: YOLO / SSD / RetinaNet

与两阶段方法不同，**单阶段检测器** 跳过显式的 region proposal 步骤，直接在特征图上预测类别和边界框。

### YOLO（You Only Look Once）

<div style="text-align: center;">
    <img src="Pasted image 20260608143757.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 20：YOLO：将图像划分为 S×S 网格</div>
</div>

**YOLO** 的核心思想：把分离的 RPN、分类器和边框回归器合并为一个统一的回归问题

YOLOv1 的核心步骤可以概括为：
1. YOLO 的 backbone 与 Fast R-CNN 一致，都是先通过卷积层得到特征图
2. 将特征图划分为 $S\times S$ 的网格，每个网格单元负责预测该位置的目标。对每个网格单元，输出 **B 个边界框** 和 **C 个类别概率**
3. 通过 NMS 等机制筛选边界框

需要注意的是，在步骤1和步骤2之间，并没有 RPN 这样的区域选择，也就是说 YOLO 是真正的 end-to-end 的神经网络，以 v1 为例，其架构可以看成是若干个卷积层（得到特征图）+若干个全连接层（从特征图得到边界框和概率）

> 如果把分类器和回归器合并了，那么怎么设计损失函数呢？
> 损失可以分为3个部分，包括边界框的坐标，置信度和类别概率，为了避免背景损失淹没目标损失，背景框和目标框也要做区分，所以总损失由四部分加权构成，具体计算可以参考[[How yolo developed#损失函数设计：多任务的平衡艺术]]

如果你和我一样对 YOLO 的发展过程感兴趣，可以参考我的笔记 [[How yolo developed]]
### DETR（Detection Transformer）

<div style="text-align: center;">
    <img src="Pasted image 20260608183641.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 21：DETR：Transformer-based 目标检测</div>
</div>


DETR（DEtection TRansformer）将目标检测重新定义为集合预测（set prediction）问题，完全抛弃了 Anchors、region proposals 和 NMS 等手工设计组件：

1. CNN backbone 提取图像特征
2. Transformer encoder-decoder 将特征图与一组可学习的 **object queries** 进行交互
3. 每个 object query 直接输出一个预测结果（类别 + 边界框）或"无物体"标记
4. 通过 **Hungarian matching（匈牙利算法）** 将预测框与 ground-truth 框一一配对，进行端到端训练

> object query 是什么？
> object query 是一个超参数，只是一个数字，代表着对象查询量，当大于实际对象数时，查询结果标记为无物体

DETR 的核心优势：

1. 端到端训练，无需 NMS 后处理
2. 无需 Anchor 等手工先验
3. 架构简洁，直接输出固定数量的预测集合

缺点是训练收敛较慢（需要数百个 epoch），后续工作如 Deformable DETR 等改善了收敛速度和精度

>如果你也对匈牙利算法感兴趣，这个例子展示了算法步骤 [HungarianAlgorithm.com - Solve the Assignment Problem](https://www.hungarianalgorithm.com/hungarianalgorithm.php)，如果你对数学推导感兴趣，我写了一份不那么严谨，但是通俗易懂的笔记 [[Understanding Hungarian Algorithm without knowing maths ]]。
>在 DETR 中，使用分类误差和框误差计算总误差，作为 边界框-GT 矩阵的值，通过匈牙利算法找到总误差最小的匹配。
---

## TASK3：Instance Segmentation

**实例分割** 结合了目标检测和语义分割：既要检测出每个物体（如 Faster R-CNN），又要为每个实例输出像素级的 mask（如语义分割）。

### Mask R-CNN

**Mask R-CNN** 在 Faster R-CNN 的基础上增加了一个并行的 mask prediction 分支：

<div style="text-align: center;">
    <img src="Pasted image 20260610230902.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 22：Mask R-CNN 流程</div>
</div>

> 这个图可能有一点误导性，容易让人以为掩膜层和分类、回归头是递进的关系，实际上掩膜头和分类头、回归头是并行的，损失函数由三个部分线性相加得到

整体流程：
1. CNN + RPN 生成 proposals
2. **RoI Align**（改进版 RoI Pooling）提取每个 proposal 的精确特征区域
3. 三个并行分支：分类（$C$ 个分数）、边界框回归（$4C$ 个坐标）、Mask 预测（$C\times28\times28$）

> 为什么使用 RoI Align 而不是 Fast R-CNN 中的 RoI Polling ? RoI Align 相比 RoI Pooling 有什么改进？
> RoI Pooling 涉及两次量化操作（将浮点坐标取整），导致特征与原始图像位置不对齐（misalignment）。RoI Align 使用双线性插值，避免了量化误差，对于像素级精度的 mask 预测至关重要。

<div style="text-align: center;">
    <img src="Pasted image 20260610231333.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 23：Mask R-CNN mask 训练目标示例</div>
</div>

Mask R-CNN 还展示了极强的通用性：同样的架构稍加修改即可用于 **人体姿态估计（pose estimation）**，只需将 mask 分支替换为关键点回归分支。

---

## Visualization & Understanding

神经网络常被批评为"黑盒"。理解模型内部到底在做什么、哪些输入区域对预测最重要，对于调试、验证和信任模型至关重要。

### First Layer Filters

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260611095427.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 24：线性分类器的权重矩阵可视化</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260611095243.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 25：可视化不同架构神经网络的第一个卷积层</div>
    </div>
</div>

要理解模型内部在做什么，一个直观的方法就是可视化其中的权重矩阵，回顾线性分类器，由于通道数始终等于输入层，所以可视化相对简单，比如 car 就大概与车的正脸匹配。

把线性分类器中的全连接层迁移到带卷积层的现代神经网络架构中，可以采用 **可视化第一层卷积核** 方法。由于第一层输入是 RGB 图像，每个 $3\times\text{kernel size}\times\text{kernel size}$ 的卷积核可以直接可视化为图像。

不同 CNN 架构的第一层滤波器表现出惊人的一致性：都学到了**定向边缘检测器**（不同方向的边缘）、**颜色检测器**（互补色对）和**纹理模式**。这与哺乳动物视觉皮层 V1 区的简单细胞功能惊人地相似。

> 为什么只可视化第一层卷积层？
> 之所以往往可视化第一层卷积层，是因为输入通道一般为RGB，直接在RGB上做可视化即可。但是对于较深的卷积层，输入通道数一般大于3，这时候有以下几种方式进行可视化：
> 1. 每个通道分别绘制一张灰度图像
> 2. 用平均、最大值或 L2 Norm 把所有通道聚合到一个维度，可视化一张灰度图像
> 3. 采用 PCA 主成分分析法得到三个主成分，把高维通道投影到三个主成分上，可视化 RGB 图像

### Saliency Maps

<div style="text-align: center;">
    <img src="Pasted image 20260611100901.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 26：计算每个像素对得分的梯度</div>
</div>

**Saliency Maps（显著性图）** 回答的问题是：对于给定预测类别，输入图像的哪些像素影响最大？

核心思想：计算目标类别得分对输入图像的梯度，取绝对值，在 RGB 通道上取最大值：

$$
\text{Saliency}_{i,j} = \max_c \left|\frac{\partial S_{\text{class}}}{\partial x_{i,j,c}}\right|
$$

其中 $S_{\text{class}}$ 是目标类别的得分（softmax 之前），$x_{i,j,c}$ 是位置 $(i,j)$ 处通道 $c$ 的像素值。

### Class Activation Mapping (CAM)

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <div style="text-align: center;">
    <img src="Pasted image 20260611104220.png" width="400" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 27：Class Activation Mapping (CAM) 结构</div>
</div>
    <div style="text-align: center;">
    <img src="Pasted image 20260611104302.png" width="400" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 28：CAM 的热力图可视化</div>
</div>
</div>

**CAM** 专门针对使用 Global Average Pooling (GAP) + 全连接层作为分类头的 CNN。其核心公式为：

$$
S_c = \sum_k w_{k,c} F_k = \sum_k w_{k,c} \left(\frac{1}{HW}\sum_{h,w} f_{h,w,k}\right) = \frac{1}{HW} \sum_{h,w} \underbrace{\sum_k w_{k,c} f_{h,w,k}}_{\text{CAM}_{h,w,c}}
$$

其中 $f\in\mathbb{R}^{H\times W\times K}$ 是最后一层卷积特征图，$F_k$ 是第 $k$ 个通道的 GAP 值，$w_{k,c}$ 是全连接层中通道 $k$ 到类别 $c$ 的权重。

以上是从类别 c 逐元素运算的角度理解，如果所有类别一起算，并用 **矩阵运算** 的角度理解的话，最后一层特征图 $X \in \mathbb{R}^{HW \times K}$，权重矩阵 $W \in \mathbb{R}^{K \times C}$，那么：

$$  
M = XW  
$$

维度是$(HW \times K)(K \times C) = HW \times C$，然后 reshape / transpose 成$M \in \mathbb{R}^{C \times H \times W}$，这就对应图里写的$M \in \mathbb{R}^{C,H,W}$

**Class Activation Map** 为：

$$
M_{c,h,w} = \sum_k w_{k,c} f_{h,w,k}
$$

热力图 $M_c$ 直接反映了每个空间位置对类别 $c$ 的贡献，可以通过 **上采样** 叠加回原始图像生成可视化结果。

### Grad-CAM

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260611112650.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 29：Grad-CAM 计算流程</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260611112754.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 30：Grad-CAM 可视化结果</div>
    </div>
</div>

**Grad-CAM** 是 CAM 的泛化版本，不需要特定的网络结构，可以应用于**任意 CNN 的任意层**。

算法步骤：

1. 选取任意层，其激活图为 $A\in\mathbb{R}^{H\times W\times K}$
2. 计算类别得分 $S_c$ 对激活图 $A$ 的梯度：$\frac{\partial S_c}{\partial A} \in\mathbb{R}^{H\times W\times K}$
3. 对梯度做全局平均池化（GAP），得到每个通道的重要性权重 $\alpha_k$：

   $$
   \alpha_k = \frac{1}{HW}\sum_{h,w}\frac{\partial S_c}{\partial A_{h,w,k}}
   $$

4. 对激活图加权求和，并通过 ReLU 保留对目标类别有正贡献的区域：

   $$
   M_{h,w}^c = \text{ReLU}\left(\sum_k \alpha_k A_{h,w,k}\right)
   $$

> $\alpha_k$ 的含义是什么？为什么 Grad-CAM 适用于所有卷积层？
> $\alpha_k$ 表示激活图第 $k$ 个通道对类别 $c$ 的"重要性"。梯度越大，说明该通道的微小变化对最终预测的影响越大。
> CAM 不能适用于所有卷积层就是因为它依赖于 $w_{k,c}$ ，而只有 GAP+FC 架构的最后一个卷积层满足这个条件；而 Grad-CAM 用梯度计算 $\alpha_k$，自然能够适用于所有网络结构。

Grad-CAM 可以应用于不同深度的层：
- **浅层**：热力图更细粒度，包含更多空间细节，但语义不明确
- **深层**：热力图更粗粒度，语义更明确，突出最重要的判别区域

> 深度与粒度的关系很好理解：层数越深，特征图的空间尺度越小，而热力图的尺寸与特征图一致，映射回输入尺寸的粒度自然更粗
### Visualizing ViT Features

<div style="text-align: center;">
    <img src="Pasted image 20260611114617.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 31：Visualizing ViT features</div>
</div>


ViT 的特征可视化与 CNN 有本质不同。由于 ViT 使用自注意力机制，其感受野从一开始就是全局的（而非 CNN 的局部感受野逐层扩大）。ViT 不同层的 attention map 可视化可以揭示模型关注图像哪些区域来进行判断。

---

## Summary

Lecture 09 涵盖了从 Transformer 改进到计算机视觉任务再到模型可视化的完整脉络：

**Transformer 改进（Recap）**：

- Pre-Norm → 训练更稳定
- RMSNorm → 计算更快
- SwiGLU MLP → 门控机制提升性能
- MoE → 稀疏激活，大参数量、适度计算量

**计算机视觉任务**：

| 任务 | 输出 | 方法演进 |
|------|------|----------|
| **Semantic Segmentation** | 逐像素类别标签 | Sliding Window → FCN → U-Net (+ skip connections) |
| **Object Detection** | 边界框 + 类别 | R-CNN → Fast R-CNN → Faster R-CNN (RPN) → YOLO/SSD (single-stage) → DETR (Transformer) |
| **Instance Segmentation** | 边界框 + 逐实例 mask | Mask R-CNN（在 Faster R-CNN 上加 mask 分支） |

**模型可视化**：

- **First Layer Filters**：可视化卷积核，发现边缘/颜色/纹理检测器
- **Saliency Maps**：通过梯度反向传播找到关键像素
- **CAM**：通过 GAP 权重加权特征图，但仅适用于特定网络结构
- **Grad-CAM**：CAM 的泛化版本，适用于任意 CNN 的任意层
- **ViT Features**：自注意力机制的可视化揭示全局感受野特性

## Materials

- [Fully Convolutional Networks for Semantic Segmentation (Long et al., CVPR 2015)](https://arxiv.org/abs/1411.4038)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., MICCAI 2015)](https://arxiv.org/abs/1505.04597)
- [Rich feature hierarchies for accurate object detection and semantic segmentation (Girshick et al., CVPR 2014)](https://arxiv.org/abs/1311.2524)
- [Fast R-CNN (Girshick, ICCV 2015)](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren et al., NeurIPS 2015)](https://arxiv.org/abs/1506.01497)
- [You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., CVPR 2016)](https://arxiv.org/abs/1506.02640)
- [SSD: Single-Shot MultiBox Detector (Liu et al., ECCV 2016)](https://arxiv.org/abs/1512.02325)
- [Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002)
- [End-to-End Object Detection with Transformers (Carion et al., ECCV 2020)](https://arxiv.org/abs/2005.12872)
- [Mask R-CNN (He et al., ICCV 2017)](https://arxiv.org/abs/1703.06870)
- [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (Simonyan et al., ICLR Workshop 2014)](https://arxiv.org/abs/1312.6034)
- [Learning Deep Features for Discriminative Localization (Zhou et al., CVPR 2016)](https://arxiv.org/abs/1512.04150)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (Selvaraju et al., CVPR 2017)](https://arxiv.org/abs/1610.02391)
- [When Vision Transformers Outperform ResNets (Chen et al., ICLR 2022)](https://arxiv.org/abs/2106.04560)
- [CS231n 2024/2025 Lecture 9 Slides](https://cs231n.stanford.edu/slides/2025/lecture_9.pdf)





