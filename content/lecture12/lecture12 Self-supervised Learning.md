---
title: "Lecture 12: Self-supervised Learning"
publish: true
target: CS231n Lecture 12 主线笔记：自监督学习的动机与框架、基于图像变换的前置任务、掩码自编码器与对比表示学习
---

>[!SUMMARY] Table of Contents
>    - [[lecture12 Self-supervised Learning#Self-Supervised Learning: Motivation and Framework|Self-Supervised Learning: Motivation and Framework]]
>        - [[lecture12 Self-supervised Learning#The Challenge: Labeled Data and Learned Representations|The Challenge: Labeled Data and Learned Representations]]
>        - [[lecture12 Self-supervised Learning#The SSL Paradigm: Pretext Task → Encoder → Downstream Task|The SSL Paradigm: Pretext Task → Encoder → Downstream Task]]
>        - [[lecture12 Self-supervised Learning#Four Classic Pretext Tasks|Four Classic Pretext Tasks]]
>        - [[lecture12 Self-supervised Learning#How to Evaluate SSL Methods|How to Evaluate SSL Methods]]
>        - [[lecture12 Self-supervised Learning#Broader Impact of SSL|Broader Impact of SSL]]
>    - [[lecture12 Self-supervised Learning#Pretext Tasks from Image Transformations|Pretext Tasks from Image Transformations]]
>        - [[lecture12 Self-supervised Learning#Rotation Prediction|Rotation Prediction]]
>        - [[lecture12 Self-supervised Learning#Jigsaw Puzzle / Patch Location|Jigsaw Puzzle / Patch Location]]
>        - [[lecture12 Self-supervised Learning#Image Inpainting (Context Encoders)|Image Inpainting (Context Encoders)]]
>        - [[lecture12 Self-supervised Learning#Image Colorization|Image Colorization]]
>        - [[lecture12 Self-supervised Learning#Extension: Video Colorization & Emergent Tracking|Extension: Video Colorization & Emergent Tracking]]
>    - [[lecture12 Self-supervised Learning#Limitations of Individual Pretext Tasks|Limitations of Individual Pretext Tasks]]
>    - [[lecture12 Self-supervised Learning#Masked Autoencoders (MAE)|Masked Autoencoders (MAE)]]
>        - [[lecture12 Self-supervised Learning#The Masking Idea and Asymmetric Architecture|The Masking Idea and Asymmetric Architecture]]
>        - [[lecture12 Self-supervised Learning#Training: MSE Loss on Masked Patches|Training: MSE Loss on Masked Patches]]
>        - [[lecture12 Self-supervised Learning#Linear Probing vs Fine-Tuning|Linear Probing vs Fine-Tuning]]
>        - [[lecture12 Self-supervised Learning#MAE vs DINO vs MoCo v3|MAE vs DINO vs MoCo v3]]
>    - [[lecture12 Self-supervised Learning#Contrastive Representation Learning|Contrastive Representation Learning]]
>        - [[lecture12 Self-supervised Learning#Core Idea and Formulation|Core Idea and Formulation]]
>        - [[lecture12 Self-supervised Learning#SimCLR: A Simple Framework|SimCLR: A Simple Framework]]
>        - [[lecture12 Self-supervised Learning#MoCo: Momentum Contrast|MoCo: Momentum Contrast]]
>        - [[lecture12 Self-supervised Learning#MoCo v2: Best of Both Worlds|MoCo v2: Best of Both Worlds]]
>        - [[lecture12 Self-supervised Learning#Sequence Contrastive Learning: CPC|Sequence Contrastive Learning: CPC]]
>        - [[lecture12 Self-supervised Learning#DINO: Self-Distillation with No Labels|DINO: Self-Distillation with No Labels]]
>    - [[lecture12 Self-supervised Learning#Summary|Summary]]
>    - [[lecture12 Self-supervised Learning#Materials|Materials]]

## Self-Supervised Learning: Motivation and Framework

### The Challenge: Labeled Data and Learned Representations

在前面的课程中，我们覆盖了计算机视觉的几乎所有核心任务——从分类、语义分割到目标检测与实例分割——这些任务都依赖于一个共同的前提：拥有足够多的标注数据。但当我们希望训练更大规模的神经网络时，这一前提本身就构成了瓶颈。

回顾 Lecture 07 和 Lecture 08 中关于特征学习的讨论：无论是 CNN 还是 Transformer，其中间层学到的特征在语义空间中是高度有意义的——在特征空间中用 L2 距离做最近邻检索，就能找到与查询图像语义相似的样本。这意味着，一个训练好的神经网络本质上是一个强大的特征提取器。然而，要获得这样的特征提取器，传统监督学习路径必须经过"图像 → 标注"的端到端训练，而大规模标注——特别是像素级标注——是极其昂贵且不可扩展的。

这就引出了本讲的中心问题：**能否在不需要海量人工标注的前提下，训练出一个能提取优质视觉特征的神经网络？** 自监督学习正是在这一动机下诞生的。它不依赖外部标注，而是从数据本身构造监督信号，让模型在解决一个"人为定义"的任务的过程中学会理解图像。

### The SSL Paradigm: Pretext Task → Encoder → Downstream Task

自监督学习的核心范式可以概括为两阶段训练。

第一阶段——**pretext task（前置任务）**——在无标签数据上定义一个由数据自身提供标签的任务：图像经过 encoder 提取特征，再通过 decoder/classifier/regressor 映射到 pretext 的输出空间，而这些输出标签完全由数据自动生成，无需人工参与。encoder 在求解 pretext task 的过程中被迫学习对图像内容的结构化理解，从而获得通用的视觉表征。

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <div style="text-align: center;">
    <img src="Pasted image 20260619155835.png" width="400" />
	</div>
    <div style="text-align: center;">
    <img src="Pasted image 20260619155908.png" width="400" />
	</div>
</div>
<div style="font-size: 1em; color: #888; margin-top: 5px;text-align: center;">图 1：SSL 两阶段范式——左：pretext task 训练阶段，标签自动从数据生成；右：downstream task 迁移阶段，冻结/微调 encoder 并附加浅层分类器</div>


第二阶段——**downstream task（下游任务）**——才是我们真正关心的应用。此时 pretext task 的 decoder 部分被丢弃，仅保留训练好的 encoder 作为特征提取器。在 encoder 之上附加一个浅层网络（通常仅一层线性分类器），用目标任务的小量标注数据训练即可。如果 pretext task 设计得当，encoder 学到的特征已经足够通用，少量标注就能实现优秀的迁移效果。

encoder 和 decoder 的具体关系取决于 pretext task 的设计：在 _rotation prediction_ 中，"decoder"只是一个简单的全连接分类头，与 encoder 构成一个统一的网络；在 _autoencoder_ 式的重建任务中，encoder 和 decoder 是两个独立的网络，有时甚至是非对称的。这种灵活性正是 SSL 的一大优势——同一个 encoder 可以适配任意下游任务。

### Four Classic Pretext Tasks

自监督学习并不是一个新概念，早期的工作围绕"对图像施加变换，然后让模型预测变换参数"这一思路展开，形成了四类经典 pretext task：**图像补全**（image completion）、**旋转预测**（rotation prediction）、**拼图**（jigsaw puzzle）和**着色**（colorization）。

<div style="text-align: center;">
    <img src="Pasted image 20260619160136.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：四类经典 pretext task——从左到右依次为图像补全、旋转预测、拼图和着色</div>
</div>

这四类任务的共同点在于它们满足一个好的 pretext task 所必需的两个条件：第一，**解决该任务迫使模型学到好的特征**——例如，要判断图像是否被旋转，模型必须理解物体的正常朝向；第二，**标签自动从数据生成，无需人工标注**——旋转角度、patch 排列、被 mask 的像素值、颜色通道，这些都是数据本身包含的信息。我们将在接下来的各节中逐一展开这四类方法。

> 看到这里感觉很熟悉啊，这不就是 lecture06 中出现的 data argumentation 吗，似乎都是为了学习高级语义特征，区别在哪里呢？
> 如果你也有相似的问题，可以参考我的笔记 [[Pretext Task vs Data Augmentation]]
### How to Evaluate SSL Methods

由于自监督学习的训练过程不涉及下游任务的标签，如何评估一个 SSL 方法的质量本身就是一个需要回答的问题。课件将评估维度归纳为五个方面：

<div style="text-align: center;">
    <img src="Pasted image 20260619160510.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：SSL 评估流程——pretext task 训练特征提取器 → 附加浅层网络 → 在目标任务的少量标注数据上评估</div>
</div>

- **Pretext Task 表现**：模型在 pretext task 本身的性能（如旋转分类准确率），虽然我们最终并不关心这个指标，但它反映了训练是否收敛。
- **表征质量**：包括 **linear evaluation protocol**（冻结 encoder，仅训练线性分类器以衡量特征线性可分性）、聚类性能、以及通过 t-SNE 等降维算法可视化的特征分离度。
- **鲁棒性与泛化性**：特征在不同数据集和分布偏移下的表现。
- **计算效率**：训练时间和资源开销。
- **下游任务性能**：这是**最重要的评估维度**——将学到的表征迁移到分类、检测、分割等目标任务上，观察实际性能增益。

需要强调的是，linear evaluation（线性 probing）和 fine-tuning 探测的是表征的不同侧面：linear probing 直接衡量特征质量——如果特征本身已经线性可分，说明 pretext task 确实学到了好的表征；fine-tuning 则发挥模型的全部潜力，允许 encoder 根据目标任务进一步调整。

### Broader Impact of SSL

自监督学习的影响远不止计算机视觉。事实上，正是 SSL 范式驱动了几乎所有现代大型语言模型的发展——GPT-4 通过 next-token prediction 这一 pretext task 在海量无标注文本上训练，无需任何人工标注即可学到强大的语言表征。在语音领域，WaveNet 使用自回归预测作为 pretext task；在机器人领域，自监督学习使机器人能够从大量未标注的传感器数据中学到有用的状态表征。这一范式之所以强大，正在于它从根本上移除了标注瓶颈：只要有原始数据，就可以训练。

---

## Pretext Tasks from Image Transformations

### Rotation Prediction

旋转预测是最直观的 pretext task 之一。其背后的假设被称为 **Visual Common Sense**：一个模型只有具备对物体"正常朝向"的常识性认知——比如汽车应该在地上行驶、人脸通常朝上、建筑物的墙壁是垂直的——才能判断一幅图像是否被旋转，以及旋转了多少度。

具体实现由一个简单的分类任务完成：对输入图像随机施加 $0^\circ$、$90^\circ$、$180^\circ$ 或 $270^\circ$ 的旋转（共四类），然后用 CNN 预测施加了哪一种旋转。值得注意的是，这里的标签不是物体类别，而是旋转角度——一个完全由数据变换自动生成的标签。

<div style="text-align: center;">
    <img src="Pasted image 20260619165317.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：RotNet 架构——随机旋转图像 → CNN 提取特征 → 4-way 分类器预测旋转角度</div>
</div>

在 CIFAR-10 上的半监督实验展示了该方法的效果：冻结前两层卷积，仅用少量标注样本训练第三层卷积和线性分类器，经过 rotation pretext task 预训练的模型在训练初期就展现出远高于随机初始化的准确率。虽然对于 CIFAR-10 这类简单任务，最终收敛后的监督学习和自监督预训练可能会达到相近的准确率，但在更难的任务上，预训练带来的优势是决定性的。

在 PASCAL VOC 2007 上的迁移实验更清晰地说明了这一点。该数据集同时包含分类、检测和分割任务，使用 AlexNet 在 ImageNet 上以 rotation pretext task 预训练后，在分类任务上的表现显著优于随机初始化，且逼近了使用完整 ImageNet 标签的监督预训练——尽管 SSL 路径完全没有使用任何类别标签。

<div style="text-align: center;">
    <img src="Pasted image 20260619165542.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：PASCAL VOC 迁移结果——Rotation SSL 远优于 No pretraining，逼近 Supervised ImageNet pretrain</div>
</div>

值得注意的是注意力图的可视化分析。监督学习训练的模型倾向于只关注最具判别性的区域——例如识别一只鸟时，可能只盯着鸟的眼睛和喙，而忽略翅膀和身体。相反，rotation pretext task 迫使模型对整个物体形成更全面的理解——因为判断"图像是否被旋转"需要考虑物体的整体结构和其与背景的关系。因此，自监督预训练的注意力图覆盖了更大的物体区域，这种 holistic 的特征在下游任务中往往更具泛化性。

<div style="text-align: center;">
    <img src="Pasted image 20260619170552.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 6：注意力图对比——自监督预训练关注更全面的物体区域（holistic），监督学习仅聚焦于最具判别性的局部</div>
</div>

### Jigsaw Puzzle / Patch Location

旋转预测测试的是模型对全局朝向的推理能力，而拼图任务则考察模型对**空间关系**的理解。基本思路是将图像划分为 $3 \times 3$ 的网格，取出 9 个 patch，打乱顺序后让模型预测正确的排列。

早期版本（Doersch et al., 2015）将这个任务简化为预测一个给定 patch 相对于中心 patch 的位置——一个 8-way 分类问题。后续工作（Noroozi & Favaro, 2016）将任务扩展为预测完整的排列：理论上有 $9! = 362,880$ 种排列，但许多排列只涉及一两个 patch 的微小位置交换，不足以构成有意义的挑战。因此，作者从所有排列中选取 64 个彼此距离最大的排列（最大汉明距离），将任务简化为一个 64-way 分类问题。

> 汉明距离指的是什么？64-way 分类问题是如何设计模型的？为什么对排列做损失，而不是逐 patch 独立预测位置呢？
> 1. 汉明距离 (Hamming Distance)：两个等长序列之间，对应位置不相等的个数。
> 2. 9个 patch 经过 shuffle （要求 shuffle 结果属于这64种汉明距离最大排列之一）后分别进 CNN 得到特征向量，经过 cancat 之后送入全连接层，最后通过 softmax 层得到一个64维分数向量，代表模型认为的原始 shuffle 结果与64种预定义位置顺序的匹配程度
> 3. 逐 patch 独立预测位置，分别进回归头的话，就丧失了 patch 之间的空间关系，而这正是 patch location 这个 pretext task 设计的目的所在。

<div style="text-align: center;">
    <img src="Pasted image 20260619170837.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：Jigsaw puzzle 框架——3×3 patch 打乱 → 模型预测 64 种排列之一</div>
</div>

这一任务的核心在于，模型不能仅靠 patch 边界的纹理连续性来"拼接"图像（否则就成了简单的纹理匹配），而必须理解每个 patch 在物体整体中的语义角色——比如"这是猫的左耳朵，所以它应该在最左上角"。为此，网络设计为 context-free：每个 patch 首先独立通过共享的 CNN 提取特征，然后再汇总到分类头——这样模型就无法利用 patch 之间的边界信息作弊，被迫学习语义层面的空间推理。

> 观察图片可以发现，这里的9个 patch 在原图中并不是紧密相连的，patch 之间有微小的间隙，这是为什么呢？
> 这是为了防止模型通过低层特征的连续性来”作弊“。如果不加入间隙，当各个 patch 的特征汇总到全连接层的时候，由于 cnn 提取到的特征仍然保留部分诸如纹理特征的低层特征，回归头还是可能从相邻 patch 的特征相关性来进行拼接；而加入间隙后，相邻 patch 之间的低层特征相关性被减弱，使得模型不得不学习高级语义特征。
> 实际上，回顾 Noroozi & Favaro 2016 的论文，他们在这方面做了消融实验——缩小 gap 确实会导致模型更容易学到低级纹理匹配而非语义理解，尤其在早期训练阶段。所以间隙更多是一种训练正则化，防止模型过早收敛到纹理捷径上。

在同样的 PASCAL VOC 迁移设置下，拼图方法展现出了优于 patch 位置预测方法的性能，验证了更难的 pretext task 可能迫使模型学到更好的特征这一直觉。

### Image Inpainting (Context Encoders)

图像补全的任务定义非常直观：给定一幅被部分遮住的图像，让模型预测被遮住部分的像素值。这项任务同时需要全局场景理解（这是一个卧室还是户外？）和局部纹理合成（缺失区域应该填充什么纹理？），因此要求 encoder 将视觉信息压缩为高度语义化的表征。

<div style="text-align: center;">
    <img src="Pasted image 20260619175106.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：Context Encoder 架构——masked 图像 → encoder → decoder → 重建输出；损失仅计算 masked 区域</div>
</div>

架构采用标准的 Encoder-Decoder（Auto Encoder）形式：encoder 接受带 mask 的输入图像，将其压缩为潜在表征；decoder 从该表征出发重建完整的输出图像。其训练损失由两部分组成：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{adv}}$$

其中 $\mathcal{L}_{\text{recon}}$ 是 L2 重建损失——但仅在被 mask 的区域上计算（通过 element-wise 乘以 mask 实现），因为我们不关心未 mask 区域的输出质量。问题在于，单独的 L2 损失倾向于产生模糊的预测——模型会输出所有可能补全结果的"平均值"，这在像素空间中看起来是平滑但缺乏细节的。为解决这个问题，Context Encoders 引入了一个**对抗损失** $\mathcal{L}_{\text{adv}}$，由一个判别器来判断重建图像是否"看起来真实"，从而推动模型生成更锐利、更逼真的纹理。

对抗损失的具体机制将在下一讲（生成模型）中详细展开，但从自监督学习的角度，关键洞察是：reconstruction loss 捕获了场景的全局结构，而 adversarial loss 确保了感知层面的真实感。二者的组合显著提升了 inpainting 质量。

<div style="text-align: center;">
    <img src="Pasted image 20260619175448.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：Inpainting 效果对比——Input → L2 reconstruction（模糊）→ Adversarial（锐利但不一致）→ Recon + Adv（最佳）</div>
</div>

在 PASCAL VOC 上的迁移实验覆盖了分类、检测和语义分割三项任务，inpainting 预训练在所有任务上均优于随机初始化，展现了重建式 SSL 的通用性。

### Image Colorization

图像着色将自监督学习从"预测变换参数"延伸到"预测数据本身的另一个模态"。其实现依赖 **LAB 色彩空间**：在 RGB 空间中将亮度与颜色分离是困难的，而 LAB 空间天然将图像分解为 L 通道（亮度/明度）和 A、B 两个通道（颜色/色度）。输入 L 通道（即灰度图），预测 A 和 B 通道——这就是一个完美的 pretext task：标签天然存在于彩色图像中，不需要任何人工标注。

<div style="text-align: center;">
    <img src="Pasted image 20260619175747.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 10：Split-Brain Autoencoder——图像分解为 LAB 通道，两个子网络各自从部分通道预测其余通道，最终合并计算损失</div>
</div>

这一思路的优雅扩展是 **Split-Brain Autoencoder**（Zhang et al., 2017）。其核心思想是 cross-channel prediction：将输入图像按通道拆分为两部分 $X_1$ 和 $X_2$，分别训练网络 $F_1: X_1 \to \hat{X}_2$ 和 $F_2: X_2 \to \hat{X}_1$，然后将两个网络的预测合并还原为完整图像，计算 L2 损失。这个框架完全通用——不仅可以用于 L ↔ AB，还可以扩展到 RGB ↔ Depth、不同光谱通道之间的互预测，甚至跨传感器模态。

颜色化的一个有趣特性是其**内在歧义性**：一辆车可以是红色也可以是蓝色，从灰度图中无法唯一确定颜色。这种歧义性恰恰是推动模型学习语义信息的动力——要做出合理的颜色预测，模型必须首先识别出物体的类别（"这是苹果"），然后回忆该类别在训练数据中的典型颜色分布（"苹果通常是红色的"）。因此，解决颜色化任务的过程，本身就是语义理解的过程。

在 Places 数据集上的迁移结果显示，colorization 预训练的特征在场景分类上优于多种早期 SSL 方法，且拼接 $F_1$ 和 $F_2$ 的特征能进一步提升性能。

如果你对 **Split-Brain Autoencoder**（Zhang et al., 2017） 这篇论文采取的损失函数等内容感兴趣，可以看看我的笔记 [[Split-Brain Autoencoder 论文精读]]
### Extension: Video Colorization & Emergent Tracking

着色任务从单张图像扩展到视频，产生的副产品是自监督学习中最优雅的发现之一。视频着色的基本设定是：给定一帧彩色参考帧和后续的灰度帧，预测灰度帧的颜色。这里的关键假设是——**同一物体在视频中应保持颜色一致**，因此模型必须学会跨帧跟踪物体和区域，才能将参考帧的颜色正确地传播到目标帧。

<div style="text-align: center;">
    <img src="Pasted image 20260619203354.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 11：视频着色——参考彩色帧 → target 灰度帧 → 通过 attention 传播颜色</div>
</div>

实现上，这与我们在 Lecture 08 中讨论的 attention 机制紧密相关：对于目标帧中的每个像素，模型计算它与参考帧中所有像素的相似度（attention 权重），然后用这些权重对参考帧的颜色做加权平均，得到该像素的预测颜色。CNN 提取每个像素周围的局部特征用于相似度计算，整个过程端到端训练，损失函数只需比较预测颜色与真实颜色。

更令人兴奋的是这一框架的 **涌现属性**：一旦模型学会了用 attention 传播颜色，这些 attention 权重实际上就编码了跨帧的像素级对应关系。利用这些学到的对应关系，可以零样本地（无需额外训练）传播语义分割 mask 和人体姿态关键点——将参考帧的标注"复制"到目标帧。也就是说，**物体跟踪作为颜色化的副产品自然浮现了出来**。

> 数学中的 Emergency 指的是单个个体在一定规则下组合之后，产生了个体本身并没有的性质，比如神经元组合成神经网络，具有了理解复杂事物的能力。这里的涌现大概是一个比喻，指的是从 video coloring pretext task 得到的 attention 权重，“涌现” 出了跨图片像素对应能力，恰巧使得模型适用于 tracking 任务。

<div style="text-align: center;">
    <img src="Pasted image 20260619203809.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 12：涌现的跟踪能力——学到的 attention 直接用于传播分割 mask</div>
</div>

这完美地展示了自监督学习的魅力：定义一个看似简单的 pretext task（为视频帧着色），模型在解决该任务的过程中不知不觉地学到了远比任务本身更有价值的表征。这也与 Lecture 10 中视频理解的时序对应概念形成了呼应。

---

## Limitations of Individual Pretext Tasks

在继续深入之前，让我们总结一下基于图像变换的 pretext task 的共同特征与局限。

这类方法的核心设计哲学是 **visual common sense**：通过对图像施加变换（旋转、打乱、遮盖、去色），迫使模型在恢复原始信息的过程中学会理解视觉世界的基本规律。一个关键认知是，我们**不关心 pretext task 本身的性能**——旋转分类的准确率有多高并不重要——而是关心 encoder 在此过程中学到的特征对下游任务有多大帮助。

然而，这类方法存在两个根本性局限。其一，**为每个 pretext task 手工设计变换是繁琐的**——旋转、拼图、补全、着色，每种都需要定义变换规则、选择超参数、设计对应的网络架构；其二，也是更本质的问题——**针对特定 pretext task 学到的表征可能不够通用**。一个在旋转预测上训练得很好的 encoder，其学到的特征可能过度适配于"识别物体朝向"这个狭窄任务，而无法很好地泛化到需要理解纹理、材质、部件关系的其他任务。

这就引出了一个更根本的问题：能否定义一个**更通用的 pretext task**，不依赖特定的图像变换，而是基于一个更广泛的原则——"同一物体的不同视图应该在表征空间中靠近，不同物体的视图应该远离"？这正是下一部分——从 Masked Autoencoders 到 Contrastive Learning——试图回答的问题。

---

## Masked Autoencoders (MAE)

### The Masking Idea and Asymmetric Architecture

Masked Autoencoders（MAE, He et al., 2021）代表了重建式 SSL 的现代化版本。其灵感来自 NLP 中极其成功的 **Masked Language Modeling（BERT）**：随机 mask 掉一部分文本 token，让模型从上下文预测被 mask 的 token。但在视觉领域，直接将这个想法照搬过来面临一个关键挑战——**图像具有高度的空间冗余性**。文本 token 是离散且语义密集的，mask 掉一个词需要真正的语言理解才能补全；而图像中相邻像素高度相关，如果你只 mask 掉几个像素，模型可以通过简单的插值从周围像素"猜"出答案，无需任何语义理解。

MAE 的解决方案是**极高的 masking 率**：随机 mask 掉 75% 的图像 patch，仅将剩余的 25% 可见 patch 送入 encoder。这一设计有双重效果：(1) 消除空间冗余——75% 的信息缺失使得简单的插值完全不可行，模型被迫学习语义层面的重建；(2) 大幅降低 encoder 的计算量——encoder 只处理 25% 的 token，因此可以使用非常大的 ViT 作为 encoder，而不会导致训练成本失控。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 14：MAE 架构——输入图像 → patchify → 随机 mask 75% → encoder 仅处理可见 patch → decoder 接收 encoder 输出 + shared mask token → 重建完整图像</div>
</div>

MAE 架构是非对称的。**Encoder** 是 ViT-based 的大模型（如 ViT-Large），仅处理可见 patch：将每个可见 patch 通过线性投影嵌入为 token，加上 positional embedding 后送入 Transformer blocks。由于输入 token 数仅为原始的 25%，encoder 每个 token 的计算量是 decoder 的约 9 倍，这使得在合理计算预算内使用超大 encoder 成为可能。

**Decoder** 则是轻量级的 Transformer，仅用于预训练阶段（下游任务中会被完全丢弃）。它接收 encoder 对所有可见 patch 的输出，以及一个可学习的 **shared mask token**——所有被 mask 位置共享同一个可训练向量，大致可以理解为"平均 patch 的表征"。这些 token 按原始空间位置排列并加上 positional embedding，一起送入 decoder，最终通过一个线性投影层输出重建后的像素值。

### Training: MSE Loss on Masked Patches

MAE 的训练目标简洁明了：**仅在 masked patches 上计算 MSE 损失**，即比较重建像素与原始像素之间的均方误差。这与 Context Encoders 的损失设计一脉相承——我们只关心模型对被遮住部分的预测能力，不关心它对可见 patch 的"复制"能力。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 15：MAE 消融——随机 masking（最佳）vs 块状 masking vs 网格 masking</div>
</div>

MAE 论文中进行了大量的消融实验，涵盖了 masking ratio、decoder 深度与宽度、mask token 是否也送入 encoder、reconstruction target（像素值 vs 归一化像素值）、数据增强策略、mask sampling method 等多个维度。几个核心结论包括：(1) **75% masking ratio 最佳**——实验表明在这个比例下模型取得最高的下游微调精度；(2) **随机 masking 优于块状或网格 masking**——随机采样切断了所有可能被模型利用的空间规律，迫使它真正理解图像内容；(3) **decoder 的深度和宽度对下游性能影响不大**——证明 decoder 的设计相对灵活，关键在于 encoder。

### Linear Probing vs Fine-Tuning

SSL 预训练模型的评估遵循两种协议，分别探测不同的能力维度：

- **Linear Probing**：冻结 encoder 的所有参数，仅在顶层训练一个线性分类器。这直接衡量 encoder 学到的特征是否已经线性可分——特征越好，线性分类器的准确率越高。这是一个严格的测试，因为它不允许 encoder 根据下游任务做任何调整。
- **Fine-Tuning**：解冻 encoder（全部或部分层），与下游任务的头一起继续训练。这发挥了预训练模型的全部潜力，允许特征根据目标任务进一步优化。

MAE 的一个有趣特性是它在 fine-tuning 上表现卓越，但在 linear probing 上相对一般——这与对比学习方法形成鲜明对比。原因在于，MAE 的 encoder 学到的特征更多是关于图像内容的"潜编码"，不一定以线性可分的形式组织；但作为微调的起点，这些特征提供了极为丰富的语义信息，允许模型快速适应新任务。

### MAE vs DINO vs MoCo v3

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 16：MAE vs DINO vs MoCo v3 性能对比——MAE 在 fine-tuning 指标上优于对比方法</div>
</div>

在 ImageNet 上与当时最先进方法的比较中，MAE 在 fine-tuning 指标上优于 DINO 和 MoCo v3。此外，MAE 还有一个重要的训练效率优势：由于 encoder 只处理 25% 的 patch，每个 epoch 的计算量远低于需要同时处理完整图像的两个增强视图的对比学习方法。这种效率优势使得 MAE 成为许多视觉任务的首选预训练范式。

---

## Contrastive Representation Learning

### Core Idea and Formulation

与重建式方法（"预测被遮住的像素"）不同，对比学习采取了完全不同的哲学：不显式地重建数据，而是学习一个**表征空间**，使得语义相似的样本在该空间中靠近，不相似的样本远离。其核心假设是：对同一幅图像施加不同的数据增强（如不同的 crop、颜色扰动、模糊），得到的两个视图应当共享相同的语义身份；而不同图像的视图应当在表征空间中保持距离。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 17：对比学习核心概念——reference image x + positive sample x⁺（同图的不同增强）应当相互吸引；negative samples x⁻（不同图像）应当相互排斥</div>
</div>

形式化地，设 $x$ 为参考样本，$x^+$ 为正样本（同一图像的另一增强），$\{x_j^-\}$ 为一组负样本（batch 中的其他图像）。我们希望通过 encoder $f$ 学到一个表征，使得 scoring function $s(f(x), f(x^+))$ 尽可能大，而 $s(f(x), f(x_j^-))$ 尽可能小。由此引出 **InfoNCE Loss**（Information Noise Contrastive Estimation, van den Oord et al., 2018）：

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(f(x), f(x^+)) / \tau)}{\exp(s(f(x), f(x^+)) / \tau) + \sum_{j=1}^{N-1} \exp(s(f(x), f(x_j^-)) / \tau)}$$

其中 $\tau$ 是温度超参数，控制分布的集中程度。这个公式的结构非常熟悉——它正是 $N$-way softmax 交叉熵损失：在 $N$ 个候选样本（1 个正样本 + $N-1$ 个负样本）中，模型必须学会"找出"正确的正样本。换句话说，对比学习本质上将表征学习重新表述为**一个大规模的分类问题**。

这一损失具有重要的理论性质：InfoNCE 的负值是 $f(x)$ 与 $f(x^+)$ 之间**互信息**（mutual information）的下界。换言之，最小化 InfoNCE loss 等价于最大化两个增强视图之间的互信息——让它们共享尽可能多的信息。互信息下界的紧致程度依赖于负样本数 $N$：$N$ 越大，下界越紧。这就从理论上解释了为什么对比学习方法需要**大量负样本**——更多的负样本意味着更好的互信息估计和更高质量的表征。

### SimCLR: A Simple Framework

SimCLR（Simple Framework for Contrastive Learning of Visual Representations, Chen et al., 2020）将对比学习的思想简化到了极致。它的整个训练流程可以用几行伪代码描述：

1. 从 batch 中取出 $N$ 张图像
2. 对每张图像施加两次随机的数据增强，得到 $2N$ 个增强视图
3. 所有视图通过 encoder（如 ResNet）提取特征 $h$
4. 特征经过一个非线性 projection head $g(\cdot)$（2-3 层 MLP）映射到 $z$ 空间
5. 在 $z$ 空间中用 cosine similarity 作为 score function，计算 InfoNCE loss
6. 对于每对来自同一原始图像的视图，它们互为对方的正样本；batch 中其余 $2(N-1)$ 个视图均为负样本

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 18：SimCLR 框架——图像 → 两次随机增强 → encoder → projection head g(·) → cosine similarity → InfoNCE loss</div>
</div>

SimCLR 极度依赖**数据增强**来定义"什么是不变性"。典型的增强组合包括 random crop、color distortion（颜色抖动）和 random blur——这些变换被认为改变了图像的表面形式但不改变其语义身份。增强策略的选择实际上定义了模型学习什么类型的 invariances。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 19：SimCLR 线性分类器评估——冻结 encoder + 训练线性层，在 ImageNet 上匹配或超越监督学习的性能</div>
</div>

SimCLR 有两个关键设计选择值得深入讨论。

**Projection Head 的作用**。为什么不在 encoder 的输出 $h$ 上直接计算对比损失，而要额外引入一个 projection head $g(\cdot)$ 将特征映射到 $z$ 空间？直觉上，对比学习的目标是使表征对数据增强不变——但过度追求不变性可能会丢弃对下游任务有用的信息。例如，颜色抖动可能改变物体的颜色，但颜色本身对于某些下游任务（如识别成熟度的水果分类）可能很重要。通过将对比损失约束在 $z$ 空间，$h$ 空间可以保留那些被 $z$ 空间"过滤掉"的信息。实验表明，使用非线性 projection head 显著优于不使用或使用线性投影，且下游任务使用 $h$ 而非 $z$ 时性能更好——这正是因为 $h$ 保留了更丰富的特征。

**大 batch size 的必要性**。从 InfoNCE 的理论性质出发，更多的负样本意味着更紧的互信息下界和更好的表征质量。SimCLR 将 batch 中的所有其他视图都用作负样本，因此负样本数与 batch size 线性相关。这就导致了一个直接的张力：要获得好的表征，需要大 batch size（实验中最佳为 4096-8192）；但大 batch size 意味着巨大的 GPU 内存占用，通常需要 TPU pods 才能训练。这直接引出了下一步改进的动机。

### MoCo: Momentum Contrast

MoCo（Momentum Contrast, He et al., 2020）的核心创新在于**将负样本数从 batch size 中解耦**。它的关键观察是：负样本不需要与当前 batch 的梯度计算耦合在一起——我们只需要它们的表征即可。

MoCo 维护一个 FIFO queue 来存储来自之前 batch 的负样本表征。当前 batch 的样本作为 query（查询），queue 中累积的历史样本作为 key（键/负样本）。由于 queue 可以远大于 batch size（如 65536），MoCo 在不增加 batch size 的情况下就能拥有大量负样本。但这里产生了一个新问题：queue 中的负样本表征来自稍早时刻的 encoder，如果 encoder 更新太快，这些"旧"表征就会与当前 encoder 不一致（stale），导致对比信号失效。

MoCo 的解决方案是将 encoder 拆分为两个：**query encoder** $f_q$ 和 **key encoder** $f_k$。Query encoder 通过正常的梯度回传更新；key encoder 则通过**动量更新**：

$$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$$

其中 $m$ 是一个非常接近 1 的动量系数（如 0.999）。这意味着 key encoder 的演化极为缓慢，确保 queue 中存储的负样本表征在时间上具有一致性。同时，key encoder 不接收梯度（`no_grad`），因此 queue 中的负样本不参与反向传播。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 20：MoCo 架构——query encoder（梯度更新）+ key encoder（动量更新）+ FIFO queue 存储历史负样本</div>
</div>

这种设计在工程上极为巧妙：它既享受了大负样本池带来的表征质量提升，又避开了超大 batch 的内存开销，使得在普通 GPU 上训练高质量的对比学习模型成为可能。

### MoCo v2: Best of Both Worlds

MoCo v2（Chen et al., 2020）是一个非常务实的贡献：它将 SimCLR 中的两个关键设计——**非线性 projection head** 和**更强的数据增强**——移植到 MoCo 的 momentum queue 框架中。结果是一个"取两家之长"的混合方案。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 21：MoCo vs SimCLR vs MoCo v2 对比——MoCo v2 以 batch size 256 超越 SimCLR 的 batch size 8192，同时内存占用大幅降低</div>
</div>

对比实验清晰地展示了这一融合的效果：

<div align="center">

| 方法 | Batch Size | 负样本来源 | Projection Head | 强增强 | 内存占用 |
|------|-----------|-----------|----------------|--------|---------|
| **SimCLR** | 8192 | In-batch | ✓ | ✓ | 极大（需 TPU）|
| **MoCo v1** | 256 | Queue (65536) | ✗ | ✗ | 中等 |
| **MoCo v2** | 256 | Queue (65536) | ✓ | ✓ | 中等 |

</div>

MoCo v2 以 batch size 256 和远小于 SimCLR 的内存开销，达到了甚至超越了 SimCLR 以 batch size 8192 在 TPU 上取得的性能。这验证了一个重要结论：**非线性投影头和强数据增强对于对比学习至关重要**，而用 queue 解耦 batch size 与负样本数使得这些关键设计能够在有限的计算资源下充分发挥作用。

### Sequence Contrastive Learning: CPC

对比学习不限于图像实例层面。**Contrastive Predictive Coding**（CPC, van den Oord et al., 2018）将对比学习推广到序列数据，其核心思想可以从名字的三个词来理解：

- **Contrastive**：使用对比损失区分"正确"的序列延续和"错误"的序列延续；
- **Predictive**：模型需要根据当前上下文预测未来的模式；
- **Coding**：模型在此过程中学到有用的特征编码（code），用于下游任务。

CPC 的架构分为三步：(1) 一个 encoder $g_{\text{enc}}$ 将序列中的每个元素 $x_t$ 独立编码为向量 $z_t$；(2) 一个自回归模型 $g_{\text{ar}}$（原始论文使用 GRU-RNN）汇总截至当前时刻的所有 $z$ 信息，产生一个上下文编码 $c_t$；(3) 使用 InfoNCE loss 对上下文 $c_t$ 和未来时刻的编码 $z_{t+k}$ 进行对比，其中正样本是真实的未来元素，负样本是随机采样的其他元素。评分函数引入了一个可训练矩阵 $W_k$：$s_k(c_t, z_{t+k}) = c_t^T W_k z_{t+k}$，允许模型根据时间步长 $k$ 学习不同的相似度度量。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 22：CPC 框架——序列元素编码为 z_t → 自回归模型汇总为 context c_t → InfoNCE loss 对比正确与错误的未来元素</div>
</div>

CPC 在音频领域展现了强大的表征学习能力：在 LibriSpeech 数据集上，用 CPC 预训练的特征训练线性分类器，在音素分类任务上显著优于其他无监督方法。在视觉领域，CPC 可以通过将图像按行拆分为 patch 序列来应用——使用上半部分的 patch 行作为上下文，预测下半部分的行。然而，CPC 在图像表征学习上的效果不如实例级对比方法（SimCLR/MoCo），可能是因为将图像强行建模为 1D 序列丢失了 2D 空间结构。

### DINO: Self-Distillation with No Labels

DINO（Self-Distillation with No Labels, Caron et al., 2021）代表对比学习之外的第三种 SSL 范式——**自蒸馏**。它不使用负样本，而是采用 Teacher-Student 架构：两个网络处理同一图像的不同增强视图（与 SimCLR 类似，都经 multi-crop 策略产生多个 view），Student 网络的输出通过 softmax 转化为概率分布，Teacher 网络同样输出一个概率分布作为"目标"，二者的交叉熵被用作训练损失。关键区别在于——Teacher 不是独立训练的，而是 Student 参数的指数移动平均（EMA），即由 Student 缓慢演化而来。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 23：DINO 架构——Teacher-Student 自蒸馏，无需负样本，Teacher 由 Student 的 EMA 更新</div>
</div>

这种"学生向老师学习、老师由学生演化"的自蒸馏机制使得模型在没有负样本的情况下依然能学到有意义的视觉表征，并且展现出一些令人惊讶的涌现属性——例如，DINO 训练的 ViT 的 attention map 自然地呈现为物体的语义分割，无需任何分割标注。DINO v2 进一步将这一范式扩展到更大规模的数据和模型，产生了极具通用性的视觉特征。

从范式的角度看，DINO 与 MAE 和 SimCLR/MoCo 共同构成了现代自监督学习的三大支柱：**生成式**（重建像素）、**判别式/对比式**（区分正负样本）和**自蒸馏式**（学习教师的知识）。三种范式各有优势——MAE 在 fine-tuning 上表现最佳，SimCLR/MoCo 在 linear probing 上更强，而 DINO 以其涌现的语义分割能力展示了 SSL 表征可能蕴含远超我们预期的结构。

---

## Summary

Lecture 12 覆盖了自监督学习的完整技术图景，从最早的手工设计 pretext task 到现代 MAE 和对比学习范式：

**基于图像变换的 Pretext Tasks**：

| 方法 | Pretext Task | 核心机制 | 关键洞察 |
|------|-------------|---------|---------|
| **Rotation Prediction** | 预测旋转角度 (4-way) | Visual common sense 假设 | 注意力图更 holistic |
| **Jigsaw Puzzle** | 预测 patch 排列 (64-way) | Context-free 网络避免作弊 | 学习物体部件间的空间关系 |
| **Inpainting** | 重建 masked 区域 | L2 + Adversarial loss | Reconstruction 定结构，Adversarial 保真实感 |
| **Colorization** | 灰度 → 彩色 (L → A/B) | LAB 空间 + Cross-channel prediction | 歧义性迫使语义理解 |
| **Video Colorization** | 跨帧颜色传播 | Attention-based tracking | 物体跟踪作为 pretext 的涌现属性 |

**现代 SSL 三大范式**：

<div align="center">

| 范式 | 代表方法 | 核心思想 | 关键设计 | 主要局限 |
|------|---------|---------|---------|---------|
| **生成式（Reconstruction）** | MAE | 重建被 mask 的 patch | 75% 随机 mask + 非对称 encoder-decoder | Linear probing 不如对比方法 |
| **对比式（Contrastive）** | SimCLR | 吸引正样本，排斥负样本 | Projection head + 大 batch size | 需要大量负样本/大 batch |
| | MoCo v2 | 同上 + momentum queue | Queue 解耦 batch size 与负样本数 | 需维护 queue 和动量 encoder |
| **自蒸馏（Self-Distillation）** | DINO | Student 学习 Teacher（EMA）的 softmax 分布 | 无需负样本 | 需要 multi-crop 和 EMA teacher |

</div>

**核心要点**：

- SSL 的根本价值在于**去除标注瓶颈**——监督信号完全由数据本身自动生成
- Pretext task 的设计是 SSL 中最大的创造性空间：必须足够难以迫使语义理解，但又不能难到让模型学到无用的捷径
- **Linear probing**（衡量特征质量）和 **fine-tuning**（衡量迁移潜力）探测表征的不同侧面，两种评估缺一不可
- MAE 的高 masking 率（75%）和对比学习的多负样本看似不同，实则服务于同一目标——**消除捷径**，迫使模型进行语义层面的推理
- 从手工设计的 pretext task 到通用对比学习再到自蒸馏，SSL 的演化方向是**越来越通用、越来越不需要人工先验**

## Materials

- [Unsupervised Representation Learning by Predicting Image Rotations (Gidaris et al., ICLR 2018)](https://arxiv.org/abs/1803.07728)
- [Unsupervised Visual Representation Learning by Context Prediction (Doersch et al., ICCV 2015)](https://arxiv.org/abs/1505.05192)
- [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (Noroozi & Favaro, ECCV 2016)](https://arxiv.org/abs/1603.09246)
- [Context Encoders: Feature Learning by Inpainting (Pathak et al., CVPR 2016)](https://arxiv.org/abs/1604.07379)
- [Colorful Image Colorization (Zhang et al., ECCV 2016)](https://arxiv.org/abs/1603.08511)
- [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction (Zhang et al., CVPR 2017)](https://arxiv.org/abs/1611.09842)
- [Tracking Emerges by Colorizing Videos (Vondrick et al., ECCV 2018)](https://arxiv.org/abs/1806.09594)
- [Masked Autoencoders Are Scalable Vision Learners (He et al., CVPR 2022)](https://arxiv.org/abs/2111.06377)
- [A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., ICML 2020)](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning (He et al., CVPR 2020)](https://arxiv.org/abs/1911.05722)
- [Improved Baselines with Momentum Contrastive Learning (Chen et al., arXiv 2020)](https://arxiv.org/abs/2003.04297)
- [Representation Learning with Contrastive Predictive Coding (van den Oord et al., arXiv 2018)](https://arxiv.org/abs/1807.03748)
- [Emerging Properties in Self-Supervised Vision Transformers (Caron et al., ICCV 2021)](https://arxiv.org/abs/2104.14294)
- [CS231n 2025/2026 Lecture 12 Slides](https://cs231n.stanford.edu/slides/2025/lecture_12.pdf)
