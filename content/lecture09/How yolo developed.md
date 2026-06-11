---
title: How YOLO developed
publish: true
target: 解释 YOLO 诞生的原因，以问题导向分析 YOLO 的主要迭代过程
---
## 前置问题：2-Stage 检测器为什么需要一个"更好的范式"？

在 YOLO 出现之前，目标检测的主流范式是**两阶段（two-stage）**：先用某种方法生成候选区域（region proposals），再对每个候选区域做分类和边界框精修。R-CNN → Fast R-CNN → Faster R-CNN 正是沿着这条线索迭代——每个阶段都在优化"如何更快、更好地生成 proposals"。Faster R-CNN 把 proposal 生成也融入神经网络（RPN），已经做到了相当程度的端到端，但它本质上仍然是"先找区域、再分类"的两步走。

Redmon et al. (2016) 提出了一个激进的重新框架：

> *"We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes."*
> 我们将目标检测重新定义为一个单一的回归问题，直接从图像像素到边界框坐标和类别概率。一个卷积网络同时预测多个边界框及其类别概率。
> — Redmon et al., You Only Look Once: Unified, Real-Time Object Detection, CVPR 2016

核心区别：
- **Two-stage**：proposal → classify（解耦的两步，需要 NMS 等后处理）
- **YOLO**：image → bounding boxes + classes（一次前向，端到端）

这种重新定义不仅改变了网络架构，更改变了检测的哲学：**不是"扫描"图像找物体，而是"理解"整张图像后直接输出物体**。

> 注意：Lecture Note 中已经详细介绍了 R-CNN → Fast R-CNN → Faster R-CNN 的演进（Selective Search、RoI Pooling、RPN、Anchor、NMS），本篇笔记不再重复这些内容，直接从 YOLO 自身的发展展开。

---

## YOLOv1：统一检测范式的诞生

### 核心思想：网格化 + 一次前向

<div style="text-align: center;">
    <img src="Pasted image 20260608143757.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：YOLOv1 将图像划分为 S×S 网格，每个网格同时预测边界框和类别</div>
</div>

YOLOv1 将输入图像划分为 $S \times S$ 的网格（VOC 数据集上 $S=7$）。**每个网格单元负责预测落在该网格中心的物体**——这是 v1 最核心的设计约定。

对于每个网格单元，输出包含两部分信息：
- **$B$ 个边界框**（$B=2$）：每个框中包含 5 个值 $(x, y, w, h, \text{confidence})$
  - $(x, y)$ 是框中心相对于网格单元边界的偏移（归一化到 $[0, 1]$）
  - $(w, h)$ 是框的宽高相对于整张图像的比值（归一化到 $[0, 1]$）
  - $\text{confidence} = P(\text{object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$，反映"这个框中有物体"以及"框的位置多准确"的综合置信度
- **$C$ 个类别条件概率** $\text{Pr}(\text{Class}_i \mid \text{object})$：只在网格中有物体的前提下才有效

最终输出张量的维度为 $S \times S \times (5B + C)$。对 PASCAL VOC（$C=20$）：$7 \times 7 \times (5 \times 2 + 20) = 7 \times 7 \times 30$。

每个网格预测 $B$ 个框，但**一个网格只预测一个类别**——因为 $C$ 个类别概率是网格级别的，不随 $B$ 变化。这构成了 v1 的空间约束缺陷（后文详述）。

推理时的完整流程：
1. 图像输入 CNN，输出 $7 \times 7 \times 30$ 张量
2. 对每个网格的每个框：类特定置信度 = $\text{Pr}(\text{Class}_i \mid \text{object}) \times \text{confidence}$
3. 总共得到 $S \times S \times B = 7 \times 7 \times 2 = 98$ 个检测框
4. 设置阈值过滤低分框，应用 NMS 去重

---

### 网络结构

> *"Our network has 24 convolutional layers followed by 2 fully connected layers... The final layer predicts both class probabilities and bounding box coordinates."*
> 我们的网络有 24 个卷积层，后接 2 个全连接层……最后的输出层同时预测类别概率和边界框坐标。
> — Redmon et al., CVPR 2016

YOLOv1 的 backbone 受 GoogLeNet 启发，但有几点关键修改：

1. **用 $1\times1$ 降维层代替 GoogLeNet 的 Inception 模块**：在 $3\times3$ 卷积前加 $1\times1$ 卷积降低通道数，减少参数
2. **24 层卷积 + 2 层全连接**：卷积负责特征提取，全连接负责将特征映射到 $7\times7$ 网格上的预测
3. **输入分辨率 $448\times448$**：ImageNet 预训练使用 $224\times224$，检测时将分辨率翻倍以获得更细粒度的空间信息
4. **激活函数**：除最后一层用线性激活外，其余层使用 leaky ReLU（$\alpha=0.1$）

> *"Instead of the inception modules used by GoogLeNet, we simply use $1\times1$ reduction layers followed by $3\times3$ convolutional layers."*
> 我们不使用 GoogLeNet 的 Inception 模块，而是简单地使用 $1\times1$ 降维层后接 $3\times3$ 卷积层。
> — Redmon et al., CVPR 2016

为什么用全连接而不是全卷积？——这是 v1 的一个设计约束。全连接层要求固定的输入尺寸，所以 v1 无法处理任意大小的图像。这个限制在 v2 中被移除。

---

### 损失函数设计：多任务的平衡艺术

YOLOv1 同时优化三个任务——定位、置信度估计、分类——而它们的尺度差异巨大。如果直接加和，定位误差和分类误差的价值不等，且"无物体"的网格远多于"有物体"的网格，会导致信度损失被背景主导。

v1 的损失函数由五部分组成，每部分有不同的权重：

> *"Our loss function penalizes classification error only if an object is present in that grid cell. It also penalizes bounding box coordinate error only if that predictor is 'responsible' for the ground truth box."*
> 我们的损失只在网格中有物体时惩罚分类误差，也只在预测器对该真实框"负责"时才惩罚定位误差。
> — Redmon et al., CVPR 2016

#### 第一部分：坐标损失（有物体的框）

$$
\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]
$$

几点设计深意：
- **$\lambda_{\text{coord}} = 5$**：给定位误差更大的权重，因为"框的位置准不准"比"框里是什么"对最终评估的影响更大
- **$\mathbb{1}_{ij}^{\text{obj}}$**：只在第 $i$ 个网格的第 $j$ 个框"负责"该 GT 框时才计入（负责 = 该框与 GT 的 IoU 在所有 $B$ 个框中最高）
- **对 $w, h$ 取平方根**：这是 v1 损失函数中最微妙的设计——大框和小框对同样的绝对误差敏感度不同（大框偏移 5 像素几乎没影响，小框偏移 5 像素可能就完全错了），开根号抑制了大框损失的绝对值，使得大小框的定位误差被更公平地对待

#### 第二部分：置信度损失（有物体的框）

$$
\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2
$$

对"负责的"框，$\hat{C}_i = 1$，损失推动预测置信度趋近于 1。

#### 第三部分：置信度损失（无物体的框）——背景抑制

$$
\lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
$$

- **$\lambda_{\text{noobj}} = 0.5$**：关键设计！大多数网格没有物体，如果不降低权重，背景的置信度损失会淹没有物体的损失。$\lambda_{\text{noobj}}=0.5$ 给背景损失"打折"，让模型不因"大多数网格预测无物体就对"而自满

#### 第四部分：类别损失

$$
\sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

只对**有物体的网格**计算类别损失。没有物体的网格不需要预测类别。

> **关于"使用平方和而非交叉熵"：** v1 的损失函数全部使用**平方误差（sum-squared error）**，包括类别概率。论文承认这不是最优选择——平方误差假设输出服从高斯分布，而分类更适合交叉熵——但当时它简化了实现，且在实验中仍能工作。从 v2 开始，YOLO 切换为交叉熵做分类损失。

---

### YOLOv1 的优势与缺陷

#### 优势

> *"YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time, so it implicitly encodes contextual information about classes as well as their appearance."*
> YOLO 在预测时对整张图像进行全局推理。不像滑窗和基于区域提议的方法，YOLO 在训练和测试时看到整张图像，因此它隐式编码了关于类别及其外观的上下文信息。
> — Redmon et al., CVPR 2016

- **极快**：45 FPS（标准版），150 FPS（Fast YOLO），比 Faster R-CNN 快一个数量级
- **全局推理**：看整张图而非局部区域，背景误报显著少于 Fast R-CNN（R-CNN 容易将背景中的纹理误判为物体）
- **泛化能力强**：在艺术画作（Picasso 数据集、People-Art 数据集）上的表现远超其他检测器——因为 YOLO 学的是物体的整体表征，而不只是局部的纹理特征

#### 缺陷

1. **空间约束粗糙**：每个网格只预测一个类别（$C$ 个概率是网格级别的）。如果一个小网格内有两个物体属于不同类别（如"人和自行车"挤在一起），v1 只能识别其中一个——这是 $S=7$ 的粗粒度划分带来的根本瓶颈
2. **定位误差大**：直接回归绝对坐标对模型而言是很困难的任务，尤其是大小框的尺度差异
3. **不擅长奇怪宽高比**：训练数据中的典型宽高比主导了预测，对新奇形状/尺度的物体泛化差
4. **全连接层限制**：固定输入尺寸，无法处理任意大小的图像

这些问题定义了后续版本改进的方向：
- 缺陷 1 → v2/v3 提高网格密度 + 多尺度
- 缺陷 2 → v2 引入 Anchor
- 缺陷 3 → v2 Dimension Clusters + 多尺度训练
- 缺陷 4 → v2 移除全连接层

---

## YOLOv2 / YOLO9000：Anchor 化 + 多项提点策略

核心问题：v1 的定位差、召回低。

Redmon & Farhadi (2017) 几乎在所有可以改进的维度上都做了优化，而非集中在一个方向上。下面按"每一项带来了多少 mAP 提升"来组织。

> *"We made a bunch of little design changes to make YOLO better. Here's what we tried."*
> 我们做了一系列小型设计变更使 YOLO 变得更好。以下是我们尝试的内容。
> — Redmon & Farhadi, YOLO9000: Better, Faster, Stronger, CVPR 2017

### 核心改进

#### 1. Batch Normalization（+2.4% mAP）

在每个卷积层后添加 BN，移除 dropout（BN 自身有正则化效果）。这看似是常规操作，但在 2017 年的检测框架中，BN 还没有被普遍采用。

#### 2. 高分辨率分类器预训练（+4.0% mAP）

v1 用 $224\times224$ 的 ImageNet 预训练，然后直接切到 $448\times448$ 做检测——模型需要同时适应更高的分辨率和新的检测任务。v2 的做法：ImageNet 预训练完成后，**用 $448\times448$ 分辨率额外 fine-tune 10 个 epoch**，让特征提取器习惯高分辨率输入，再切换到检测任务。

#### 3. Anchor Boxes（-0.3% mAP，但召回 +7%）

这是 v2 最关键的架构变更：

> *"We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes... We use only convolutional layers because that allows the network to take images of any size as input."*
> 我们移除了 YOLO 的全连接层，使用锚框来预测边界框……我们只使用卷积层，因为这让网络可以接受任意尺寸的图像作为输入。
> — Redmon & Farhadi, CVPR 2017

变化细节：
- 移除全连接层，改用全卷积架构
- 不再直接预测 $(x, y, w, h)$，改为预测**相对 Anchor 的偏移**
- 输出分辨率从 $7\times7$ 提升到 $13\times13$
- 每个网格预测 5 个 Anchor（而非 v1 的 2 个框）
- 每个 Anchor 现在拥有独立的类别预测（而非 v1 的"一个网格只有一个类别"）——这直接修复了 v1 无法处理同一网格多类别的问题

虽然 mAP 微降了 0.3%，但召回率从 81% 提升到 88%——这意味着模型能找到更多物体，只是定位精度略降。后续改进会补上这个缺口。

#### 4. Dimension Clusters：K-Means 自动选定 Anchor 尺寸

> *"Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically find good priors."*
> 我们不手工选择先验框尺寸，而是在训练集的边界框上运行 K-means 聚类来自动找到好的先验。
> — Redmon & Farhadi, CVPR 2017

Faster R-CNN 的 Anchor 尺寸是手工设计的（3 种尺度 × 3 种宽高比 = 9 种）。v2 的洞见：**直接用训练数据中 GT 框的统计分布来设定 Anchor 尺寸**，比人工设计更贴合数据。

在聚类时使用 $\text{IoU}$ 距离而非欧氏距离（因为框的尺寸不应该被绝对大小主导），最终在 VOC 和 COCO 上聚类得到 5 组 Anchor（比 Faster R-CNN 的 9 组少——因为 YOLO 的 Anchor 更"精准"所以需要更少）。

#### 5. 直接坐标预测（+5.0% mAP）

Anchor 化的回归公式有一个常见问题——如果预测的偏移 $t_x$ 很大，框的中心可能从当前网格"漂移"到远处，导致训练不稳定。v2 的解决方案：限制偏移范围。

预测框的中心坐标为：

$$
\begin{aligned}
b_x &= \sigma(t_x) + c_x \\
b_y &= \sigma(t_y) + c_y \\
b_w &= p_w e^{t_w} \\
b_h &= p_h e^{t_h}
\end{aligned}
$$

其中 $(c_x, c_y)$ 是网格左上角的坐标，$(p_w, p_h)$ 是 Anchor 的宽高。$\sigma(t_x)$ 将偏移限制在 $[0, 1]$ 内——框的中心不会离开当前网格。这显著提升了训练的稳定性。

#### 6. 多尺度训练

每 10 个 batch 随机切换输入尺寸（从 $\{320, 352, \dots, 608\}$ 中抽取，步长 32）。这使得同一个模型在不同分辨率下都能工作——低分辨率更快（228×228 → 90 FPS），高分辨率更准（544×544 → 78.4 mAP）。

### 小结

| 改进项 | mAP 变化 |
|--------|---------|
| Baseline（v1） | 63.4 |
| + BN | +2.4 → 65.8 |
| + 高分辨率分类器 | +4.0 → 69.8 |
| + Anchor Boxes | -0.3 (但召回+7%) |
| + Dimension Clusters | +4.8 |
| + 直接坐标预测 | +5.0 |
| + 细粒度特征（passthrough, 类似残差）| +1.0 |
| + 多尺度训练 | +1.4 |
| **最终** | **78.6** (VOC 2007) |

---

## YOLOv3：多尺度 + 更深

核心问题：v2 对不同尺度物体的检测仍不够好——特别是在 COCO 这样多尺度物体丰富的数据集上。

> *"We made YOLOv3 a little bigger, a lot better, and about as fast... It's a half decade of incremental improvements."*
> 我们把 YOLOv3 做得大了一点、好了很多，速度差不多……这是五年增量改进的成果。
> — Redmon & Farhadi, YOLOv3: An Incremental Improvement, arXiv 2018

### 三尺度预测（FPN 雏形）

YOLOv3 在三个不同的特征图尺度上进行预测：

- **$13\times13$**（步长 32）：感受野最大，检测大物体
- **$26\times26$**（步长 16）：中等感受野，检测中等物体  
- **$52\times52$**（步长 8）：感受野最小，检测小物体

每个尺度使用 3 种 Anchor（共 9 种 Anchor，由 K-means 聚类得到），每个 Anchor 预测 $4+1+C$ 个值：

- 输出维度：$N \times N \times [3 \times (4 + 1 + C)]$

三尺度之间的信息流通是关键：深层特征经过上采样后与浅层特征**拼接（concatenate）**，然后继续卷积——上采样实现语义信息的传递，拼接保留了浅层的高分辨率细节。这就是后来被称为 **FPN（Feature Pyramid Network）** 的架构，但 v3 的实现以简洁著称，没有额外的横向连接或复杂的特征融合。

### Darknet-53

YOLOv3 设计了新的 backbone——Darknet-53：

> *"This new network is a hybrid approach between the network used in YOLOv2, Darknet-19, and that newfangled residual network stuff."*
> 这个新网络是 YOLOv2 的 Darknet-19 和那些新潮残差网络东西的混合体。
> — Redmon & Farhadi, arXiv 2018

- 53 层卷积（不含池化层，用步长 2 的卷积做下采样）
- 全局使用残差连接（ResNet 风格）
- 比 ResNet-101 更快，比 ResNet-152 准度相近

### 多标签分类

> *"We use independent logistic classifiers instead of softmax, because softmax imposes the assumption that each box has exactly one class which is often not the case."*
> 我们使用独立的逻辑分类器而不是 softmax，因为 softmax 强加了每个框只有一个类别的假设，这通常不符合实际情况。
> — Redmon & Farhadi, arXiv 2018

用 sigmoid + 二元交叉熵取代 softmax——"Person"和"Woman"可以同一框共存。这对 COCO 等含重叠标签的数据集特别有用。

### 速度与精度的平衡

YOLOv3 在 COCO 上达到 33.0 AP (at 320×320, 22ms) → 37.5 AP (at 608×608, 51ms)，显著快于同精度的 RetinaNet 和 SSD。用 Redmon 自己的话说："YOLOv3 is extremely good. Not as good as RetinaNet but fast." 他的自我评价带点调侃但精准——v3 是最均衡的检测器，没有单项最强但在"速度-精度"的 trade-off 上达到了当时的最优位置。

---

## YOLOv4：Bag of Freebies + Bag of Specials

核心问题：v3 之后如何**在几乎不增加推理成本的前提下提点**？

Bochkovskiy et al. (2020) 的回答是系统化的——把改进手段分为两类：

> *"We define Bag of Freebies as methods that only change the training strategy or only increase training cost without affecting inference time. Bag of Specials as plugin modules and post-processing methods that slightly increase inference cost but can significantly improve accuracy."*
> 我们定义 Bag of Freebies 为只改变训练策略、只增加训练成本而不影响推理时间的方法。Bag of Specials 为略微增加推理成本但能显著提升精度的插件模块和后处理方法。
> — Bochkovskiy et al., YOLOv4: Optimal Speed and Accuracy of Object Detection, arXiv 2020

这种分类框架本身就是一个贡献——它不仅告诉我们"用了什么"，更告诉我们"为什么这么组合"以及"trade-off 在哪里"。

### Bag of Freebies（训练时免费午餐）

| 方法 | 作用 |
|------|------|
| **Mosaic 数据增强** | 将 4 张图拼接为 1 张，丰富小物体上下文，减少 GPU 对 batch size 的需求 |
| **CIoU Loss** | 在 IoU Loss 基础上加入中心点距离和宽高比惩罚，比 L2 回归更贴合 mAP 评估 |
| **CmBN（Cross mini-Batch Normalization）** | 跨小 batch 收集统计量，在单 GPU 训练时也能获得类似大 batch 的 BN 效果 |
| **Self-Adversarial Training（SAT）** | 前向对图像做对抗扰动使模型"困惑"，反向再正常训练——增强鲁棒性 |
| **DropBlock 正则化** | 不是随机丢弃单个神经元而是丢弃连续的块区域——更适合空间密集的检测任务 |
| **Label Smoothing / Class label smoothing** | 软标签减少过拟合 |

### Bag of Specials（微增推理成本换提点）

| 方法 | 作用 |
|------|------|
| **CSPDarknet-53** | 在 Darknet-53 中引入 CSPNet（跨阶段部分连接）——减少计算量同时保持精度 |
| **PANet（Path Aggregation Network）** | 在 FPN 自顶向下后追加自底向上的路径，让浅层细节更快传递到深层 |
| **Mish 激活函数** | $Mish(x) = x \cdot \tanh(\ln(1+e^x))$——比 ReLU 更平滑，带来轻微但稳定的精度提升 |
| **SAM（Spatial Attention Module）** | 从空间维度做注意力，不显著增加计算量 |

v4 的核心方法论是**组合优于单个**——几十个改进的叠加效果大于各自之和。最终在 COCO 上达到 43.5% AP (Tesla V100, ~65 FPS)，同等速度下比 EfficientDet 更准。

---

## 后续关键演进：面向工程的成熟化

Redmon 在 v3 后宣布退出计算机视觉研究（部分因对检测技术的军事应用表示不安）。此后 YOLO 的发展由 Ultralytics（Glenn Jocher 等）和其他社区成员接手，发展方向从"学术创新"转向"工业部署友好"。

### YOLOv5：工业级标准化流水线

Ultralytics 发布的 YOLOv5 并不是论文产出，而是一个**工程化框架**。它的关键贡献：
- 完整的 PyTorch 实现 + 训练/验证/部署脚本
- 自动 Anchor 计算（训练前自动在数据集上运行 K-means）
- 模型缩放（n/s/m/l/x 五个等级，按 depth/width multiplier 系统缩放）
- 与 ONNX、TensorRT、CoreML 的原生集成

### Anchor-free 的转向（YOLOv8）

到了 YOLOv8，YOLO 经历了一场深刻的范式转换——**从 Anchor-based 转向 Anchor-free**。这个转向的背景是 FCOS (Tian et al., ICCV 2019) 等工作的成功——证明了 Anchor-free 可以达到甚至超越 Anchor-based 的精度，同时：
- 消除了手工设计 Anchor 的超参
- 减少了每个位置的候选框数量（不再需要多个 Anchor）
- 简化了 NMS 前的后处理逻辑

YOLOv8 的具体设计：用 Anchor-free head 直接预测框的四个边到中心点的距离，配合 Decoupled Head（分类和回归分支分离）+ C2f 模块（比 C3 更轻量的特征提取），形成了一个更干净、更快的流水线。

---

## 总结

全维度对比：

| 版本 | 年份 | 关键创新 | 核心解决问题 | Backbone | 检测范式 | 代表速度 (VOC) |
|------|------|---------|-------------|----------|---------|----------|
| **YOLOv1** | 2016 | 统一回归范式 | 2-stage 太慢 | GoogLeNet-style 24层 | Grid + 直接回归 | 45 FPS |
| **YOLOv2** | 2017 | Anchor + BN + 多尺度训练 | v1 定位差/召回低 | Darknet-19 | Anchor-based | 67 FPS |
| **YOLOv3** | 2018 | FPN 三尺度 + Darknet-53 | v2 多尺度不足 | Darknet-53 | Anchor-based + FPN | 35 FPS |
| **YOLOv4** | 2020 | Bag of Freebies + Specials | 系统性提点（不掉速） | CSPDarknet-53 | Anchor-based + PANet | ~50 FPS |
| **v5/v8** | 2020– | 工业级标准化 + Anchor-free | 部署友好 + 消 Anchor 超参 | 多级缩放 | Anchor-free | 极快 |

### 核心理由因果链

> 2-stage 解耦范式（proposal → classify）存在速度瓶颈 →
> **YOLOv1** 统一回归，一次前向输出检测结果，但定位差、每个网格仅一个类别 →
> **YOLOv2** 引入 Anchor（提升召回）、移除全连接层、多尺度训练（提升定位） →
> **YOLOv3** 三尺度 FPN 处理不同大小物体，Darknet-53 加深 → 速度-精度最优平衡点 →
> **YOLOv4** 系统化改进手段分类（Freebies vs Specials），组合优于单个 →
> **v5/v8** 工业成熟化，Anchor-free 转向 → YOLO 从学术概念跃迁为工业标准流水线

---

## 参考文献

- [YOLOv1 (Redmon et al., CVPR 2016)](https://arxiv.org/abs/1506.02640)：将目标检测重新定义为统一回归问题，提出 S×S 网格 + 端到端架构。
- [YOLOv2 / YOLO9000 (Redmon & Farhadi, CVPR 2017)](https://arxiv.org/abs/1612.08242)：引入 Anchor Box、Dimension Clusters、多尺度训练等，全面改善 v1 的定位和召回。
- [YOLOv3 (Redmon & Farhadi, arXiv 2018)](https://arxiv.org/abs/1804.02767)：三尺度预测 + Darknet-53 + 多标签分类，确立"速度-精度最优平衡"的检测器定位。
- [YOLOv4 (Bochkovskiy et al., arXiv 2020)](https://arxiv.org/abs/2004.10934)：系统化分类 Bag of Freebies/Bag of Specials，组合优化达到速度-精度新高度。
- [FCOS (Tian et al., ICCV 2019)](https://arxiv.org/abs/1904.01355)：证明 Anchor-free 检测器可达到甚至超越 Anchor-based 精度，影响了 YOLOv8 的 Anchor-free 转向。
- [Ultralytics YOLOv5 / YOLOv8](https://github.com/ultralytics/ultralytics)：工业级 YOLO 实现，标准化训练/部署/模型缩放流水线。