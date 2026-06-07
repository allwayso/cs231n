---
title: "The loss function for semantic segmentation"
publish: true
target: "从逐像素交叉熵的朴素起点出发，沿类别不平衡、区域评估、边界精炼三条线索，梳理语义分割损失函数从 CE 到 Dice、Focal、Lovász-Softmax、Boundary Loss 的演进逻辑"
---
## 一个每像素都是多分类任务的问题

语义分割的输出来看：模型对每个像素独立输出一个 $C$ 维的类别概率向量（$C$ 为类别数）。把这个结构摊开——$H \times W$ 个像素，每个像素做一个 $C$ 类 softmax 分类——最直接的想法自然出现：**对每个像素算交叉熵（Cross-Entropy），然后全部加起来**。

这就是逐像素交叉熵（Pixel-wise Cross-Entropy），也是 Long et al. (2015) 在提出 FCN 时使用的损失：

> *"We define a per-pixel multinomial logistic loss... The loss is summed over all pixels in a mini-batch, with each pixel's contribution weighted equally."*
> 我们定义了一个逐像素的多项逻辑损失……损失在一个 mini-batch 的所有像素上求和，每个像素的贡献权重相等。
> — Long et al., Fully Convolutional Networks for Semantic Segmentation, CVPR 2015

$$
L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{H \times W} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}
$$

其中 $y_{i,c} \in \{0, 1\}$ 是像素 $i$ 在类别 $c$ 上的 one-hot 标签，$\hat{y}_{i,c}$ 是模型预测的概率。

这个损失函数胜在简单、可微、端到端可训练。但它有一个隐含假设：**每个像素同等重要**。现实中，这个假设在两个维度上被严重打破——有些像素太多了（类别不平衡），有些像素被错误分类的代价和其他像素不一样（边界像素 vs 中心像素）。实际上，即使给像素分配了合适的权重，得到了较小的损失值，也不一定说明损失函数适配了语义分割的实际评价标准——区域重叠程度。

下面的演进就沿着 **类别不平衡** 、 **区域重叠程度** 和 **边界像素的歧义性** 三条线索展开。

---

## 线索一：类别不平衡——多数类淹没了少数类

### 问题：背景像素主导梯度

在典型的语义分割场景中，背景（或"空"类）往往占据图像的大部分区域，前景物体（如行人、车辆、器官）只占少数像素。逐像素交叉熵对所有像素一视同仁，导致梯度被背景像素主导——模型学到一个"把所有像素预测为背景"的捷径也能拿到很低的总损失。

### Weighted Cross-Entropy：给少数类加权

最朴素的补救：按类别频率给损失加权，少数类权重更高：

$$
L_{\text{wCE}} = -\frac{1}{N} \sum_{i=1}^{H \times W} w_{y_i} \log \hat{y}_{i, y_i}
$$

其中 $w_c$ 通常设为该类别频率的倒数或平方根倒数。FCN 论文中已经使用了这种策略——对背景给低权重，对前景物体给高权重。

问题是：**权重是固定超参**，需要手动设置，且不能适应训练过程中难易样本分布的变化。

### OHEM：在线挖掘难例

Shrivastava et al. (2016) 提出了 Online Hard Example Mining（OHEM），一个更自适应的思路：不是给所有像素加权，而是**只对损失最大的那些像素回传梯度**。

> *"Rather than using all RoIs for backpropagation, OHEM selects only the hardest examples—those with the highest loss—for training. This focuses learning on the most informative regions."*
> OHEM 并不用所有 RoI 进行反向传播，而是只选择最难的样本——损失最高的那些——来训练。这让学习聚焦在最有信息量的区域上。
> — Shrivastava et al., Training Region-based Object Detectors with Online Hard Example Mining, CVPR 2016

OHEM 原为检测设计，但"难例挖掘"的思想可以直接迁移到分割：每次前向传播后，按像素损失排序，仅对 top-$K$ 个像素回传梯度。优势在于自适应——随着训练推进，"难"的定义自动变化；劣势在于引入了超参 $K$，且丢弃了"容易"像素的信息。

### Focal Loss：软化的难例挖掘

Lin et al. (2017) 提出的 Focal Loss 给出了一个更优雅的方案——不给硬阈值，而是让损失函数**自动降低"已分类良好"像素的权重**：

> *"Focal Loss is designed to address the extreme foreground-background class imbalance... It down-weights the loss assigned to well-classified examples, focusing training on a sparse set of hard examples."*
> Focal Loss 被设计来应对极端的前景-背景类别不平衡……它降低已分类良好的样本的损失权重，将训练聚焦在稀疏的难例集合上。
> — Lin et al., Focal Loss for Dense Object Detection, ICCV 2017

$$
L_{\text{Focal}} = -\alpha_t (1 - \hat{p}_t)^\gamma \log \hat{p}_t
$$

其中 $\hat{p}_t$ 是模型对正确类别的预测概率，$\gamma \geq 0$ 调节"软化"程度：
- $\gamma = 0$ 退化为普通 CE
- $\gamma = 2$（典型值）时，$\hat{p}_t = 0.9$ 的像素损失被降到原来的 $1\%$，而 $\hat{p}_t = 0.1$ 的像素损失基本不变

Focal Loss 原本为 RetinaNet（目标检测）设计，但在语义分割中同样有效——特别是前景像素极少（如小目标、细线）的场景。它和 weighted CE 可以叠加使用（$\alpha_t$ 负责类别级平衡，$\gamma$ 负责难易平衡），形成了一个完整的"像素级权重分配"方案。

---

## 线索二：逐像素评估 ≠ 区域分割质量

### 问题：CE 的优化目标和评估指标脱节

语义分割最常用的评估指标是 **IoU / mIoU** 和 **Dice coefficient**——它们衡量的是**区域级别的重叠**，不是逐像素正确率。

#### IoU 与 mIoU

**IoU（Intersection over Union，交并比）**，也称 Jaccard 指数，衡量两个集合的重叠程度。对于类别 $c$，设预测为正类的像素集合为 $\mathcal{P}_c$，真值像素集合为 $\mathcal{G}_c$：

$$
\text{IoU}_c = \frac{|\mathcal{P}_c \cap \mathcal{G}_c|}{|\mathcal{P}_c \cup \mathcal{G}_c|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

直观理解：预测区域和真值区域的交集面积，除以它们的并集面积——完全重叠时为 1，完全不重叠时为 0。一个极端例子：一张图中有一个小物体，模型把它完全预测错了（IoU = 0），但因为这个物体只占 1% 的像素，CE 损失仍然很低——**IoU 把这个问题暴露出来了，但 CE 没有**。

**mIoU（mean IoU）** 则是对所有 $C$ 个类别分别算 IoU 后取平均：

$$
\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \text{IoU}_c
$$

#### Dice Coefficient

**Dice 系数**（也称 F1-score）是 IoU 的近亲，定义为：

$$
\text{Dice} = \frac{2 |\mathcal{P} \cap \mathcal{G}|}{|\mathcal{P}| + |\mathcal{G}|} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}
$$

Dice 和 IoU 之间存在单调映射关系：$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$。

在实践中的区别：
- Dice 对交集的变化更敏感（分子系数为 2），容易被用作训练损失（Dice Loss）
- IoU 更直观（"交并比"的物理含义清晰），通常作为最终上报指标
- 两者高度正相关，Dice Loss 优化好了，mIoU 一般也会提升

这个脱节——CE 优化像素精度，评估却看区域重叠——引出了一个核心问题：**能否直接优化分割评估指标？**

### Dice Loss：直接优化重叠率

Milletari et al. (2016) 在 V-Net（3D 医学影像分割）中提出了 Dice Loss，将 Dice coefficient 转化为可微的损失函数：

> *"The Dice coefficient is a measure of overlap between two samples... We formulate the objective function as the Dice Loss, which is directly derived from the Dice coefficient and is fully differentiable."*
> Dice 系数是衡量两个样本之间重叠的指标……我们将目标函数公式化为 Dice Loss，它直接从 Dice 系数导出，并且完全可微。
> — Milletari et al., V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, 3DV 2016

二类情形下的 Dice Loss：

$$
L_{\text{Dice}} = 1 - \frac{2 \sum_i y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon}
$$

对于多类分割，对每个类别独立计算 Dice Loss 后取平均。

Dice Loss 的核心优势：
1. **天然抗类别不平衡**：Dice 是比值（交集比上面积和），与前景像素的绝对数量无关
2. **优化目标与评估指标一致**：直接优化重叠率，训练损失下降意味着 mIoU 在上升（虽不完全等价，但强相关）
3. **在医学影像分割中成为事实标准**：从 nnU-Net (Isensee et al., 2021) 到 MONAI，Dice 是默认组件

缺点：对小目标训练不稳定——前景像素太少，一个像素的错误预测就能导致 Dice 大幅波动，梯度方差大。实践中通常与 CE 组合使用。

### Lovász-Softmax：直接优化 mIoU 的理论保证

Berman et al. (2018) 更进一步——他们直接问：**能否对 IoU / mIoU 做梯度下降？**

答案是可以，但需要一些数学工具。核心思路：Jaccard 指数（即 IoU）在预测分数上是**子模函数（submodular function）**，而子模函数可以通过 **Lovász 扩展** 构造凸代理（convex surrogate），从而获得光滑的梯度。

> *"We propose a loss function for semantic image segmentation that directly optimizes the mean Intersection-over-Union (mIoU) metric... Based on the Lovász extension of submodular set functions, we derive a tractable surrogate for the Jaccard loss."*
> 我们提出了一个直接优化平均 IoU 指标的语义图像分割损失函数……基于子模集合函数的 Lovász 扩展，我们为 Jaccard 损失导出了一个可处理的代理函数。
> — Berman et al., The Lovász-Softmax: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks, CVPR 2018

数学原理（简要）：
1. 定义集合函数 $\Delta_{J_c}(y^*, \tilde{y})$ —— 类别 $c$ 上的 Jaccard 损失（= $1 - \text{IoU}$）
2. 证明 $\Delta_{J_c}$ 是子模函数
3. 对其做 Lovász 扩展，得到一个分段线性凸函数 $\bar{\Delta}_{J_c}$，可直接优化

Lovász-Softmax 是目前语义分割中唯一有理论保证"直接优化 mIoU"的损失函数。实验表明它在多类分割中系统性地好于 CE 和 Dice，尤其在小目标类别上提升明显。代价是实现更复杂（需要排序操作），计算开销略高于 Dice。

---

## 线索三：边界像素的歧义性

分割的边界区域天然存在歧义——像素落在两个物体的交界处，标注本身也可能有误差。CE 和 Dice 都同等对待中心像素和边界像素，而边界像素的难度和不确定性远高于中心像素。

### Boundary Loss：显式建模轮廓距离

Kervadec et al. (2019) 提出了 Boundary Loss，将分割的边界信息编码为损失函数：

> *"We propose a boundary loss that takes the form of a distance metric on the space of contours, not regions. This loss complements regional losses like Dice, and is particularly effective when the boundaries are poorly defined."*
> 我们提出了一个边界损失，它采取轮廓空间上的距离度量形式，而非区域。这个损失是 Dice 等区域损失的补充，在边界定义不清时特别有效。
> — Kervadec et al., Boundary Loss for Highly Unbalanced Segmentation, MIDL 2019

核心思想：对每个前景像素计算其到真值轮廓的**符号距离**，损失惩罚那些预测为前景但距离真值轮廓远的像素（反之亦然）。Boundary Loss 不是替代 CE 或 Dice，而是作为它们的**补充**——通常以加权组合的方式加入。

在实践中，Boundary Loss 的权重通常从低到高逐步增加（类似退火），避免训练早期边界信息过于嘈杂。

---

## 多视角综合

### 视角 1：Dice Loss 的实验证据最强（Milletari 2016 → Isensee 2021）

说 Dice Loss 是语义分割（尤其在医学影像）的事实标准，不是因为它数学上最漂亮，而是因为它**简单、有效、经过大规模验证**。nnU-Net 的默认配置就是 CE + Dice 的组合（Isensee et al., 2021），且 nnU-Net 在几十个医学分割基准上取得了 SOTA——这本身就构成了最有力的经验证据。对于绝大多数语义分割场景，CE + Dice 已经足够好。

### 视角 2：Focal Loss 架起了分类-分割的桥梁（Lin 2017）

Focal Loss 虽诞生于检测社区，但其"软化的难例权重分配"思想深刻影响了分割损失设计。它的核心洞见——**类别不平衡和难易不平衡是正交的两个问题，需要用 $\alpha$ 和 $\gamma$ 分别处理**——构成了一个完整的"像素级权重分配"理论框架。即使是 Dice Loss，其抵抗不平衡的能力本质上也是在做一个隐式的动态权重调整。

### 视角 3：Lovász-Softmax 的数学优雅性与实际收敛速度（Berman 2018）

Lovász-Softmax 是唯一一个"理论保证优化 mIoU"的损失函数，其数学推导非常干净——子模性 → Lovász 扩展 → 凸代理。但争议点在于：**优化更准确的代理是否意味着更好的实际性能？** 实验表明 Lovász-Softmax 在许多任务上确实优于 Dice，但"优于 Dice"的幅度并不如"Dice 优于 CE"那么大。考虑到实现和维护成本，Lovász-Softmax 更适合对 mIoU 有极致要求、且愿意为几分提升付出工程复杂度的场景。

---

## 现代实践：组合优于单一

现实中几乎没有只用一种损失函数的语义分割模型。以下是经过大规模验证的典型组合：

| 组合 | 适用场景 | 代表工作 |
|------|---------|---------|
| **CE + Dice** | 通用分割，最普遍 | nnU-Net (Isensee 2021), V-Net (Milletari 2016) |
| **Focal + Dice** | 极端不平衡（前景<1%） | 小目标分割、卫星影像 |
| **CE + Lovász** | 对 mIoU 极致要求 | Berman 2018 原论文 |
| **CE + Dice + Boundary** | 医学影像，边界精度关键 | Kervadec 2019 原论文 |

nnU-Net 的损失设计策略尤其值得注意：它自动根据数据集特征（类别数、样本量、前景占比）选择损失组合，而不是由研究者手动调参。这种"自动配置损失"的思想折射出一个深刻共识：**没有最好的损失函数，只有最适合数据特征的损失组合**。

---

## 总结

语义分割损失函数的演进沿着三条线索展开：

| 损失函数 | 优化粒度 | 处理不平衡 | 训练稳定性 | 计算开销 | 代表论文 |
|---------|---------|-----------|-----------|---------|---------|
| **Pixel-wise CE** | 像素 | 无（需加权） | ⭐⭐⭐⭐⭐ | ⭐ | FCN (Long 2015) |
| **Weighted CE** | 像素 | 固定权重 | ⭐⭐⭐⭐ | ⭐ | FCN (Long 2015) |
| **Focal Loss** | 像素 | 自适应软权重 | ⭐⭐⭐⭐ | ⭐ | Lin 2017 |
| **Dice Loss** | 区域 | 天然（比值） | ⭐⭐⭐ | ⭐⭐ | Milletari 2016 |
| **Lovász-Softmax** | 区域 | 天然 | ⭐⭐⭐ | ⭐⭐⭐ | Berman 2018 |
| **Boundary Loss** | 边界 | 需组合使用 | ⭐⭐ | ⭐⭐ | Kervadec 2019 |

核心理由因果链：

> 逐像素 CE 让每个像素独立分类 → 类别不平衡：少数类被多数类淹没 → Weighted CE / Focal Loss 引入像素级权重分配 → 评估指标脱节：mIoU 看区域，CE 看像素 → Dice Loss / Lovász-Softmax 直接优化重叠率 → 边界歧义：中心像素和边界像素不应同权 → Boundary Loss 补充轮廓信息 → **最终实践：所有损失加权组合，根据数据特征自动或手动配置**

---

## 参考文献

- [FCN (Long et al., CVPR 2015)](https://arxiv.org/abs/1411.4038)：提出逐像素多类 softmax 损失，确立语义分割损失函数的基准范式。
- [OHEM (Shrivastava et al., CVPR 2016)](https://arxiv.org/abs/1604.03540)：提出在线难例挖掘，通过自适应选择高损失样本回传梯度来应对不平衡——虽为检测提出，但其"难例关注"思想深刻影响了分割损失设计。
- [Focal Loss (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002)：提出软化的难例权重分配，用 $\alpha$ 处理类别不平衡、$\gamma$ 处理难易不平衡，构成像素级权重分配的完整框架。
- [V-Net / Dice Loss (Milletari et al., 3DV 2016)](https://arxiv.org/abs/1606.04797)：直接从 Dice 系数导出可微损失，天然抗不平衡——成为医学影像分割的事实标准。
- [Lovász-Softmax (Berman et al., CVPR 2018)](https://arxiv.org/abs/1705.08790)：通过子模函数的 Lovász 扩展，为 mIoU 提供可处理的凸代理——唯一有理论保证直接优化 mIoU 的损失函数。
- [Boundary Loss (Kervadec et al., MIDL 2019)](https://arxiv.org/abs/1812.07032)：将轮廓距离编码为损失，作为区域损失的补充——特别在边界不清晰时有效。
- [nnU-Net (Isensee et al., Nature Methods 2021)](https://arxiv.org/abs/1809.10486)：CE + Dice 组合是 nnU-Net 自动管线的一部分，在数十个基准上验证了其作为通用默认的可靠性。