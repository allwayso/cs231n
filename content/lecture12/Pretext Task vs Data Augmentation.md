---
title: "Pretext Task vs Data Augmentation"
publish: true
target: 辨析自监督前置任务与数据增强的深层联系与本质区别，兼论 NLP 中两种 mask 的不同角色
---

## 共同的根基：语义不变性假设

在讨论 Pretext Task 与 Data Augmentation 的关系之前，有必要先回答一个更基本的问题：**为什么它们看上去如此相似？**

答案在于，两者共享同一个核心前提——**对图像施加某些低层变换，不会改变它的高层语义内容**。

- 一只狗旋转 90° 还是那只狗 → Rotation Prediction（旋转预测）和旋转变换增强都依赖这个事实
- 一只狗调整了颜色还是那只狗 → Colorization（着色）pretext 和颜色抖动增强都以此为出发点
- 同一只狗的不同裁剪区域都包含"狗" → Jigsaw Puzzle（拼图）pretext 和 Random Crop 增强的合法性都基于这条假设
- 同一张图的不同增强视图仍然是同一个语义对象 → 这正是 SimCLR 等对比学习方法运作的前提

这不是巧合。Data Augmentation 和 SSL Pretext Task 本质上都在利用"**变换不改语义**"这条先验知识——区别在于**利用的方式和程度**。

---

## 核心区别：被动适应 vs. 主动求解

虽然根基相同，但两条路径从"同一个出发点"走向了"完全不同的方向"。

<div align="center">

| 维度 | Data Augmentation | SSL Pretext Task |
|------|------------------|------------------|
| **变换的角色** | 输入层面的扰动手段 | **训练目标本身** |
| **监督信号来源** | 人工标注的类别标签 | 变换参数 / 原始数据（自动生成） |
| **模型的任务** | "不管你怎么变，标签不变，所以我要忽略这些变化" | "你变了什么？把它还原 / 识别出来" |
| **不变性如何获得** | 被动——作为分类任务的**副作用** | 主动——求解变换过程**本身就是**迫使语义理解的机制 |
| **范式定位** | 正则化手段，辅助监督学习 | 独立训练范式，可**完全替代**标签 |
| **变换强度要求** | 相对温和（轻微旋转、裁剪、颜色抖动），保证"标签仍然正确" | 可以极端（MAE mask 75%、完全灰度化），因为标签由数据自动提供 |
| **模型输出** | 类别概率 | 变换角度 / 打乱排列 / 缺失像素 / 原始颜色 |

</div>

用一个比喻来理解这种区别：

- **Data Augmentation** 像是老师把一个单词用不同字体、不同大小、不同角度写出来，但每次都告诉你"这还是 `cat`"——你学会了**容忍书写变化**，但你的核心任务仍然是记住 `cat` 这个标签。
- **SSL Pretext Task** 像是老师把单词的一部分遮住、或者把字母顺序打乱，然后问你"原词是什么？"—你为了答对这道题，**必须真的理解单词的结构和含义**，而不是简单地匹配已知标签。

前者的不变性是**副产品**（byproduct），后者把 "找到不变性" 直接做成了**训练目标本身**。这就是为什么 SSL 可以完全脱离人工标注——它的监督信号来自数据内部的结构，而不是外部的类别名称。

这也解释了为什么单纯的数据增强不能取代 SSL。数据增强只是让监督学习的分类器更鲁棒，但**如果没有标签，增强本身提供不了任何学习信号**；而 SSL 的 pretext task 本身就构成了一个完整的监督回路——输入是变换后的图像，标签是变换参数或原始数据，二者皆来自数据本身。

---

## 最紧密的交汇：对比学习

如果 Data Augmentation 和 Pretext Task 之间存在一个"合流点"，那一定是**对比学习**（Contrastive Learning）。

在 SimCLR 的框架下，两者的界限直接消失了：

1. 从同一张图像出发，施加两次**随机数据增强**（crop + color jitter + blur），得到两个视图 $x_i$ 和 $x_j$
2. 这两个视图通过 encoder 和 projection head 映射到 $z$ 空间
3. 使用 InfoNCE loss 让 $z_i$ 和 $z_j$ 在表征空间中相互靠近，同时与其他图像的视图远离

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(s(z_i, z_k) / \tau)}
$$

这意味着什么？**数据增强策略本身就是 pretext task 的定义**。不需要旋转、不需要拼图、不需要着色——pretext task 就是"同一张图的不同增强版本应该在特征空间中是邻居"。增强策略的选择直接定义了模型要学习什么样的不变性：

- Random Crop → 空间构成的不变性
- Color Jitter → 颜色分布的不变性
- Gaussian Blur → 纹理细节的不变性
- 组合使用 → 多维度语义不变性

所以在 SimCLR 中，那句经典问题——"它们是同一个硬币的两面吗？"——有了最直接的答案：**数据增强定义了变换空间，对比损失驱动模型在这个变换空间中学到不变的表征。两者合二为一。**

然而，即使在这一交汇点上，关键差异仍然存在。对比学习中的增强比监督学习中的增强**更激进**：SimCLR 使用的 color jitter 强度远高于标准 ImageNet 训练的配置——因为在对比学习中，过强的增强不会"破坏标签"（没有标签可破坏），只会让 pretext task 更难，而更难的 pretext task 恰恰迫使模型学到更强的语义表征。这是 SSL 独有的自由度：**变换的设计空间不再受"标签是否仍然有效"的约束。**

---

## NLP 视角：两种 Mask，两个世界

上一个问题——Pretext Task 与 Data Augmentation 的关系——在 NLP 领域有一个对应的"镜像问题"：**Masked Self-Attention 和 Masking as Pretext（如 BERT 的 `[MASK]` token）有什么区别？** 这同样是两种"mask"在"同一个词"下的混淆，而且由于 Transformer 架构的普遍性，视觉和语言两边的概念往往被放在一起讨论，更容易混为一谈。

### Masked Self-Attention：架构层面的因果约束

这是 [[lecture08 Attention and Transformers#Masked Self-Attention|Lecture 08]] 中讨论的概念。它作用于**注意力权重矩阵** $E$ 上——在计算 softmax 之前，将 $j > i$（未来位置）的相似度设为 $-\infty$，使注意力权重变为 0。

$$
E_{i,j} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{D}}, & j \leq i \\ -\infty, & j > i \end{cases}
$$

它的目的是保证**自回归生成的因果性**：第 $i$ 个 token 在预测第 $i+1$ 个 token 时不能"偷看"后面的内容。这是一个**架构层面的约束**——它定义的是"模型内部的信息流规则"，和训练任务本身无关。所有自回归语言模型（GPT 系列）都需要它。

### Masking as Pretext：训练任务层面的信号设计

这是 [[lecture12 Self-supervised Learning#Masked Autoencoders (MAE)|Lecture 12]] 中 MAE/BERT 的核心思想。它作用于**输入数据**——随机选择一些 input token，把它们替换为特殊的 `[MASK]` token，然后让模型预测这些被遮住位置上的原始 token。

它的目的是**构造自监督训练信号**：模型为了补全被 mask 的词，必须理解上下文语义、句法结构和词与词之间的关系。这是一个**训练目标层面的设计**——它定义的是"模型要学什么"。

### 为什么 NLP 中这两个概念更容易混淆？

一个重要的原因是：**BERT 同时涉及了两种 mask**，而且它们的"有无"恰好构成了一组对照。

- **GPT**：有 Masked Self-Attention（因果遮罩，不能看未来），无输入 mask → next-token prediction 作为 pretext
- **BERT**：无 Masked Self-Attention（双向 attention，每个 token 都能看到所有其他 token），**有**输入 mask → masked language modeling 作为 pretext

BERT 之所以去掉因果 mask，是因为 masked language modeling 需要**双向上下文**来预测被遮住的词——如果只能看左边，"The `[MASK]` is barking" 就无法利用 "barking" 的信息来猜出 "dog"。而 GPT 之所以必须保留因果 mask，是因为它的 pretext task 是"预测下一个词"，如果能看到未来，任务就退化成了抄写。

这两种 mask 在"是否阻挡信息流"这个意义上确有相似之处，但它们在系统中所处的层级完全不同：

<div align="center">

| | Masked Self-Attention (Lecture 8) | Masking as Pretext (Lecture 12) |
|---|---|---|
| **Mask 作用在哪** | 注意力权重矩阵 $E$ | 输入 token 序列 |
| **目的** | 阻止看到未来，保证自回归因果性 | 构造 pretext task，迫使语义理解 |
| **本质** | 架构约束（怎么算 attention） | 训练目标（让模型学什么） |
| **典型代表** | GPT 系列（自回归 LM） | BERT、MAE、Context Encoders |
| **能否去除** | 如果去掉 → 模型作弊（直接看答案） | 如果去掉 → 任务消失（没有要预测的东西了） |
| **视觉中的对应** | ViT 做图像分类不用 causal mask | MAE 对输入 patch 做 75% 随机 mask |
| **与 RNN 的关系** | RNN 天然因果，不需要显式 mask | RNN 时代有类似思路（双向 RNN + mask token），但效率远不如 Transformer |

</div>

### 延伸：视觉中的两种 mask 并存

值得注意的是，**视觉 Transformer 也存在同样的概念区分**：

- ViT 做图像分类时，所有 patch 彼此可见——**没有** causal mask（因为分类不需要自回归）。这相当于"去掉了 Masked Self-Attention"。
- MAE 在 ViT 的基础上对**输入**施加 75% 的随机 mask——encoder 只看到 25% 的可见 patch，decoder 负责重建被 mask 的 patch。这相当于"加上了 Masking as Pretext"。

因此，MAE 在视觉领域的角色恰好类似于 BERT 在 NLP 中的角色：同为 **双向 attention（无因果 mask）+ 输入层面的随机 mask**。两者共享同一个哲学——"遮住一部分，让模型从上下文去猜"——只是 BERT 对 token 操作，MAE 对 patch 操作。正如 [[lecture12 Self-supervised Learning#The Masking Idea and Asymmetric Architecture|Lecture 12 主线笔记]] 中所指出的，MAE 的灵感直接来自 BERT 的 Masked Language Modeling，但需要应对视觉领域的特有挑战——图像的空间冗余性使得低 masking 率下模型可以通过临近像素插值作弊，因此必须将 masking 率提高到 75%。

---

## 总结

Data Augmentation 和 SSL Pretext Task 的"相似感"不是错觉——它们共享"语义不变性"这条根——但生长方向截然不同：

<div align="center">

| 对比维度 | Data Augmentation | SSL Pretext Task | 对比学习（交汇点） |
|---------|------------------|------------------|------------------|
| **核心逻辑** | "变来变去还是同一个东西，所以忽略变化" | "你告诉我发生了什么变化 / 把变化还原" | "同一事物的不同变化应该在表征空间中靠近" |
| **不变性获取** | 被动——作为分类的副产品 | 主动——作为训练目标本身 | 主动——作为训练目标本身 |
| **对标签的依赖** | 依赖人工标签 | 不依赖（标签自动生成） | 不依赖（标签自动生成） |
| **变换强度** | 温和（保证标签有效） | 可以极端（无此约束） | 激进（越难越好） |
| **代表性方法** | 标准监督学习 + 增强 | Rotation、Jigsaw、Colorization、MAE | SimCLR、MoCo |

</div>

对于 Mask 的双重含义，关键区分在于它作用的层级：

- **Masked Self-Attention** 是**架构约束**——控制模型内部信息流，"不许看未来"
- **Masking as Pretext** 是**训练信号**——定义模型要做什么，"猜猜被遮住的是什么"

前者回答"怎么算"，后者回答"学什么"。BERT 之所以特别，正是因为它**同时去掉了前者（双向 attention）而加上了后者（input mask）**——这一去一加恰好展现了两种 mask 的独立性和互补性。MAE 将此设计迁移到视觉领域，用 75% 的极高 masking 率克服了图像的空间冗余问题，在 fine-tuning 任务上取得了优于同期对比学习方法的性能。
