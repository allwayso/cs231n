---
title: "Lecture 10: Video Understanding"
publish: true
target: CS231n Lecture 10 主线笔记：视频理解的 CNN 方法、Transformer 方法及多模态扩展
---

>[!SUMMARY] Table of Contents
>    - [[lecture10 Video Understanding#Video Data and Tasks|Video Data and Tasks]]
>        - [[lecture10 Video Understanding#Video = 2D + Time|Video = 2D + Time]]
>        - [[lecture10 Video Understanding#Video Classification|Video Classification]]
>        - [[lecture10 Video Understanding#Videos are Big!|Videos are Big!]]
>    - [[lecture10 Video Understanding#CNN-based Video Classification (2014–2021)|CNN-based Video Classification (2014–2021)]]
>        - [[lecture10 Video Understanding#Single-Frame CNN|Single-Frame CNN]]
>        - [[lecture10 Video Understanding#Late Fusion|Late Fusion]]
>        - [[lecture10 Video Understanding#Early Fusion with 3D CNNs|Early Fusion with 3D CNNs]]
>        - [[lecture10 Video Understanding#Two-Stream Networks|Two-Stream Networks]]
>        - [[lecture10 Video Understanding#I3D: Inflating 2D Networks to 3D|I3D: Inflating 2D Networks to 3D]]
>    - [[lecture10 Video Understanding#Transformer-based Video Understanding (2021–2026)|Transformer-based Video Understanding (2021–2026)]]
>        - [[lecture10 Video Understanding#ViT for Video: Token Explosion|ViT for Video: Token Explosion]]
>        - [[lecture10 Video Understanding#Strategy A: Modify Attention Operator|Strategy A: Modify Attention Operator]]
>        - [[lecture10 Video Understanding#Strategy B: Reduce Number of Tokens|Strategy B: Reduce Number of Tokens]]
>    - [[lecture10 Video Understanding#Beyond Short Clip Classification|Beyond Short Clip Classification]]
>        - [[lecture10 Video Understanding#Temporal Action Localization|Temporal Action Localization]]
>        - [[lecture10 Video Understanding#Spatio-Temporal Detection|Spatio-Temporal Detection]]
>        - [[lecture10 Video Understanding#Audio-Visual & Multisensory Video Understanding|Audio-Visual & Multisensory Video Understanding]]
>        - [[lecture10 Video Understanding#Long-form Video Understanding|Long-form Video Understanding]]
>    - [[lecture10 Video Understanding#Summary|Summary]]
>    - [[lecture10 Video Understanding#Materials|Materials]]

## Video Data and Tasks

### Video = 2D + Time

视频本质上是图像的序列。与单张 2D 图像（$3 \times H \times W$）不同，视频是一个 **4D tensor**：

$$
\text{Video} \in \mathbb{R}^{T \times 3 \times H \times W}
$$

其中 $T$ 是帧数（时间维度），$3$ 是 RGB 通道，$H$ 和 $W$ 是空间分辨率。


<div style="text-align: center;">
    <img src="Pasted image 20260615140105.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：Video = 2D + Time — 视频是图像的时间序列，组织为 4D tensor T×3×H×W</div>
</div>

### Video Classification

图像识别的任务是识别**物体**（objects），而视频理解的核心任务是识别**动作**（actions）：

- **图像分类**：给定一张图像，判断其中的物体类别——狗、猫、鱼、卡车……
- **视频分类**：给定一段视频，判断其中的动作类别——游泳、跑步、跳跃、吃饭、站立……

一个有代表性的视频数据集是 **Sports-1M**：包含 100 万个 YouTube 视频，标注了 487 种不同运动类别。

### Videos are Big!

视频数据面临的首要问题是**数据量巨大**。视频通常以约 30 fps（帧每秒）记录，未经压缩的视频大小估算如下：

- **SD（640 × 480）**：约 1.5 GB/分钟
- **HD（1920 × 1080）**：约 10 GB/分钟

这不仅对存储构成挑战，对训练和推理的**计算量**而言更是不切实际。


<div style="text-align: center;">
    <img src="Pasted image 20260615140229.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：训练短片段 — 使用低帧率、低分辨率的短 clip，训练时分类短片段，测试时在多个 clip 上运行并平均预测</div>
</div>

**解决方案**：在短片段（short clips）上训练模型——使用低帧率和低空间分辨率的小片段。例如 $T=16$，$H=W=112$，在 5 fps 下对应约 3.2 秒，大小约 588 KB。

训练/测试策略：
- **训练**：在低帧率的短片段上训练模型分类
- **测试**：在视频的不同片段上分别运行模型，**平均所有预测**(average predictions)得到最终结果

---

## CNN-based Video Classification (2014–2021)

### Single-Frame CNN

最朴素的想法：训练一个普通的 2D CNN，对视频的每一帧**独立**进行分类，测试时对各帧预测概率取平均。


<div style="text-align: center;">
    <img src="Pasted image 20260615140357.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：Single-Frame CNN — 每帧独立通过 2D CNN，测试时平均预测结果</div>
</div>

这种方法完全忽略了时间维度的运动信息，但令人惊讶的是，它往往是视频分类中一个**很强的 baseline**。因为在很多场景中，单帧的静态外观已经包含了足够的判别信息（如游泳池场景 → 游泳动作）。

### Late Fusion

Single-Frame CNN 的明显不足是无法建模帧间的时序关系。一个自然的改进是**晚融合（Late Fusion）**：先分别提取每帧的高层特征，再将它们融合。

**Late Fusion with FC Layers**：对每帧独立运行 2D CNN 得到特征图，将所有帧的特征拼接（concatenate），送入 MLP 进行分类。


<div style="text-align: center;">
    <img src="Pasted image 20260615140521.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：Late Fusion with FC Layers — 每帧提取特征→拼接→MLP分类</div>
</div>

然而拼接方式的缺点是**无法处理任意长度**的视频——MLP 的输入维度是固定的，只能处理预设帧数的片段。而且当时间长度 T 较长时，Fatten 得到的向量维度将会非常大，这会导致 MLP 中的层数很深，且参数量很大，不利于训练。


<div style="text-align: center;">
    <img src="Pasted image 20260615141148.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：Late Fusion with Pooling —— 用跨时间池化代替拼接</div>
</div>

**Late Fusion with Pooling**：用平均池化替代拼接，在空间和时间维度上做全局平均池化（average pool over space and time），得到固定维度的特征向量，再通过线性层分类。

Late Fusion with Pooling 可以处理任意长度的视频（池化不依赖固定帧数），但它的根本局限在于：**难以比较帧间的低层运动信息**。由于每帧先独立经过完整 CNN 提取高级语义特征，低层的运动线索（如边缘移动）已在逐帧的深层抽象中丢失。因此这种方法更适合高层场景分类，而不适合需要精细时序建模的任务。

Pooling 和 Flatten 的策略各自有局限，Flatten 将向量展平导致计算量巨大，而沿时间 pooling 又舍弃了底层运动线索，说明 T Features 全部都要和全都不要的策略都不可行。于是我们容易想到可以采用卷积，加入时间维度，这就是 3D CNNs

> 为什么不在特征图上做滑动窗口呢？
> 这种在 T 张特征图上做滑动窗口融合的方法，虽然能局部保留细节，但本质上会带来重复计算开销（窗口重叠导致大量冗余融合）、难以有效建模长程依赖（跨窗口信息传播受限且易衰减），同时在工程实现上造成不连续的内存访问和较低的GPU利用效率，最终往往在效率和全局表达能力之间都不如全局融合或注意力机制更均衡的方案。

### Early Fusion with 3D CNNs

与 Late Fusion 的"先提取后融合"不同，**早融合（Early Fusion)** 在网络的最早期就开始融合时间信息。核心工具是 **3D 卷积**。

**核心思想**：将 2D CNN 中的卷积和池化操作全部替换为 3D 版本，在网络的前向传播中逐步融合时间信息。网络中的每个激活图都是一个 4D tensor：$D \times T \times H \times W$。


<div style="text-align: center;">
    <img src="Pasted image 20260615143014.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 6：Early Fusion with 3D CNN — 使用 3D 卷积和 3D 池化逐步融合时间信息</div>
</div>

回顾 2D 卷积：输入 $C_{in} \times H \times W$，卷积核 $C_{out} \times C_{in} \times K_h \times K_w$，输出 $C_{out} \times H' \times W'$。

**3D 卷积** 在此基础上增加了一个时间维度：
- 输入：$C_{in} \times T \times H \times W$
- 卷积核：$C_{out} \times C_{in} \times K_t \times K_h \times K_w$
- 输出：$C_{out} \times T' \times H' \times W'$


<div style="text-align: center;">
    <img src="Pasted image 20260615143238.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：3D 卷积细节 — 卷积核增加了时间维度，在时间轴上也可滑动（stride rules apply）</div>
</div>

3D 卷积与 2D 卷积的关键差异：
- **卷积核多了一个时间维度**：$K_t \times K_h \times K_w$，一个滤波器可以同时捕捉空间模式和时间变化
- **滑动窗口机制相同**，但可以在时间轴上前后移动（stride 规则同样适用）
- **输出多了一个时间维度**：输出激活图形状为 $T' \times C_{out} \times H' \times W'$

> 实际上存在 Early fusion with 2D conv，与 3D conv 的区别在于前者在一开始就把 T 加入了特征维度，此后对 $H*W*C'$ 做二维卷积，也就是说前者在一开始就具有整个 T 维度的感受野。这一部分在 2025 Spring 的ppt中出现，并花了大量篇幅做区分，而在本年ppt中略去了两者的比较，如有兴趣可以参考[Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 10: Video Understanding](https://www.youtube.com/watch?v=wElqklprhPE&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)中 16-30min 的内容。
### Two-Stream Networks

人类可以从纯运动信息中轻松识别动作——即使只看到关节处的光点运动（Johansson 1973 的生物运动感知实验），也能判断出走路、跑步等动作。这启发了一个想法：**显式地分离运动信息**和外观信息。

<div style="text-align: center;">
    <img src="Pasted image 20260615155206.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：Optical Flow — 光流给出帧间位移场 F(x,y) = (dx, dy)</div>
</div>

**光流（Optical Flow）** 是度量帧间运动的核心工具。给定相邻两帧 $I_t$ 和 $I_{t+1}$，光流给出一个位移场 $F$：

$$
F(x, y) = (dx, dy), \quad I_{t+1}(x+dx, y+dy) = I_t(x, y)
$$

光流可以被可视化为水平位移（$dx$）和垂直位移（$dy$）两个通道，突出显示局部运动区域。

**Two-Stream 网络架构**：
- **空间流（Spatial Stream）**：输入单帧 RGB 图像（$3 \times H \times W$），捕捉外观信息
- **时间流（Temporal Stream）**：输入堆叠的光流帧（$[2(T-1)] \times H \times W$），捕捉运动信息
- 两条流的预测结果通过平均或 SVM 进行融合


<div style="text-align: center;">
    <img src="Pasted image 20260615155329.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：Two-Stream Networks — 空间流 + 时间流，分别建模外观和运动</div>
</div>

在 UCF-101 数据集上的精度对比：

| 方法 | 精度 |
|------|------|
| 3D CNN | 65.4% |
| Spatial only | 73.0% |
| Temporal only | 83.7% |
| Two-stream (average) | 86.9% |
| Two-stream (SVM) | 88.0% |

两个有趣的发现：(1) 纯运动信息（光流）的精度（83.7%）显著高于单帧外观（73.0%），说明运动对动作识别极其重要；(2) 两条流互补，融合后进一步提升。

不过近年来，光流在视频理解中的直接使用已大大减少，更多作为其他任务（如机器人控制）的中间表示。

### I3D: Inflating 2D Networks to 3D

图像领域有大量精心设计的 CNN 架构（如 Inception、ResNet）。能否直接复用这些架构到视频？

**I3D（Inflated 3D ConvNets）** 的核心思想非常简洁：**取一个预训练好的 2D CNN，将每个 2D 卷积/池化层替换为对应的 3D 版本**。

**权重初始化技巧**：用预训练的 2D 卷积核初始化 3D 卷积核——将 2D 卷积核沿时间维度复制 $K_t$ 次，再除以 $K_t$。这样，给定"恒定"的输入视频（所有帧相同），3D 卷积的输出与 2D 卷积完全一致，从而完美继承了预训练的图像特征。


<div style="text-align: center;">
    <img src="Pasted image 20260615155544.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 10：I3D 在 Kinetics-400 上的结果 — 使用 ImageNet 预训练权重初始化的 I3D 显著优于从头训练</div>
</div>

在 Kinetics-400 数据集上的 Top-1 精度：

- 单帧 CNN + LSTM：53.9%
- 3D CNN（从头训练）：62.2%
- Two-stream CNN：65.6%
- Two-stream inflated CNN（ImageNet 预训练）：**74.2%**

I3D 的核心启示：**利用大规模图像预训练是视频理解的关键**。将 2D 知识迁移到 3D 远比从头训练有效。

---

## Transformer-based Video Understanding (2021–2026)

看完了 CNN 时代（2014–2021）的方法，接下来看 Transformer 时代（2021–2026）。核心问题从"如何设计卷积核"转向了"如何高效处理大量 token"。

### ViT for Video: Token Explosion

> 这里直接将 ViT 作为前置知识，但是实际上前面的笔记中只有 [[lecture08 Attention and Transformers#Vision Transformers]] 中粗略的介绍了 Vision Transformer，考虑到它的重要性，我补充了一篇关于 ViT 的笔记 [[Vision Transformer (ViT)]] 供你参考

将 ViT 直接应用于视频看似自然——把每一帧切分成 patches，将所有帧的 patches 一起送入 Transformer。但这带来严重的 **token 爆炸**问题：

- 单帧（224×224，16×16 patch）：$14 \times 14 = 196$ 个 token
- 小片段（$T=16$）：$16 \times 196 = 3136$ 个 token（~一本书的篇幅）
- 5 分钟 @ 1 fps（$T=300$）：$300 \times 196 = 58,800$ 个 token（~一篇短篇小说）
- 5 分钟 @ 24 fps（$T=7200$）：$7200 \times 196 \approx 1,411,200$ 个 token（**接近现代 LLM 的上下文长度极限**）

<div style="text-align: center;">
    <img src="Pasted image 20260615173412.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 11：ViT for Video 遇到的问题</div>
</div>

标准 self-attention 的复杂度是 $O(N^2)$，$N$ 为 token 数。naïve patch-level attention 根本无法扩展到长视频。解决视频 Transformer 效率问题有**两大策略**。


<div style="text-align: center;">
    <img src="Pasted image 20260615174411.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 12：Two Broad Strategies — 修改注意力算子 vs 减少 token 数量</div>
</div>
### Strategy A: Modify Attention Operator

#### Divided Space-Time Attention (ViViT)

标准 joint space-time attention 让每个 token 关注**所有帧的所有空间位置**——复杂度 $O((NT)^2)$，极其低效。

**Divided Space-Time Attention** 将联合注意力拆分成两步：

<div style="text-align: center;">
    <img src="Pasted image 20260615174718.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 13：Divided Space-Time Attention — Time Attention + Space Attention 两步操作</div>
</div>

1. **Time Attention**：每个 token 只关注**同一空间位置、不同帧**的 token（沿时间轴）
2. **Space Attention**：每个 token 只关注**同一帧内、不同空间位置**的 token（沿空间轴）

计算复杂度的变化：

$$
O(NT)^2 \rightarrow O(N^2 + T^2) \quad \text{或更精确地说} \quad O(N \cdot T^2) + O(T \cdot N^2)
$$

实际实现中，Time Attention 每个空间位置的复杂度为 $O(T^2)$，共有 $N$ 个空间位置 → $O(N \cdot T^2)$；Space Attention 每帧的复杂度为 $O(N^2)$，共有 $T$ 帧 → $O(T \cdot N^2)$。两个操作可作用于不同 token 集合，大幅降低总计算量。关键的是，经过多个 block 的堆叠，信息仍然可以在空间和时间维度间自由传播。

#### Video Swin Transformer

另一个思路：**将 self-attention 限制在局部时空立方体中**。这与 3D CNN 的设计哲学相似——用局部窗口 + 窗口移位（shift cubes between layers）让信息跨窗口传播。


<div style="text-align: center;">
    <img src="Pasted image 20260615174837.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 14：Video Swin Transformer — 局部时空立方体内的 self-attention</div>
</div>
#### MViT: Multiscale Vision Transformers

MViT 借鉴了 CNN 中"逐层减小空间分辨率、增加通道数"的经典设计。关键做法：**在计算 attention 之前，先压缩 K 和 V 的空间尺寸**（通过卷积聚合邻近 token）。


<div style="text-align: center;">
    <img src="Pasted image 20260615175354.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 15：MViT — 压缩 K/V 序列再计算 attention，逐层减半空间维度、翻倍通道数</div>
</div>

例如，56×56 网格的 K/V 向量通过 4×4 卷积（stride 4）压缩为 14×14 网格。Q 保持原始分辨率以保证输出长度不变。随着层数加深，MViT 逐渐降低空间分辨率并增加通道数：

$$
56 \times 56 \text{ (96 dim)} \rightarrow 28 \times 28 \rightarrow 14 \times 14 \text{ (384 dim)}
$$

这与 ResNet 的设计原则完全一致。

### Strategy B: Reduce Number of Tokens

第二条路径是从源头减少 token 数量。最直接的方法是 **Tubelets**。


<div style="text-align: center;">
    <img src="Pasted image 20260615175950.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 16：Tubelets — 跨帧提取时空立方体，远少于 patches + 包含运动信息</div>
</div>

与 patch 只在空间维度切分不同，tubelet 在**空间和时间维度上同时切分**：
- 传统 patches：每帧独立切分，得到 $T \times N$ 个 token，各 patch 不含运动信息
- Tubelets：跨多帧提取时空立方体（如 $t \times h \times w$），token 数量大幅减少，且每个 tubelet 天然包含运动信息

例如，跨 4 帧的 tubelet：token 数减少 4 倍 → 计算量减少 **16 倍**（因为 attention 是 $O(N^2)$）。

Tubelets 已成为视频 Transformer 的**标配技术**，被 ViViT、VideoMAE、Video Swin、MViT、V-JEPA 等大量模型采用。此外还有自适应 token 选择、token 合并、可学习压缩等方法（将在 Lecture 16 中讨论）。

---

## Beyond Short Clip Classification

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260615180926.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 17：Temporal Action Localization — 在长视频中识别不同动作的时间区间</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260615181009.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 18：Spatio-Temporal Detection</div>
    </div>
</div>

### Temporal Action Localization

目前为止讨论的都是短片段分类。但真实视频通常很长且未经剪辑，包含多种动作。**时序动作定位（Temporal Action Localization** 的任务是：给定一段长视频，找出不同动作对应的时间区间。

这可以类比目标检测中的 Faster R-CNN：先生成时序上的候选区域（temporal proposals），再对每个区域分类。事实上，Faster R-CNN 的架构可以自然地适配到时域——将一维时间轴视为空间轴的类比，用类似 RPN 的机制生成和精修 temporal proposals。

### Spatio-Temporal Detection

更进一步，**时空检测（Spatio-Temporal Detection)** 同时定位 **空间中的每个人** 和 **时间上的动作** 。例如，在一个多人的视频中，检测每个人在每一时刻的活动。

AVA Dataset 是一个代表性的时空检测数据集，标注了视频中每个人的边界框和动作类别。

### Audio-Visual & Multisensory Video Understanding

视频天然是多模态的——不仅包含视觉信息，还包含**音频**。多感官融合（multisensory fusion）让模型同时利用视觉和听觉线索。

一个令人印象深刻的应用是**视觉引导的音频源分离**（visually-guided audio source separation）：给定一段多人交谈的视频（混合语音），利用视觉信息（说话者的面部/嘴部运动）分离出每个人的语音。类似地，也可以分离不同乐器的声音。

另一个应用方向是**以音频为预览机制的高效动作识别**：用音频信号引导视觉模型关注视频中的关键时刻，避免对每一帧都运行完整的视觉处理。


<div style="text-align: center;">
    <img src="Pasted image 20260615181831.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 19：Audio-Visual Video Understanding — 视觉引导音频分离、多模态融合等</div>
</div>
在 VideoLLM 方向，Video-LLaVA、VideoLLaMA 3、Video-ChatGPT 等工作正在将视觉-语言大模型扩展到视频理解领域，实现视频问答、描述生成等功能。

### Long-form Video Understanding

人类具有在超长时间跨度上处理视觉刺激的非凡能力，但当前的计算机视觉系统能理解长视频吗？

**HourVideo**（NeurIPS 2024）提供了一个评估基准：包含长达 1 小时的 egocentric 视频，提问需要跨时间推理（如"相机佩戴者在锻炼后把 AirPods 放在哪里？""如何从厨房走到后院？"）。结果揭示当前模型与人类之间仍然存在**显著差距**——这将是未来重要的研究方向。

---

## Summary

Lecture 10 覆盖了视频理解的完整技术演进路线：

**视频数据特点**：4D tensor（$T \times 3 \times H \times W$），数据量巨大，实际中在短片段上训练。

**CNN 时代（2014–2021）**：

| 方法 | 核心思想 | 特点 |
|------|----------|------|
| **Single-Frame CNN** | 逐帧分类，平均预测 | 强 baseline，忽略运动 |
| **Late Fusion** | 逐帧提取特征，后期融合 | 可处理变长视频，缺失低层运动 |
| **3D CNN (Early Fusion)** | 3D 卷积逐步融合时空信息 | 显式建模时空，计算量大 |
| **Two-Stream** | 空间流（外观）+ 时间流（光流） | 显式分离运动和外观 |
| **I3D** | 2D CNN → 3D 膨胀 + 预训练迁移 | 复用图像架构和权重，性能大幅提升 |

**Transformer 时代（2021–2026）**：

| 策略 | 方法 | 核心思想 |
|------|------|----------|
| **修改注意力** | Divided Space-Time Attention | 分离时间注意力和空间注意力，$O(N^2T^2) \rightarrow O(NT^2 + TN^2)$ |
| | Video Swin Transformer | 局部时空窗口 + 窗口移位 |
| | MViT | 压缩 K/V，逐层减半空间分辨率 |
| **减少 token** | Tubelets | 时空立方体替代空间 patches，大幅减少 token 数 |

**超越短片段**：
- **Temporal Action Localization**：在长视频中定位动作时间区间（类比 Faster R-CNN 的时域版本）
- **Spatio-Temporal Detection**：同时在空间和时间上检测人的动作
- **Audio-Visual Understanding**：融合视觉和音频，实现音源分离、多模态识别
- **Long-form Video**：HourVideo 等基准推动模型向小时级视频理解发展，当前与人类差距仍然巨大
- **VideoLLMs**：视觉-语言大模型向视频领域的扩展

## Materials

- [Large-scale Video Classification with Convolutional Neural Networks (Karpathy et al., CVPR 2014)](https://arxiv.org/abs/1412.0767)
- [3D Convolutional Neural Networks for Human Action Recognition (Ji et al., TPAMI 2010)](https://ieeexplore.ieee.org/document/6165309)
- [Two-Stream Convolutional Networks for Action Recognition in Videos (Simonyan & Zisserman, NeurIPS 2014)](https://arxiv.org/abs/1406.2199)
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (Carreira & Zisserman, CVPR 2017)](https://arxiv.org/abs/1705.07750)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., ICLR 2021)](https://arxiv.org/abs/2010.11929)
- [ViViT: A Video Vision Transformer (Arnab et al., ICCV 2021)](https://arxiv.org/abs/2103.15691)
- [Is Space-Time Attention All You Need for Video Understanding? (Bertasius et al., ICML 2021)](https://arxiv.org/abs/2102.05095)
- [Video Swin Transformer (Liu et al., CVPR 2022)](https://arxiv.org/abs/2106.13230)
- [Multiscale Vision Transformers (Fan et al., ICCV 2021)](https://arxiv.org/abs/2104.11227)
- [Visual perception of biological motion and a model for its analysis (Johansson, Perception & Psychophysics 1973)](https://link.springer.com/article/10.3758/BF03207878)
- [Rethinking the Faster R-CNN Architecture for Temporal Action Localization (Chao et al., CVPR 2018)](https://arxiv.org/abs/1804.07667)
- [AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions (Gu et al., CVPR 2018)](https://arxiv.org/abs/1705.08421)
- [Listen to Look: Action Recognition by Previewing Audio (Gao et al., CVPR 2020)](https://arxiv.org/abs/1912.04487)
- [HourVideo: 1-Hour Video-Language Understanding (Chandrasegaran et al., NeurIPS 2024)](https://arxiv.org/abs/2411.04998)
- [CS231n 2024/2025 Lecture 10 Slides](https://cs231n.stanford.edu/slides/2025/lecture_10.pdf)
