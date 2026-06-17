---
title: "Lecture 11: Large-Scale Distributed Training"
publish: true
target: CS231n Lecture 11 主线笔记：GPU 硬件架构、大规模分布式训练与并行策略
---

>[!SUMMARY] Table of Contents
>    - [[lecture11 Large-Scale Distributed Training#Opening: Llama3-405B|Opening: Llama3-405B]]
>    - [[lecture11 Large-Scale Distributed Training#GPU Hardware|GPU Hardware]]
>        - [[lecture11 Large-Scale Distributed Training#Inside an H100 GPU|Inside an H100 GPU]]
>        - [[lecture11 Large-Scale Distributed Training#GPU Performance Evolution: 1000× in 12 Years|GPU Performance Evolution: 1000× in 12 Years]]
>    - [[lecture11 Large-Scale Distributed Training#GPU Clusters: Many GPUs as One Computer|GPU Clusters: Many GPUs as One Computer]]
>        - [[lecture11 Large-Scale Distributed Training#Memory Hierarchy Across the Cluster|Memory Hierarchy Across the Cluster]]
>        - [[lecture11 Large-Scale Distributed Training#Meta's Llama3 Cluster|Meta's Llama3 Cluster]]
>        - [[lecture11 Large-Scale Distributed Training#Other Training Hardware|Other Training Hardware]]
>    - [[lecture11 Large-Scale Distributed Training#Five Degrees of Parallelism|Five Degrees of Parallelism]]
>    - [[lecture11 Large-Scale Distributed Training#Data Parallelism (DP)|Data Parallelism (DP)]]
>        - [[lecture11 Large-Scale Distributed Training#Basic Data Parallelism|Basic Data Parallelism]]
>        - [[lecture11 Large-Scale Distributed Training#Fully Sharded Data Parallelism (FSDP)|Fully Sharded Data Parallelism (FSDP)]]
>        - [[lecture11 Large-Scale Distributed Training#Hybrid Sharded Data Parallelism (HSDP)|Hybrid Sharded Data Parallelism (HSDP)]]
>        - [[lecture11 Large-Scale Distributed Training#Activation Checkpointing|Activation Checkpointing]]
>        - [[lecture11 Large-Scale Distributed Training#Scaling Recipe (DP → FSDP → HSDP)|Scaling Recipe (DP → FSDP → HSDP)]]
>    - [[lecture11 Large-Scale Distributed Training#Model FLOPs Utilization (MFU)|Model FLOPs Utilization (MFU)]]
>        - [[lecture11 Large-Scale Distributed Training#Hardware FLOPs Utilization (HFU)|Hardware FLOPs Utilization (HFU)]]
>        - [[lecture11 Large-Scale Distributed Training#From HFU to MFU|From HFU to MFU]]
>    - [[lecture11 Large-Scale Distributed Training#Context Parallelism (CP)|Context Parallelism (CP)]]
>    - [[lecture11 Large-Scale Distributed Training#Pipeline Parallelism (PP)|Pipeline Parallelism (PP)]]
>    - [[lecture11 Large-Scale Distributed Training#Tensor Parallelism (TP)|Tensor Parallelism (TP)]]
>    - [[lecture11 Large-Scale Distributed Training#ND Parallelism: Putting It All Together|ND Parallelism: Putting It All Together]]
>        - [[lecture11 Large-Scale Distributed Training#为什么必须分层混合|为什么必须分层混合]]
>    - [[lecture11 Large-Scale Distributed Training#Summary|Summary]]
>    - [[lecture11 Large-Scale Distributed Training#Materials|Materials]]

## Opening: Llama3-405B

十年前，用单个 GPU 训练模型是常态。而今天，**在数十、数百、数千甚至数万个设备上并行训练**已成为深度学习的新范式。本讲讨论的核心问题是：如何利用一个拥有数万 GPU 的集群，高效训练一个巨大的神经网络？

本讲以 **Llama3-405B** 作为贯穿实例——不是因为它是最好的模型，而是因为它是最开放的模型之一。2023 年 GPT-4 技术报告中的一段话标志了行业的转向：

> *"Given both the competitive landscape and the safety implications of large-scale models like GPT-4, this report contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar."*

相比之下，Meta 于 2024 年 4 月开源的 Llama3 论文分享了大量关于模型训练和系统基础设施的细节，为我们提供了窥见大规模 LLM 训练实践的窗口。

---

## GPU Hardware

### Inside an H100 GPU

GPU（Graphics Processing Unit）最初为计算机图形学设计，因其天然的大规模并行特性而被重新定位为**通用并行处理器**。NVIDIA H100 是当前深度学习训练的主力。

<div style="text-align: center;">
    <img src="Pasted image 20260617120113.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：NVIDIA H100 GPU 内部 — 中央为计算核心，周围环绕 80GB HBM 高带宽内存，带宽约 3 TB/s</div>
</div>

深入 H100 的计算核心，可以看到**三级存储层次**：

| 层级 | 大小 | 特点 |
|------|------|------|
| **HBM 内存** | 80 GB | 离计算核心最远，~3 TB/s 带宽 |
| **L2 Cache** | 50 MB | 更接近计算核心，访问更快 |
| **L1 Cache + 寄存器** | 256 KB / SM | 最近、最快 |

<div style="text-align: center;">
    <img src="Pasted image 20260617120213.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：H100 Streaming Multiprocessor (SM) — 共 132 个活跃 SM（芯片上有 144 个，由于良率问题仅启用 132 个），每个 SM 包含 FP32 核心和 Tensor 核心</div>
</div>

每个 SM 内部有两类关键计算单元：

**FP32 Cores（128 个/SM）**：
- 每个时钟周期计算 $a \cdot x + b$（标量操作）
- 每个 SM 每周期 $128 \times 2 = 256$ FLOP

**Tensor Cores（4 个/SM）**：
- 每个时钟周期做一次**小矩阵乘法**：$A_{[16 \times 4]} \times B_{[4 \times 8]} + C_{[16 \times 8]}$
- 单次操作 $16 \times 4 \times 8 \times 2 = 1024$ FLOP
- 每个 SM 每周期 $4 \times 1024 = 4096$ FLOP
- Tensor Cores 是 GPU **吞吐量的真正来源**（16× vs FP32 cores）

Tensor Cores 采用**混合精度**工作：输入为 16-bit（低精度乘法），累加过程使用 32-bit（高精度加法）。这意味着如果在 PyTorch 中忘记将模型 cast 为 16-bit，模型会回退到 FP32 cores 运行，速度可能比预期慢 **20 倍**。

### GPU Performance Evolution: 1000× in 12 Years

从 2013 年的 K40 到 2025 年的 B200，GPU 算力经历了惊人增长：

<div style="text-align: center;">
    <img src="Pasted image 20260617120626.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：GPU 算力增长 — 从 K40 (5 FP32 TFLOP/s) 到 B200 (5000 TC TFLOP/s)，12 年间约 1000× 提升</div>
</div>

关键转折点出现在 **V100（2016-2017）**——首次引入 Tensor Cores。此后每一代都在增大 Tensor Core 的芯片面积占比。近 12 年单设备算力提升约 1000 倍，这是 AI 能力飞跃的根本驱动力之一。

---

## GPU Clusters: Many GPUs as One Computer

### Memory Hierarchy Across the Cluster

存储层次不仅在单个 GPU 内部存在，在**集群层面**同样延续——离计算核心越远，通信带宽越低：
### Meta's Llama3 Cluster

Meta 为训练 Llama3 构建的 GPU 集群提供了非常详细的公开数据：

| 层级               | 配置       | GPU 数      | GPU 间带宽      |
| ---------------- | -------- | ---------- | ------------ |
| **单个 GPU**       | 1× H100  | 1          | ~3 TB/s (内部) |
| **GPU Server**   | 8× H100  | 8          | ~900 GB/s    |
| **Server Rack**  | 2 Server | 16         | —            |
| **GPU Pod**      | 192 Rack | 3,072      | ~50 GB/s     |
| **Full Cluster** | 8 Pod    | **24,576** | <50 GB/s     |

<div style="text-align: center;">
    <img src="Pasted image 20260617120917.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：集群存储层次 — GPU 内部 ~3 TB/s → Server 内 GPU 间 ~900 GB/s (3× 降低) → Pod 内 ~50 GB/s (再降 18×)</div>
</div>

将这个集群视为**一台巨大的计算机**：

- 24,576 个 GPU
- 1.875 PB GPU 内存
- 4.15 亿 FP32 Core
- 1300 万 Tensor Core
- **24.3 EFLOP/s** 总算力（$24.3 \times 10^{18}$）

核心思维转变：**不再考虑单个设备，而是将整个数据中心视为一台超级计算机**。问题变成：如何在这台超级计算机上连续数月训练一个巨型神经网络？

最长的训练通常持续**数月**（人的规划周期限制），GPT-4.5/5 等可能接近一年。但这些集群不是最大的——世界上已知存在 50,000–100,000 GPU 的集群。

### Other Training Hardware

除 NVIDIA 外，其他竞争者也在涌现：

- **Google TPU**（Tensor Processing Unit）：已迭代六代，v5p TPU 约 3151 BF16 TFLOP/s，216GB 内存，可组成 8960 芯片的 Pod。Gemini 模型几乎确定在 TPU 上训练。只能通过 Google Cloud 使用。
- **AMD MI355X**：2500 BF16 TFLOP/s，288GB 内存
- **AWS Trainium3**：2500 FP8 TFLOP/s，144GB 内存，Anthropic 用于部分训练

---

## Five Degrees of Parallelism

Transformer 模型的激活值是 4D tensor：$(\text{Layer}, \text{Batch}, \text{Sequence}, \text{Dim})$。这定义了**四种可并行的轴**：

<div style="text-align: center;">
    <img src="Pasted image 20260617121256.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：四种并行策略 </div>
</div>

此外还有 **Expert Parallelism**（用于 MoE 模型）。

- Data Parallelism :批次维度的并行，即每个 GPU 分管若干个 minibatch
- Context Parallelism : 序列维度的并行，即每个 GPU 分管样本中一定长度的序列
- Pipeline Parallelism ：层维度的并行，即每个 GPU 分管模型中的若干个层级，并通过 Microbatch 实现并行
- Tensor Parallelism (Dim 维度)：通道维度的并行，即每个 GPU 分管权重矩阵的某一块

---

## Data Parallelism (DP)

### Basic Data Parallelism

数据并行是最自然、最古老的并行策略，核心思想来自**梯度的线性性**。

回忆标准训练：对 minibatch 中每个样本的 loss 取平均，计算梯度。由于梯度算子线性：

$$
L = \frac{1}{MN} \sum_{j=1}^{M} \sum_{i=1}^{N} L(x_{i,j}, W)
$$

$$
\frac{\partial L}{\partial W} = \frac{1}{M} \sum_{j=1}^{M} \underbrace{\left[ \frac{1}{N} \sum_{i=1}^{N} \frac{\partial L(x_{i,j}, W)}{\partial W} \right]}_{\text{每个 GPU 独立计算}}
$$

**这不是近似，而是一模一样的数学运算**——只是改变求和与求平均的顺序。

<div style="text-align: center;">
    <img src="Pasted image 20260617231039.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：数据并行 — M 个 GPU 各维护模型副本，加载不同 minibatch，独立 forward/backward，然后 All-Reduce 梯度</div>
</div>

**数据并行六步**：
1. 每个 GPU 拥有自己的模型参数和优化器状态副本
2. 每个 GPU 加载**不同**的 minibatch 数据
3. 每个 GPU 独立完成 forward pass 计算 loss
4. 每个 GPU 独立完成 backward pass 计算局部梯度
5. 通过 **All-Reduce** 在所有 GPU 间平均梯度
6. 每个 GPU 用相同的平均梯度独立更新权重（权重保持同步）

> **关键优化**：步骤 4 和 5 可以重叠——GPU 在计算第 L 层的 backward 时，同时在做第 L+1 层梯度的 All-Reduce。

<div style="text-align: center;">
    <img src="Pasted image 20260617121439.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：All-Reduce — 每个 GPU 持有不同 tensor，操作后每个 GPU 都得到所有 tensor 的归约结果（如求和）。这是数据并行最早需要、也是唯一需要的通信原语</div>
</div>

步骤 5 中每个 GPU 需要把自己算出的局部梯度求和后发回所有人——这个"**归约（reduce）+ 广播（broadcast）**"的复合操作就是 **All-Reduce**。它随着最早的数据并行需求而诞生，在纯 DP 时代是唯一的通信原语。

**局限**：纯 DP 要求每个 GPU 保存完整的模型副本。每个权重需要约 4 个数（权重 + 梯度 + Adam 的 $\beta_1$ + $\beta_2$），16-bit 精度下每参数约 8 字节。1B 参数需要 8 GB → H100 80 GB 最多容纳约 10B 参数。这不够大。

### Fully Sharded Data Parallelism (FSDP)

FSDP 的解决方案：**把模型权重也拆分到不同 GPU 上**，而不仅仅拆分数据。

<div style="text-align: center;">
    <img src="Pasted image 20260617234650.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：FSDP 工作流程 — 权重由不同 GPU 拥有，forward 时广播、用后删除，backward 时重新广播、聚合梯度</div>
</div>

**FSDP 核心机制**：

- 每个权重矩阵 $W_i$ 被分配给一个 **owner GPU**，该 GPU 负责维护该权重的全局梯度和优化器状态
- **Forward pass**：owner 广播 $W_i$ → 所有 GPU 计算 → 非 owner 删除 $W_i$（节省内存）→ 同时**预取** $W_{i+1}$
- **Backward pass**：owner 广播 $W_i$ → 所有 GPU 计算局部梯度 → 各 GPU 将局部梯度发送回 owner → owner 聚合并更新权重
- 稳态时，三个操作并行进行：**计算 layer L 的 backward + 发送/更新 layer L+1 的梯度 + 预取 layer L-1 的权重**

FSDP 的 forward 和 backward 各需要一种新的通信模式：

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <div style="text-align: center;">
    <img src="Pasted image 20260617121636.png" width="400" />
	</div>
    <div style="text-align: center;">
    <img src="Pasted image 20260617121527.png" width="400" />
	</div>
</div>
<div style="font-size: 1em; color: #888; margin-top: 5px;text-align: center;">图 10：All-Gather 和 Reduce-Scatter</div>


**Forward → All-Gather**：权重被分片在各 GPU 上，但 forward 时每个 GPU 都需要完整权重。于是每个 GPU 持有权重的不同分片，通过 All-Gather 将碎片"收集齐"发给所有人。

**Backward → Reduce-Scatter**：每个 GPU 对所有权重都算出了局部梯度，但每个权重只归一个 owner GPU 管理。Reduce-Scatter 先对各 GPU 的局部梯度求和，再把结果**分散**到各自的 owner。

实际上，每个权重并非由一个 GPU 完整拥有，而是被 **分片（shard)** 到所有 GPU。Forward 用 All-Gather 收集权重，Backward 用 Reduce-Scatter 聚合梯度——比简单的广播/Reduction 更高效。注意 *All-Reduce = Reduce-Scatter + All-Gather* ：FSDP 用这对组合替换了 DP 时代的单次 All-Reduce——因为所有权已分散，不再需要"每人拿到完整梯度"。

FSDP 大幅降低了模型参数、梯度和优化器状态的显存占用，使得单个模型能够扩展到远超单卡容量的规模。然而，当训练规模继续扩大到数十甚至数百台服务器时，FSDP 的**通信成本**开始成为新的瓶颈。

原因在于，FSDP 的 **All-Gather 和 Reduce-Scatter 都发生在整个 FSDP 通信组内**。若通信组覆盖所有 GPU，则每一层都需要跨整个集群交换参数和梯度。随着 GPU 数量增加，通信路径逐渐从单机 NVLink 扩展到跨机 InfiniBand，甚至跨 Pod 网络：

1. **显存占用继续下降**（分片数更多）；
2. **通信量并不会减少**，反而会受到网络层级和带宽限制；
3. 每层都需要执行一次 All-Gather 和 Reduce-Scatter，使得训练逐渐从 **计算受限（compute-bound）** 转变为 **通信受限（communication-bound）**。

因此，在大规模集群上，人们通常不会将整个集群组成一个超大的 FSDP 组，而是让 FSDP 仅在高带宽节点内部工作，再利用数据并行连接多个节点。这便引出了 **Hybrid Sharded Data Parallelism (HSDP)**。

### Hybrid Sharded Data Parallelism (HSDP)

HSDP 将 GPU 组织为 **2D 网格**，同时使用两种并行：

- **组内（K 个 GPU）**：FSDP —— 高频通信（3× 模型大小 / forward-backward pass）
- **组间（M 个组）**：纯 DP —— 低频通信（1× 模型大小 / forward-backward pass）

<div style="text-align: center;">
    <img src="Pasted image 20260617235313.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 11：HSDP — 二维 GPU 网格，组内 FSDP（高带宽） + 组间 DP（低带宽），匹配集群拓扑</div>
</div>

**拓扑感知设计**：将 FSDP 组放在同一台服务器或同一 Pod 内（利用高带宽），组间 DP 通信可以跨 Pod（容忍较低带宽）。这是**多维并行**的首个例子。

**局限**：即使权重被分片，**激活值**（layer outputs，backward 需要用来计算梯度）仍然占满内存。Llama3-405B（126 层，$D=16384$，seq=4096）仅 FFN 隐藏激活值就需 ~63 GB。需要某种机制来节省激活值带来的开销，这就引入了 Activation Checkpointing 。

### Activation Checkpointing

**核心思想**：不在 forward 时保存所有激活值，而是在 backward 时**重新计算**。

我们通过一个层数为4的小模型作为例子，就可以很容易理解激活值检查点的四种策略：假设前向传播和反向传播的时间、内存复杂度都为 $O(1)$ 以方便理解


<div style="display: flex; justify-content: center; gap: 15px; align-items: stretch; width: 900px; margin: 0 auto;">
    <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center;">
        <div style="height: 260px; display: flex; align-items: center; justify-content: center;">
            <img src="Pasted image 20260618004623.png" style="max-height: 100%; max-width: 100%; width: auto; height: auto;" />
        </div>
        <div style="font-size: 0.9em; color: #666; margin-top: auto; padding-top: 5px;">step 4</div>
    </div>
    <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center;">
        <div style="height: 260px; display: flex; align-items: center; justify-content: center;">
            <img src="Pasted image 20260618004655.png" style="max-height: 100%; max-width: 100%; width: auto; height: auto;" />
        </div>
        <div style="font-size: 0.9em; color: #666; margin-top: auto; padding-top: 5px;">step 5</div>
    </div>
    <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center;">
        <div style="height: 260px; display: flex; align-items: center; justify-content: center;">
            <img src="Pasted image 20260618004729.png" style="max-height: 100%; max-width: 100%; width: auto; height: auto;" />
        </div>
        <div style="font-size: 0.9em; color: #666; margin-top: auto; padding-top: 5px;">step 6</div>
    </div>
    <div style="text-align: center; flex: 1; display: flex; flex-direction: column; align-items: center;">
        <div style="height: 260px; display: flex; align-items: center; justify-content: center;">
            <img src="Pasted image 20260618004802.png" style="max-height: 100%; max-width: 100%; width: auto; height: auto;" />
        </div>
        <div style="font-size: 0.9em; color: #666; margin-top: auto; padding-top: 5px;">step 7</div>
    </div>
</div>
<div style="font-size: 1em; color: #888; margin-top: 10px; text-align: center; width: 900px; margin: 10px auto 0 auto;">
    图 11：以 Forward+backward 策略为例，其中 Peak Memory 记录了峰值内存占用
</div>

**计算-内存权衡表（以 $N = 4$ 为例）**:

| **策略**                       | **Compute ** | **Peak Memory ** | **计算复杂度**              | **内存复杂度**            |
| ---------------------------- | ------------ | ---------------- | ---------------------- | -------------------- |
| **Forward+backward**         | **8**        | **4**            | $O(N)$                 | $O(N)$               |
| **Full Recomputation**       | **14**       | **1**            | $O(N^2)$               | $O(1)$               |
| **C checkpoints** (C=2)      | **10**       | **4**            | $O(N + \frac{N^2}{C})$ | $O(C + \frac{N}{C})$ |
| ** $\sqrt{N}$ checkpoints ** | **10**       | **4**            | $O(N\sqrt{N})$         | $O(\sqrt{N})$        |

这四种策略代表了深度学习中典型的 **“时间换空间”** 思想：
- **Forward+backward** 速度最快，但内存消耗随层数线性暴增，极易导致显存溢出（OOM）；
- **Full Recomputation** 虽然将中间激活值降到了最低，但其带来的二次方级（$N^2$）计算开销在深层网络中是无法接受的；
- 为了打破这种两极分化，**C checkpoints** 引入了局部重计算来平衡两端；
- 而当分段跨度取最优解 **$\sqrt{N}$ checkpoints** 时，该策略成功达到了最佳折中——它能够以略多于线性的微弱计算代价，换取内存占用从线性级别（$N$）直接断崖式下跌到根号级别（$\sqrt{N}$），是大模型训练中用少量计算耗时对冲海量显存压力的核心利器。

### Scaling Recipe (DP → FSDP → HSDP)

结合 DP、FSDP、HSDP 和激活检查点，可以得到实用的扩展路线：

1. **~128 GPU，~1B 参数**：纯 DP，设置 per-GPU batch size 填满 GPU 内存
2. **模型 >1B 参数**：切换到 FSDP
3. **激活值占满内存**：开启 Activation Checkpointing
4. **>256 GPU**：切换到 HSDP
5. **>1000 GPU，>50B 参数，序列长度 >16K**：需要 CP、PP、TP 等高级策略

---

## Model FLOPs Utilization (MFU)

当并行策略有大量 knobs 需要调节时（batch size、FSDP size、HSDP 维度、checkpoint 粒度……），优化的**唯一指导指标**就是 MFU。

### Hardware FLOPs Utilization (HFU)

H100 理论峰值：989.4 BF16 TFLOP/s。实际能获得多少？

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 13：HFU Benchmark — 大矩阵乘法在 H100 上可达到约 80% HFU</div>
</div>

**HFU = 实际实现的吞吐 / 理论最大吞吐**。大矩阵乘法（~8000×8000）在 H100 上约 80% HFU。但 HFU 不计算激活检查点、数据增强、优化器、通信等开销。

### From HFU to MFU

**MFU = Forward/Backward 理论耗时 / Forward/Backward 实际耗时**

计算步骤：
1. 计算模型 Fwd+Bwd 的理论 FLOPs 总数（可近似 Bwd = 2× Fwd）
2. 查设备理论峰值 FLOP/s
3. $t_{theoretical} = \text{FLOPs}_{model} / \text{FLOP/sec}_{peak}$
4. 实测完整训练迭代耗时（含数据加载、forward、backward、优化器更新）
5. $\text{MFU} = t_{theoretical} / t_{actual}$

**基准**：
- **>30% 是不错的**，**>40% 是优秀的**
- Llama3-405B 在 8000-16000 GPU 上实现 38-41% MFU
- 新设备的 MFU 有时反而更低——A100→H100 算力提升 3.1× 但内存带宽仅提升 2.1×（通信相对变慢）

---

## Context Parallelism (CP)

当序列长度变得很长（如 131K tokens）时，单个 GPU 内存不足以处理。**Context Parallelism** 将序列切分到多个 GPU：

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 14：Context Parallelism — N-way CP 下每个 GPU 处理序列的 1/N 部分</div>
</div>

- **LN / Residual / FFN**：天然独立于序列维度，易于并行（FFN 中的权重梯度需 All-Reduce）
- **QKV 投影**：同 MLP，按序列并行 + 同步梯度
- **Attention 核心**：最难并行——需要每对 token 的 all-pair 交互

两种 Attention 并行方案：

| 方案 | 思路 | 特点 |
|------|------|------|
| **Ulysses** | 按 attention head 维度并行，All-to-All 重分片 | 实现简单，但 head 数需能被 GPU 数整除 |
| **Ring Attention** | 将注意力矩阵分块，GPU 环形传递 K/V 块 | 可扩展到极长序列，实现复杂 |

**Ulysses 的核心操作**：先按 sequence 分片做 QKV 投影，再重新按 head 分片做 attention——本质是对数据做一次"转置"。这需要 **All-to-All**：每个 GPU 把自己的数据切成 N 块，第 i 块发给 GPU i，同时从所有人那里各收一块。

<div style="text-align: center;">
    <img src="Pasted image 20260617121653.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 15：All-to-All — 将每个 GPU 的 tensor 分块并在 GPU 间做"转置"。这是为 Context Parallelism 而生的最晚出现的通信原语，纯重排、不归约</div>
</div>

All-to-All 是四种通信原语中出现最晚的——直到长序列训练需求催生了 Context Parallelism，才需要这种"纯重排、不归约"的操作。

Llama3 预训练实例：
- 第一阶段：seq=8192，无 CP
- 第二阶段：seq=131,072，**16-way CP**（每 GPU 负责 8192 tokens）

---

## Pipeline Parallelism (PP)

Pipeline Parallelism 将网络的**层**分配到不同 GPU——每个 GPU 负责若干连续层的计算。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 16：Pipeline Parallelism 的 Bubble 问题和 Microbatch 解决方案</div>
</div>

**问题：Sequential Dependency → Bubble**。Naïve 实现下，GPU 大部分时间在等待上一阶段的输出——N-way PP 的最大理论 MFU 仅为 $1/N$（如 8-way PP → 12.5%）。

**解决方案：Microbatches**。将大 batch 拆分为多个 microbatches，交错送入 pipeline：

- 4-way PP + 4 microbatches：MFU 从 $1/4 = 25\%$ 提升到 $16/28 \approx 57\%$
- 更多 microbatches → 更高 MFU，但更耗显存（需要保存更多激活值）
- 需要在 microbatch 数量、激活检查点粒度和并行度之间权衡——**最大化 MFU**

---

## Tensor Parallelism (TP)

Tensor Parallelism 将**单个权重矩阵**切分到多个 GPU，通过块矩阵乘法并行计算。

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 17：Tensor Parallelism — 两层连续 TP 的技巧（第一层列分片 + 第二层行分片），避免层间通信</div>
</div>

**基本操作**：$XW = Y$ 被切分为 $X[W_1 W_2 W_3 W_4] = [Y_1 Y_2 Y_3 Y_4]$，每个 GPU 计算一块。

**关键技巧**：连续两个线性层配合使用时，可避免中间通信：
- 第一层：按**列**拆分权重（每个 GPU 输出 $Y_i$）
- 第二层：按**行**拆分权重（每个 GPU 计算 $Y_i U_i$）
- 最终输出 $Z = Y_1U_1 + Y_2U_2 + Y_3U_3 + Y_4U_4$（仅需一次 All-Reduce）

这与 Transformer FFN 中的两层 MLP 完美匹配——因此 TP 特别适合 Transformer 的 MLP 块。

---

## ND Parallelism: Putting It All Together

在实际训练最大规模的模型时，**所有并行策略同时使用**：

<div style="text-align: center;">
    <img src="" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 18：ND Parallelism — Llama3-405B 最大训练配置：TP×8, CP×16, PP×16, DP×8 同时运行在 16,384 GPU 上</div>
</div>

**Llama3-405B 最大训练配置**（16,000 GPU）：

| 并行维度 | 倍数 | 说明 |
|----------|------|------|
| Tensor Parallelism | 8× | 拆分单个权重矩阵 |
| Context Parallelism | 16× | 处理 131K 长序列 |
| Pipeline Parallelism | 16× | 拆分 126 层 |
| Data Parallelism | 8× | 拆分 batch |

总并行度：$8 \times 16 \times 16 \times 8 = 16,384$ GPU（实际约 16,000）

不同的并行策略有不同的通信需求——将它们巧妙排列到集群的拓扑结构上（高速通道放 FSDP/TP，低速通道放 DP），才能最大化 MFU。

### 为什么必须分层混合

不同策略的**通信量**差异巨大：

| 策略 | 通信密集度 | 通信内容 | 带宽需求 |
|------|-----------|---------|---------|
| TP / FSDP | 极高 | 权重分片、梯度归约（每层都通信） | 最高 |
| PP | 中等 | 激活值和梯度（仅层边界通信） | 中等 |
| DP | 较低 | 梯度归约（仅 backward 结束通信） | 可容忍较低 |

恰好，集群拓扑也有天然的**速度层次**：

```
GPU 内部     >   Server 内     >   Pod 内      >   跨 Pod
(3 TB/s)        (900 GB/s)       (50 GB/s)       (<50 GB/s)
  ↑                ↑               ↑               ↑
 NVLink           NVSwitch        InfiniBand      以太网
```

**设计的核心原则**：把通信最密集的策略放在物理距离最近的地方。这也是 HSDP 区分组内/组间通信的深层原因——本质上是**让算法适配硬件拓扑**，而非反过来。

Llama3-405B 的典型映射：
- **TP（8 路）**放在单机 8 GPU 内 → NVLink 最快，权重切碎也能吃得消
- **PP（16 路）**放在同 Pod 内 → 传激活/梯度，中等带宽即可
- **DP（8 路）**跨 Pod → 只传梯度，容忍较低带宽
- **CP（16 路）**与 DP 类似，跨 Pod 部署

---

## Summary

Lecture 11 覆盖了大规模分布式训练的完整技术栈：

**GPU 硬件**：
- H100：132 SM，每 SM 含 128 FP32 Core + 4 Tensor Core
- Tensor Cores 是吞吐量的关键（16× vs FP32），使用混合精度（16-bit in, 32-bit accumulate）
- 12 年间单 GPU 算力提升 ~1000×

**GPU 集群**：
- 存储层次跨越整个集群：GPU 内 3 TB/s → Server 内 900 GB/s → Pod 内 50 GB/s → 跨 Pod <50 GB/s
- Meta Llama3 集群：24,576 H100，24.3 EFLOP/s，视为一台超级计算机

**四种并行策略**：

| 策略 | 拆分维度 | 核心方法 | 通信模式 |
|------|----------|----------|----------|
| **Data Parallelism (DP)** | Batch | All-Reduce 梯度 | 梯度的 1× 通信 |
| **FSDP / HSDP** | Batch + Weights | 权重分片 + 按需广播 | 权重的 3× 通信（组内）|
| **Context Parallelism (CP)** | Sequence | Ulysses / Ring Attention | 注意力计算通信 |
| **Pipeline Parallelism (PP)** | Layers | Microbatches 减少 bubble | 激活值/梯度传递 |
| **Tensor Parallelism (TP)** | Dim (Channel) | 块矩阵乘法 | 双层技巧减少通信 |

**关键工具**：
- **Activation Checkpointing**：$O(N\sqrt{N})$ compute + $O(\sqrt{N})$ memory（$C=\sqrt{N}$ 时）
- **MFU**：优化分布式训练的唯一指导指标，>30% good，>40% excellent

**终极方案 → ND Parallelism**：所有策略组合，Arranged by network topology to maximize MFU。

## Materials

- [The Llama 3 Herd of Models (Dubey et al., arXiv 2024)](https://arxiv.org/abs/2407.21783)
- [GPT-4 Technical Report (OpenAI, arXiv 2023)](https://arxiv.org/abs/2303.08774)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., arXiv 2019)](https://arxiv.org/abs/1910.02054)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Huang et al., arXiv 2018)](https://arxiv.org/abs/1811.06965)
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., arXiv 2022)](https://arxiv.org/abs/2204.02311)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context (Liu et al., arXiv 2023)](https://arxiv.org/abs/2310.01889)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models (Jacobs et al., arXiv 2023)](https://arxiv.org/abs/2309.14509)
- [NVIDIA NCCL Collective Operations Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [CS231n 2025/2026 Lecture 11 Slides](https://cs231n.stanford.edu/slides/2025/lecture_11.pdf)
