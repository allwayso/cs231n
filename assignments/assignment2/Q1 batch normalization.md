### 1. BN forward 和 backprop 数学推导

这个讲义从最简单的双变量一维x输入为例，将链式求导结果推广到N变量D维x矩阵输入：[Aditya Agrawal --- Back Propagation in Batch Normalization Layer | Aditya Agrawal](https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm)

我觉得这是一个不错的思路，尤其是对于矩阵求导来说，直接拿着矩阵推很容易搞晕了，这个思路很好，像下图这样的计算图对于链式求导来说也很实用。

![[反向传播的计算图.png]]

### 2. BN对超参数选择的影响

|维度|无 BN|有 BN|
|---|---|---|
|**初始化敏感度**|非常高。weight_scale 需要精心调优，偏差 1 个数量级就可能导致梯度消失或爆炸|非常低。weight_scale 在 2-3 个数量级范围内都能正常工作|
|**有效初始化范围**|窄（如 1e-3 附近的一个小窗口）|宽（如 1e-4 到 1e-1 连续可行）|
|**Xavier/He 必要性**|必须使用精心设计的初始化策略|即使使用简单的随机初始化也能收敛|

**原因**：BN 对输入做标准化，无论前一层权重多大或多小，输出都被拉回均值为 0、方差为 1 的分布，打破了层与层之间参数 scale 的连锁放大效应。

|维度|无 BN|有 BN|
|---|---|---|
|**可用的 LR 范围**|窄。LR 太大容易震荡/发散，太小收敛极慢|宽。可以安全使用比无 BN 大 **5-10 倍** 甚至更高的学习率|
|**最优 LR**|需要精细搜索|可以在较大范围内取一个值，表现仍然稳定|
|**LR schedule 鲁棒性**|低，衰减策略错了容易炸|高，对衰减策略不敏感|

**原因**：BN 使损失函数的景观更平滑（Santurkar et al., 2018 证明了 BN 降低了损失函数的 Lipschitz 常数），使得大学习率不会导致梯度步长过大而跳出优化区域。此外，BN 引入的 γ,β 参数还能自动补偿梯度的 scale 变化，相当于对每层的有效学习率做了自动调节。

> 注意：此处原论文提出的论点被证伪，并不是因为Reducing Internal Covariate Shift（减少内部协变量偏移）

|维度|无 BN|有 BN|
|---|---|---|
|**对 batch size 的依赖**|相对较小（只要不极端）|**大**：BN 的统计量估计质量高度依赖 batch size|
|**小 batch（如 4, 8）**|可以训练，梯度噪声大但仍可收敛|**表现明显下降**。小 batch 对均值和方差的估计不准，导致训练不稳定|
|**大 batch（如 128, 256）**|需要适当调大 LR|表现很好，统计量估计更准确|

**这是 BN 的主要弱点**：当 batch size 很小时（如目标检测中常用的 batch size=2），BN 效果急剧下降。这也催生了后续的 **Layer Normalization、Group Normalization、Instance Normalization** 等替代方案。

### 3. layer normalization

与batch normalization基本一致，只是σ和μ的运算方向调整一下就行，反向传播也基本一致，不再展开
