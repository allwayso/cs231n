---
title: "Lecture 08: Attention and Transformers"
publish: true
---
>[!SUMMARY] Table of Contents
>    - [[lecture08 Attention and Transformers#Seq2seq example|Seq2seq example]]
>        - [[lecture08 Attention and Transformers#seq2seq with RNNs|seq2seq with RNNs]]
>        - [[lecture08 Attention and Transformers#Seq2seq with Attention|Seq2seq with Attention]]
>    - [[lecture08 Attention and Transformers#From Seq2seq Attention to General Attention Layer|From Seq2seq Attention to General Attention Layer]]
>        - [[lecture08 Attention and Transformers#1. Single Query Attention|1. Single Query Attention]]
>        - [[lecture08 Attention and Transformers#2. Multiple Queries|2. Multiple Queries]]
>        - [[lecture08 Attention and Transformers#3. Saperate X into Keys and Values|3. Saperate X into Keys and Values]]
>    - [[lecture08 Attention and Transformers#Different forms of self-attention|Different forms of self-attention]]
>        - [[lecture08 Attention and Transformers#Positional Encoding or Rotary Position Embedding|Positional Encoding or Rotary Position Embedding]]
>        - [[lecture08 Attention and Transformers#Masked Self-Attention|Masked Self-Attention]]
>        - [[lecture08 Attention and Transformers#Multi-Head Self-Attention|Multi-Head Self-Attention]]
>        - [[lecture08 Attention and Transformers#Only Matrix Multiplies|Only Matrix Multiplies]]
>    - [[lecture08 Attention and Transformers#The Transformer|The Transformer]]
>        - [[lecture08 Attention and Transformers#Stacking Transformer Blocks|Stacking Transformer Blocks]]
>        - [[lecture08 Attention and Transformers#Transformers for Language Modeling|Transformers for Language Modeling]]
>        - [[lecture08 Attention and Transformers#Vision Transformers|Vision Transformers]]
>    - [[lecture08 Attention and Transformers#Summary|Summary]]
>    - [[lecture08 Attention and Transformers#Materials|Materials]]


## Seq2seq example

seq2seq 任务指的是将一个输入序列映射为另一个输出序列的任务，例如机器翻译、文本摘要、语音识别等。它的核心特点是输入和输出都是序列，并且二者长度可以不同，因此模型需要先理解整个输入序列，再逐步生成目标序列
### seq2seq with RNNs

<div style="text-align: center;">
    <img src="Pasted image 20260601163921.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1： seq2seq with RNNs</div>
</div>

简要解释：

1. 输入序列经过 Encoder 后得到上下文向量 c （通常等于 $h_T$ ）和 Decoder 初始状态 s0（由 $h_T$ 变换得到）
2. Decoder 递归计算得到输出序列

> 为什么这里使用 Encoder + Decoder ？
> 因为 seq2seq 中输出序列往往与输入序列长度并不一致，所以使用 Encoder 理解英文输入 ，再通过 Decoder 把理解的内容输出为意大利语翻译

当前的结构有一个问题，Decoder 递归输出的过程依赖于固定的上下文向量 c，这在数据规模较小的时候是可以接受的，但当输入序列增长的时候，c 往往不能够概括输入序列的内容 ，这就会导致信息瓶颈（Information bottlenecks）。

> 那把上下文向量由定长换成变长不就行了？
> 诚然，变长向量能够随着输入序列的增长而增长，从而避免了信息缺失的问题，但是信息瓶颈并不是 RNN seq2seq 的唯一问题。更大的问题是，无论上下文向量定长还是变长，Decoder 解码的时候都需要重点关注最相关内容，而不是整个向量。换句话说，我们需要一个可微分的寻址功能，这也就引入了注意力机制。

### Seq2seq with Attention

<div style="text-align: center;">
    <img src="Pasted image 20260601173014.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：Seq2seq with RNNs and Attention</div>
</div>

正如上一节所讨论的，Decoder 在生成当前输出时，需要从 Encoder 的所有隐藏状态中找到与当前解码状态最相关的信息。因此，需要一种机制来衡量 Decoder 当前状态与各个 Encoder 隐藏状态之间的相关程度。

为此，我们定义对齐分数（alignment score）

 $$  
 e_{t,i}=f_{\text{att}}(s_{t-1},h_i),  
$$

其中，$s_{t-1}$ 表示 Decoder 在前一时刻的隐藏状态，$h_i$ 表示 Encoder 的第 $i$ 个隐藏状态。函数 $f_{\text{att}}$ 用于计算二者之间的匹配程度，并输出一个标量作为相关性得分。得分越高，说明当前解码状态与对应输入位置的关联越强。

> $f_{\text{att}}$ 如何计算？
> $f_{\text{att}}$ 可以由神经网络实现，但实际上采用最简单的向量点积即可。当 $q,X_i\in\mathbb{R}^{D_X}$ 时，相似度可写为 $e_i=q\cdot X_i$。点积越大，说明两个向量方向越接近。但维度 $D_X$ 较大时，点积数值容易过大，使 softmax 饱和、梯度变小。因此通常使用 scaled dot-product attention：$e_i=\frac{q\cdot X_i}{\sqrt{D_X}}$ , 其中 $\sqrt{D_X}$ 用来稳定数值尺度。

由于这些得分并不满足概率分布的性质，因此进一步通过 softmax 函数进行归一化：

$$  
 a_{t,i}=\frac{\exp(e_{t,i})}  {\sum_j \exp(e_{t,j})}.  
$$
 归一化后的 $a_{t,i}$ 称为注意力权重（attention weight），表示在生成第 $t$ 个输出时，Decoder 对 Encoder 第 $i$ 个隐藏状态的关注程度。

最后，利用这些注意力权重对所有 Encoder 隐藏状态进行加权求和，得到当前时刻的上下文向量： 
$$  
c_t\sum_i a_{t,i} h_i  
$$
上下文向量 $c_t$ 汇聚了当前解码状态最关注的输入信息，因此能够为 Decoder 生成当前输出提供更加有针对性的上下文表示。

> 引入注意力机制的神经网络需要数据标注这种相关性作为先验信息吗？
> 不需要。整个计算过程都是可微的，意味着可以采用在全过程采用梯度下降法进行监督学习，神经网络可以自己学习相关性。


<div style="text-align: center;">
    <img src="Pasted image 20260601205932.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：Trying to understand visualized attention weights</div>
</div>

通过可视化注意力权重，我们注意到几个有代表性的区域：
1. 左上角和右下角的蓝紫色区域：呈现明显的对角线，表明输入与输出是按原顺序一对一直译的
2. 中部靠左的黄色区域：呈现明显的非对角线，表明输入与输出虽然一一对应，但并不是按照原顺序直译，说明神经网络学习到了某种法语语法
3. 中间靠右的未框选区域：权重图较为模糊，表明输入与输出并不是一一对应的，或许是 two2two 词组对应

这表明尽管我们没有提供任何先验的语法资料，带有注意力的神经网络仍然能够从大量的输入输出对中学习到语法信息。

## From Seq2seq Attention to General Attention Layer

Attention 的本质是：给定一个 query 和一组 data vectors，先计算相关性，再按相关性加权求和得到输出。

在 seq2seq attention 中，query 是 Decoder state，data vectors 是 Encoder states，输出是 context vector $c_t$。然而 Attention 不只是机器翻译或 RNN 的技巧，而是一个通用的向量集合读取操作，所以在本节，我们将从 seq2seq 拓展到一般的注意力模型。
### 1. Single Query Attention

<div style="text-align: center;">
    <img src="Pasted image 20260601231753.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：Single Query Attention</div>
</div>

单个查询向量很好理解，其实与 rnn+attention 并没有任何区别，只是把英文输入泛化为输入向量，并把所有运算转变为矩阵运算。

### 2. Multiple Queries

<div style="text-align: center;">
    <img src="Pasted image 20260601232527.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：Attention layer with multiple queries</div>
</div>

如果有多个 queries，记 $Q\in\mathbb{R}^{N_Q\times D_X}$，data vectors 为 $X\in\mathbb{R}^{N_X\times D_X}$。可以一次性计算所有 query 和 data vectors 的相似度矩阵：

$$
E=\frac{QX^T}{\sqrt{D_X}},\quad A=\text{softmax}(E,\text{dim}=1),\quad Y=AX
$$

其中 $E,A\in\mathbb{R}^{N_Q\times N_X}$，$Y\in\mathbb{R}^{N_Q\times D_X}$，并且 $Y_i=\sum_j A_{i,j}X_j$。

> 为什么感觉这里和之前的 seq2seq 模型完全不一样了？
> 从 Single Query Attention 到 Multiple Queries ，看似只是像 CNN 等神经网络中为了矩阵运算方便，把向量组合为矩阵而已，其实暗中发生了一次抽象——之前的查询 $q_{t}$ 其实仍然依赖于 $q_{t-1}$ ，但是当查询向量变为矩阵后，其实已经抛弃了递归依赖关系，每一次查询之间都是相互独立的，也就是 RNN 的思想被完全剥离了。

### 3. Saperate X into Keys and Values

前面的 $X_i$ 同时承担两个角色：一是和 query 计算相似度，二是被加权求和形成输出。实际中可以把这两个角色拆开：key 用于匹配，value 用于输出。

给定 $X\in\mathbb{R}^{N_X\times D_X}$，通过线性变换得到 $K=XW_K$ 和 $V=XW_V$，其中 $W_K\in\mathbb{R}^{D_X\times D_Q}$，$W_V\in\mathbb{R}^{D_X\times D_V}$。

核心计算变为：

$$
E=\frac{QK^T}{\sqrt{D_Q}},\quad A=\text{softmax}(E,\text{dim}=1),\quad Y=AV
$$

也就是 $Y_i=\sum_j A_{i,j}V_j$。这样模型可以分别学习 “ 用什么信息匹配 query ” 和 “ 把什么信息传递到输出 ”。

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <div style="text-align: center;">
    <img src="Pasted image 20260601234342.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 6：Cross-attention </div>
</div>
    <div style="text-align: center;">
    <img src="Pasted image 20260601235930.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：Cross-attention layer</div>
</div>
</div>

像图6这样，输入和查询序列不同的模型，就叫做交叉注意力（Cross-attention），实际上就是用一个序列去查另一个序列；有的时候我们只需要当前序列上下文，通过把输入矩阵再映射一个矩阵到查询矩阵中，这就是自注意力（Self-attention）模型。

> Self-Attention 本质上作用在一组向量集合上，而不是天然作用在有序序列上。如果对输入 $X$ 做排列 $\sigma$，输出也会做相同排列：$F(\sigma(X))=\sigma(F(X))$ 
>这称为置换等变性（permutation equivariance），其定义为：输入元素打乱顺序，模型输出的对应元素也同步打乱相同顺序。

## Different forms of self-attention

### Positional Encoding or Rotary Position Embedding

尽管我们得到了更快的矩阵运算，但是抛弃 RNN是有代价的 —— Self-Attention 本身不知道序列顺序。例如 “dog bites man” 和 “man bites dog” 包含相同单词，但语义完全不同。

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260602000827.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：Adding positional encoding to self-attention</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260602000946.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：Adding RoPE to self-attention</div>
    </div>
</div>


为了解决顺序问题，需要给输入加入位置信息。最直接的方法是给第 $i$ 个输入向量加位置编码 $p_i$，即 $\tilde{x}_i=x_i+p_i$，整体写作 $\tilde{X}=X+P$。

位置编码告诉模型 token 的绝对或相对位置，使 Self-Attention 可以处理有序序列。

或者采用旋转位置编码（Rotary Position Embedding）。RoPE 不是直接把位置向量加到输入上，而是根据位置旋转 query 和 key，使相似度天然包含相对位置信息。

$$
(R(\theta_i)q_i)^T(R(\phi_j)k_j)=q_i^T R(\phi_j-\theta_i)k_j
$$

因此，旋转后的 attention similarity 不只取决于内容，也取决于相对位置 $\phi_j-\theta_i$。普通 positional encoding 更像告诉模型“我在哪里”，RoPE 更像在相似度里编码“我和你相隔多远”。

### Masked Self-Attention

<div style="text-align: center;">
    <img src="Pasted image 20260602001614.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 10：Masked self-attention layer</div>
</div>

语言模型预测下一个 token 时，当前位置不能看到未来 token。因此需要在 softmax 前把不允许关注的位置设为 $-\infty$。

对于第 $i$ 个位置，只允许关注 $j\leq i$：

$$
E_{i,j}=\begin{cases}\frac{Q_i\cdot K_j}{\sqrt{D}}, & j\leq i \\ -\infty, & j>i\end{cases}
$$

softmax 后，未来位置的权重变成 $A_{i,j}=0$。Masked Self-Attention 是自回归语言模型的核心：训练时可以并行处理整句话，但每个位置都不能偷看未来。

### Multi-Head Self-Attention

<div style="text-align: center;">
    <img src="Pasted image 20260602001942.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 11：Multi-head self-attention layer</div>
</div>

单个 Self-Attention 只有一套 $W_Q,W_K,W_V$，只能用一种方式建模 token 关系。Multi-Head Self-Attention 并行运行 $H$ 个 heads，每个 head 有自己的 $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$，可以从不同子空间捕捉关系。

第 $h$ 个 head：

$$
A^{(h)}=\text{softmax}\left(\frac{Q^{(h)}(K^{(h)})^T}{\sqrt{D_H}}\right),\quad Y^{(h)}=A^{(h)}V^{(h)}
$$

然后拼接所有 heads，并用输出投影融合：

$$
Y=\text{Concat}(Y^{(1)},\dots,Y^{(H)}),\quad O=YW_O
$$

通常设 $D_H=D/H$，这样拼接后维度仍为 $D$。

### Only Matrix Multiplies

实际运算中，Multi-Head Self-Attention 可以概括为四次主要矩阵乘法。

<div style="text-align: center;">
    <img src="Pasted image 20260602002701.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 12：4 matrix to calculate self-attention</div>
</div>

设输入 $X\in\mathbb{R}^{N\times D}$，$N$ 是序列长度，$D$ 是模型维度。

1. QKV Projection

一次性计算所有 heads 的 $Q,K,V$：$[Q,K,V]=X[W_Q,W_K,W_V]$
形状为 $[N\times D][D\times 3HD_H]\rightarrow[N\times 3HD_H]$，再 reshape 得到 $Q,K,V\in\mathbb{R}^{H\times N\times D_H}$。

2. QK Similarity：计算相似度 $E=QK^T/\sqrt{D_H}$，形状为 $E\in\mathbb{R}^{H\times N\times N}$。

3. V-Weighting：对 $E$ 做 softmax 得到 $A=\text{softmax}(E,\text{dim}=2)$，再计算 $Y=AV$。此时 $Y\in\mathbb{R}^{H\times N\times D_H}$，随后 reshape 为 $Y\in\mathbb{R}^{N\times HD_H}$。

4. Output Projection：最后通过输出矩阵投影回模型维度：$O=YW_O$ , 其中 $W_O\in\mathbb{R}^{HD_H\times D}$，$O\in\mathbb{R}^{N\times D}$。

> $O(N^2)$ 不会导致计算量爆炸吗？
> 在 transformer 提出的初期，由于主要任务为自然语言处理（NLP），上下文窗口较小， $O(N^2)$ 的计算量是可以接受的。而且虽然复杂度较高，但是 transformer 相比于 RNN而言，有以下几点优势：
> 1. 并行计算远快于串行运算
> 2. 长距离依赖的距离更短：以 token1 为例 ，transformer 的查询开销为 O(1)，而 RNN 的开销为 O(N)
> 而当上下文窗口越来越大时，比如 GPT-3 的 32k 上下文，存储复杂度 $O(N^2)$ 的开销大于了并行带来的加速，所以现代 transformer 基本上都是在优化如何不完整存储 $N^2$ 矩阵 



## The Transformer

<div style="text-align: center;">
    <img src="Pasted image 20260602114539.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 13：Transformer block</div>
</div>

Transformer 的基本单位是 Transformer Block，输入一组向量 $x_1,\dots,x_N$，输出一组向量 $y_1,\dots,y_N$。一个 block 通常包含 Multi-Head Self-Attention、Residual Connection、LayerNorm、MLP/FFN，再加一组 residual 和 LayerNorm。

Tranformer Block中各个部件的作用：
1. Self-Attention 是 block 中唯一让不同 token 交互的部分
2. MLP 是逐 token 独立作用的，负责非线性变换，通常是两层全连接网络 $D\rightarrow4D\rightarrow D$，即 $\text{MLP}(x)=W_2\sigma(W_1x)$
3. Residual Connection 保留原输入并学习增量更新
4. LayerNorm 稳定训练

经典 Post-Norm 写法可以概括为：

$$
X_1=\text{LayerNorm}(X+\text{SelfAttention}(X)),\quad Y=\text{LayerNorm}(X_1+\text{MLP}(X_1))
$$

现代实现常用 Pre-Norm，训练更稳定：

$$
X_1=X+\text{SelfAttention}(\text{LayerNorm}(X)),\quad Y=X_1+\text{MLP}(\text{LayerNorm}(X_1))
$$

Transformer 的核心：Self-Attention 负责 token 间通信，MLP 负责 token 内部变换，Residual 和 LayerNorm 负责稳定训练。

### Stacking Transformer Blocks

<div style="text-align: center;">
    <img src="Pasted image 20260602114915.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 14：A Transformer is a stack of Transformer blocks</div>
</div>

完整 Transformer 是多个 Transformer Blocks 的堆叠：$X^{(0)}=X$，$X^{(l+1)}=\text{TransformerBlock}^{(l)}(X^{(l)})$，最终 $Y=X^{(L)}$。

堆叠多层后，模型可以逐层构建更复杂的表示。Transformer 自 2017 年以来核心结构变化不大，但规模不断增大：更多层、更大隐藏维度、更多 heads、更长上下文。

### Transformers for Language Modeling

<div style="text-align: center;">
    <img src="Pasted image 20260602115101.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 15：Transformers for language modeling</div>
</div>

语言模型的目标是根据前面的 tokens 预测下一个 token。首先用 embedding matrix $E_{\text{embed}}\in\mathbb{R}^{V\times D}$ 把 token id 映射成 $D$ 维向量，得到输入 $X\in\mathbb{R}^{N\times D}$。

之后将 $X$ 输入多个带 Masked Self-Attention 的 Transformer Blocks，保证第 $i$ 个 token 只能看到之前的 tokens。最后用 projection matrix $W_{\text{out}}\in\mathbb{R}^{D\times V}$ 映射回 vocabulary scores：

$$
S=YW_{\text{out}},\quad p(y_i\mid x_{\leq i})=\text{softmax}(S_i)
$$

训练时使用 cross-entropy loss 预测下一个 token。

### Vision Transformers

<div style="text-align: center;">
    <img src="Pasted image 20260602115336.png" width="800" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 16：Vision Transformer pipeline</div>
</div>

Vision Transformer，简称 ViT，把图像切成 patches，并把每个 patch 当作一个 token。例如输入图像为 $224\times224\times3$，patch size 为 $16\times16\times3$，每个 patch 展平后维度为 $16\times16\times3=768$，再通过线性层映射到 $D$ 维。

对于 $224\times224$ 图像和 $16\times16$ patch，共有 $14\times14=196$ 个 patches，因此 Transformer 的序列长度为 $196$。

Patch embedding 也可以看作一个卷积操作：kernel size 为 $16\times16$，stride 为 $16$，输入通道为 $3$，输出通道为 $D$。ViT 还需要 positional encoding 来表示每个 patch 的二维位置。

与语言模型不同，图像分类通常不需要 mask：每个 image patch 可以看见所有其他 patches，即使用普通 Self-Attention。

## Summary

从 RNN seq2seq 到 Transformer，可以看成逐级泛化：
1. 固定 context vector 产生瓶颈；
2. seq2seq attention 让 Decoder 动态读取 Encoder states；
3. general attention 抽象为 query 对 data vectors 的读取；
4. scaled dot-product attention 用矩阵乘法高效实现；
5. cross-attention 让一个序列查询另一个序列；
6. self-attention 让同一序列内部 tokens 相互查询；
7. positional encoding / RoPE 补充顺序信息；
8. masked self-attention 支持自回归语言建模；
9. multi-head attention 从多个子空间建模关系；
10. Transformer Block 将 attention、MLP、residual 和 LayerNorm 组合起来。

Transformer 的核心是 Self-Attention，它让序列中每个 token 能够直接访问和整合所有其他 token 的信息；通过 堆叠多个 Transformer Blocks，模型逐层构建更复杂的表示。同时，每个 token 内部的 MLP 层 提供非线性变换，残差连接和 LayerNorm 保证训练稳定和信息流通 。

| 模型结构           | 适用场景          | 优势                  | 劣势                      | 时间/内存复杂度 |
| -------------- | ------------- | ------------------- | ----------------------- | -------- |
| RNN            | 一维有序序列        | 天然适配时序数据            | 无法并行，隐藏状态必须按时间步依次计算     | $O(N)$   |
| Convolution    | 规则网格数据、局部特征提取 | 支持并行计算，擅长捕捉局部模式     | 单层感受野有限，建模长距离依赖需堆叠大量网络层 | -        |
| Self-Attention | 存在长距离依赖的序列数据  | 可直接关联全局所有token，并行度高 | 标准实现存在二次方复杂度，注意力矩阵占用内存大 | $O(N^2)$ |

## Materials

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [CS231n 2024/2025 Lecture 8 Slides](https://cs231n.stanford.edu/slides/2025/lecture_8.pdf)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)