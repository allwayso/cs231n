# CNN反向传播的维度推导

## 维度思考(~~31051原创思路~~)

> 所有矩阵运算推导都可以先从维度入手，维度正确是必要条件

设输入 $X \in \mathbb{R}^{H \times W}$，卷积核 $F \in \mathbb{R}^{K \times K}$，步长 1，无padding。前向 valid 卷积：

$O = X * F, \quad O \in \mathbb{R}^{(H-K+1) \times (W-K+1)}$

已知 $\frac{\partial L}{\partial O} \in \mathbb{R}^{(H-K+1) \times (W-K+1)}$，求 $\frac{\partial L}{\partial X}$ 与 $\frac{\partial L}{\partial F}$。

---

### 1. 求 $\frac{\partial L}{\partial F}$

**目标尺寸：** $K \times K$

**约束：** 卷积结果的尺寸公式 $n_{out} = n_{in} - k + 1$（valid 卷积下）。令 $n_{in} = H$，$n_{out} = K$，解得卷积核尺寸应为 $H - K + 1$。

**匹配：** $\frac{\partial L}{\partial O}$ 的尺寸恰为 $H - K + 1$。以 $\frac{\partial L}{\partial O}$ 为卷积核、$X$ 为输入做 valid 卷积：

$\text{dim}\left(X *_{valid} \frac{\partial L}{\partial O}\right) = (H - (H - K + 1) + 1) \times (W - (W - K + 1) + 1) = K \times K$

由链式法则逐元素展开即得等式：

$\boxed{\frac{\partial L}{\partial F} = X \; *_{valid} \; \frac{\partial L}{\partial O}}$

---

### 2. 求 $\frac{\partial L}{\partial X}$

**目标尺寸：** $H \times W$

**约束：** 以 $F$ 为卷积核对 $\frac{\partial L}{\partial O}$ 做 valid 卷积，得：

$\dim\left(\frac{\partial L}{\partial O} * F\right) = (H - 2K + 2) \times (W - 2K + 2) \neq H \times W$

引入零填充。在 $\frac{\partial L}{\partial O}$ 四周各补 $K-1$ 层零，记作 $\text{pad}_{K-1}\!\left(\frac{\partial L}{\partial O}\right)$，尺寸变为：

$(H - K + 1 + 2(K-1)) \times (\cdots) = (H + K - 1) \times (W + K - 1)$

再以 $\operatorname{rot180}(F)$ 为卷积核做 valid 卷积：

$\dim\left(\text{pad}_{K-1}\!\left(\frac{\partial L}{\partial O}\right) * \operatorname{rot180}(F)\right) = (H + K - 1 - K + 1) \times (\cdots) = H \times W$

尺寸匹配。由链式法则验证权重对应关系即得：

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot180}(F) \; *_{valid} \; \text{pad}_{K-1}\!\left(\frac{\partial L}{\partial O}\right)}$

或等价地以 full convolution 记号表示为：

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot180}(F) \; \otimes_{full} \; \frac{\partial L}{\partial O}}$

---

### 3. 汇总

| 梯度 | 表达式 | 卷积模式 |
|------|--------|---------|
| $\frac{\partial L}{\partial F}$ | $X * \frac{\partial L}{\partial O}$ | valid（无padding） |
| $\frac{\partial L}{\partial X}$ | $\operatorname{rot180}(F) \otimes_{full} \frac{\partial L}{\partial O}$ | full（pad $K-1$ 层零） |

## 从 $X_{3\times3}, F_{2\times2}, S=1$ 推广至一般卷积的反向传播

---

### 0. 基础情形

**给定：**

$X = \begin{bmatrix} x_{00} & x_{01} & x_{02} \\ x_{10} & x_{11} & x_{12} \\ x_{20} & x_{21} & x_{22} \end{bmatrix}, \quad F = \begin{bmatrix} f_{00} & f_{01} \\ f_{10} & f_{11} \end{bmatrix}, \quad S=1, \quad P=0$

**前向：**

$O = X * F, \quad O_{ij} = \sum_{p=0}^{1}\sum_{q=0}^{1} x_{i+p,\,j+q} \cdot f_{pq}, \quad i,j \in \{0,1\}$

即 $O \in \mathbb{R}^{2 \times 2}$。

**已知** $\frac{\partial L}{\partial O} \in \mathbb{R}^{2 \times 2}$，**求** $\frac{\partial L}{\partial F} \in \mathbb{R}^{2 \times 2}$ 与 $\frac{\partial L}{\partial X} \in \mathbb{R}^{3 \times 3}$。

---

#### 0.1 对滤波器权重的梯度

由链式法则：

$\frac{\partial L}{\partial f_{pq}} = \sum_{i=0}^{1}\sum_{j=0}^{1} \frac{\partial L}{\partial o_{ij}} \cdot x_{i+p,\,j+q}$

逐项写出（以 $f_{00}$ 为例）：

$\frac{\partial L}{\partial f_{00}} = x_{00}\frac{\partial L}{\partial o_{00}} + x_{01}\frac{\partial L}{\partial o_{01}} + x_{10}\frac{\partial L}{\partial o_{10}} + x_{11}\frac{\partial L}{\partial o_{11}}$

观察：此式即 $\frac{\partial L}{\partial O}$ 作为卷积核在 $X$ 上做 valid 卷积（步长 1）：

$\boxed{\frac{\partial L}{\partial F} = X * \frac{\partial L}{\partial O}}$

维度验证：$3 - 2 + 1 = 2 = \dim(F)$。

---

#### 0.2 对输入的梯度

$\frac{\partial L}{\partial x_{rs}} = \sum_{(i,j):\; x_{rs} \in \text{receptive}(o_{ij})} \frac{\partial L}{\partial o_{ij}} \cdot f_{r-i,\,s-j}$

逐项写出（中间像素 $x_{11}$ 参与全部 4 个输出）：

$\frac{\partial L}{\partial x_{11}} = f_{11}\frac{\partial L}{\partial o_{00}} + f_{10}\frac{\partial L}{\partial o_{01}} + f_{01}\frac{\partial L}{\partial o_{10}} + f_{00}\frac{\partial L}{\partial o_{11}}$

完整 9 个位置的结果等价于：

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot}_{180}(F) \; *_{valid} \; \text{pad}_{1}\!\left(\frac{\partial L}{\partial O}\right)}$

其中 $\operatorname{rot}_{180}(F) = \begin{bmatrix} f_{11} & f_{10} \\ f_{01} & f_{00} \end{bmatrix}$，$\text{pad}_1$ 在 $\frac{\partial L}{\partial O}$ 四周各补一层 0。维度验证：$(4 - 2 + 1) \times (4 - 2 + 1) = 3 \times 3$。

---

### 1. 推广一：一般尺寸，$S=1, P=0$

**设：** $X \in \mathbb{R}^{H \times W}$，$F \in \mathbb{R}^{K \times K}$，$S=1$，$P=0$。

**前向：**

$O_{ij} = \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} x_{i+p,\,j+q} \cdot f_{pq}, \quad \begin{cases} 0 \leq i < H-K+1 \\ 0 \leq j < W-K+1 \end{cases}$

$O \in \mathbb{R}^{(H-K+1) \times (W-K+1)}$

**滤波器梯度：**

$\frac{\partial L}{\partial f_{pq}} = \sum_{i=0}^{H-K}\sum_{j=0}^{W-K} \frac{\partial L}{\partial o_{ij}} \cdot x_{i+p,\,j+q}$

$\boxed{\frac{\partial L}{\partial F} = X * \frac{\partial L}{\partial O}}$

**输入梯度：**

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot}_{180}(F) \; *_{valid} \; \text{pad}_{K-1}\!\left(\frac{\partial L}{\partial O}\right)}$

---

### 2. 推广二：引入步长 $S > 1$

**前向：**

$O_{ij} = \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} x_{Si+p,\, Sj+q} \cdot f_{pq}, \quad O \in \mathbb{R}^{H_{out} \times W_{out}}, \quad H_{out} = \left\lfloor\frac{H-K}{S}\right\rfloor + 1$

与 $S=1$ 的关键差异在于 $x$ 的索引步进了 $S$。

**滤波器梯度：**

$\frac{\partial L}{\partial f_{pq}} = \sum_{i=0}^{H_{out}-1}\sum_{j=0}^{W_{out}-1} \frac{\partial L}{\partial o_{ij}} \cdot x_{Si+p,\, Sj+q}$

若直接视作卷积，则 $\frac{\partial L}{\partial O}$ 在 $X$ 上以步长 $S$ 跳跃式采样。为规约为步长-1 卷积，引入 **膨胀算子**：

$\text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)_{u,v} = \begin{cases} \frac{\partial L}{\partial o_{u/S,\,v/S}} & \text{if } S \mid u \text{ and } S \mid v \\ 0 & \text{otherwise} \end{cases}$

即在 $\frac{\partial L}{\partial O}$ 的行/列间各插入 $S-1$ 个零，其尺寸变为：

$(S(H_{out}-1)+1) \times (S(W_{out}-1)+1)$

以膨胀后的矩阵为卷积核与 $X$ 做步长-1 卷积：

$\boxed{\frac{\partial L}{\partial F} = X * \text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)}$

**输入梯度：**

类似地，$\frac{\partial L}{\partial X}$ 需要先将 $\frac{\partial L}{\partial O}$ 膨胀以恢复前向采样步距，再填充以匹配输出尺寸：

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot}_{180}(F) * \text{pad}_{?}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)\right)}$

填充层数由维度约束确定：经膨胀后尺寸为 $S(H_{out}-1)+1$，需填充至 $H+K-1$ 方能输出 $H$。解得需补 $K-1$ 层（与 $S=1$ 情形等量填充），统一记为 $\text{pad}_{K-1}$。

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot}_{180}(F) *_{valid} \text{pad}_{K-1}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)\right)}$

---

### 3. 推广三：引入填充 $P$

**前向（padding $P$ 层零后卷积）：**

$O = \text{pad}_P(X) *_{S} F, \quad H_{out} = \left\lfloor\frac{H + 2P - K}{S}\right\rfloor + 1$

填充仅影响输入边界，不改变卷积运算本身的结构。反向传播形式上**与之相同**，仅需将前向 padding 的边界效应传递回 $X$ 的梯度（即对 $\frac{\partial L}{\partial X}$ 去掉 padding 对应的边界梯度即可）。

$\boxed{\frac{\partial L}{\partial F} = \text{pad}_P(X) * \text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)}$

$\boxed{\frac{\partial L}{\partial X} = \text{crop}_P\!\left(\operatorname{rot}_{180}(F) *_{valid} \text{pad}_{K-1}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)\right)\right)}$

其中 $\text{crop}_P(\cdot)$ 裁去外围 $P$ 层，恢复原始输入尺寸 $H \times W$。

---

### 4. 推广四：引入膨胀系数 $D$（Dilated Convolution）

**前向（卷积核膨胀，膨胀率 $D$）：**

$O_{ij} = \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} x_{Si + pD,\, Sj + qD} \cdot f_{pq}$

**反向传播：** 膨胀卷积等价于用膨胀后的 $F$ 做标准卷积。因此 $\frac{\partial L}{\partial F}$ 与 $\frac{\partial L}{\partial X}$ 的公式中，膨胀算子作用域扩展为同时作用于 $\frac{\partial L}{\partial O}$（处理步长 $S$）与 $F$（处理膨胀率 $D$）。形式上：

$\boxed{\frac{\partial L}{\partial F} = X * \text{dil}_S\!\left(\frac{\partial L}{\partial O}\right) \quad \text{（对 } F \text{ 的梯度取膨胀后对应位置）}}$

$\boxed{\frac{\partial L}{\partial X} = \operatorname{rot}_{180}\!\left(\text{dil}_D(F)\right) *_{valid} \text{pad}_{\text{eff}}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O}\right)\right)}$

其中 $\text{dil}_D(F)$ 将 $F$ 的行/列间插入 $D-1$ 个零，有效卷积核尺寸变为 $K_{\text{eff}} = (K-1)D + 1$，填充量随之用 $K_{\text{eff}}$ 计算。

---

### 5. 推广五：多通道

**设：** $X \in \mathbb{R}^{C_{in} \times H \times W}$，$F \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$。

**前向（对第 $c_o$ 个输出通道）：**

$O_{c_o} = \sum_{c_i=0}^{C_{in}-1} X_{c_i} * F_{c_o, c_i}$

**反向传播：**

跨通道求和即前向通道累加的逆运算。对于 $\frac{\partial L}{\partial X}$，来自各输出通道的梯度沿 $C_{out}$ 维度求和（因为每个输入通道 $X_{c_i}$ 参与所有 $C_{out}$ 个输出通道）：

$\boxed{\frac{\partial L}{\partial X_{c_i}} = \sum_{c_o=0}^{C_{out}-1} \operatorname{rot}_{180}(F_{c_o, c_i}) *_{valid} \text{pad}_{K-1}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O_{c_o}}\right)\right)}$

对于 $\frac{\partial L}{\partial F}$，仅第 $c_o$ 个输出通道的梯度通过第 $c_i$ 个输入通道反向传播：

$\boxed{\frac{\partial L}{\partial F_{c_o, c_i}} = X_{c_i} * \text{dil}_S\!\left(\frac{\partial L}{\partial O_{c_o}}\right)}$

---

### 6. 推广六：Batch 维度 $N$

Batch 维度独立于空间和通道维度，按样本累加即可：

$\boxed{\frac{\partial L}{\partial F} = \sum_{n=0}^{N-1} \frac{\partial L}{\partial F}\Big|_{X^{(n)}}}$

$\frac{\partial L}{\partial X}$ 则保持 batch 维度分离，逐样本独立计算。

---

### 总结：最一般形式

| 梯度 | 表达式 |
|------|--------|
| $\frac{\partial L}{\partial F_{c_o, c_i}}$ | $\displaystyle\sum_{n} X_{c_i}^{(n)} * \text{dil}_S\!\left(\frac{\partial L}{\partial O_{c_o}^{(n)}}\right)$ |
| $\frac{\partial L}{\partial X_{c_i}^{(n)}}$ | $\displaystyle\sum_{c_o} \operatorname{rot}_{180}(F_{c_o, c_i}) \; *_{valid} \; \text{pad}_{\hat{K}-1}\!\left(\text{dil}_S\!\left(\frac{\partial L}{\partial O_{c_o}^{(n)}}\right)\right)$ |

其中 $\hat{K} = (K-1)D + 1$（膨胀后有效核尺寸），$\text{crop}$ 操作在 padding 情形下施加。

> **统一原则：** 卷积反向传播始终可归约为步长-1 的标准卷积，仅需对梯度张量施加 $\text{dil}$（应对 stride）与 $\text{pad}$（应对尺寸匹配）两种结构变换。通道维度上做求和聚合，batch 维度上做累加聚合。

## 卷积运算的矩阵表示

### 前向传播

朴素前向传播4重循环显然是不可接受的，我们希望通过将其转变为矩阵乘法来加速：

```
输入 x: (N, C, H, W)
         ↓  im2col (把每个C*HH*WW的三维卷积窗口展开为一列)
x_cols: (C*HH*WW,  N*H_out*W_out)
         ↓  矩阵乘法 w_flat · x_cols + b (这里w_flat是把每个C*HH*WW卷积核展开为一行)
res:     (F,  N*H_out*W_out)
         ↓  reshape + transpose()
输出 out: (N, F, H_out, W_out)

```

对应源码：

```python
def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im.
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # 维度检查
    assert (W + 2 * pad - filter_width) % stride == 0, "width does not work"
    assert (H + 2 * pad - filter_height) % stride == 0, "height does not work"

    # Create output
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    # x_cols.shape:C*HH*WW,N*out_weight*out_width
    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    # w.shape:F,C,HH,WW->F,C*HH*WW b.shape:F,->F,1
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
	# out.shape:F,N*out_weight*out_width->F,out_height,h_width,N->N,F,out_height,out_width
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache
```

### 反向传播

朴素反向传播的效率比前向传播还要糟糕，达到了惊人的7重循环，不过在已经完成前向传播的矩阵乘法运算化的情况下，优化反向传播的思路是容易得到的

```
dout: (N, F, H_out, W_out)
        ↓ transpose + reshape
dout_flat: (F, N*H_out*W_out)
        ↓
   ┌────┼────┐
   ↓    ↓    ↓
  dw   dx   db
(F,C, (N,C, (F,)
HH,WW) H,W)

```

具体代码不再展开，此处只需要注意一个公式：对矩阵乘法 `Y = W @ X`，有 `dW = dY @ X^T`，且`dX=W^T*dY`

### 优化结果

```
Testing conv_forward_fast:
Naive: 8.967643s
Fast: 0.046368s
Speedup: 193.402034x
Difference:  4.926407851494105e-11

Testing conv_backward_fast:
Naive: 37.256277s
Fast: 0.010858s
Speedup: 3431.209675x
dx difference:  1.949764775345631e-11
dw difference:  3.681156828004736e-13
db difference:  3.1393858025571252e-15
```

优化效果达到了惊人的三千多倍

## 思考：引入max pool后梯度检查误差骤增

观察到：之前三层全连接神经网络在梯度检查时的相对误差大致在**e-7**数量级，但是引入最大池化之后梯度误差达到了**e-2**。已知相对误差来自于**不可微点**处数值法和解析法的不同处理策略，而网络中的不平滑因素只有**池化层**和**ReLU**层，而先后两个模型都包括ReLU层，说明最大池化的影响更大。

### ReLU 的梯度不连续性：相对温和

ReLU 的梯度在 `x=0` 处发生**跳跃**，但两个分支（梯度 0 和梯度 1）至少都是**有定义**且**非零的常数**。数值上，当 `h=1e-5` 扰动跨越零点时，数值梯度会变成 `0.5` 左右（因为 `(ReLU(h) - ReLU(-h))/(2h) = h/(2h) = 0.5`），而解析梯度取 0 或 1。最大误差是有限且可控的，且只在少数接近零的神经元上发生。

在纯 ReLU 网络中（比如 `affine - relu - affine - relu - affine`），虽然也有误差，但因为：

- 激活值通常分布在比零远得多的区域，只有极少数神经元恰好被扰动跨越零点
- 误差不会随时间步放大（不会出现反直觉的梯度流向切换）

因此整体相对误差通常仍保持在 `1e-7~1e-9`。

---

### 2. Max Pooling 的梯度不连续性：灾难级

Max pooling 的梯度不仅是不连续的，而且是**高度非局域的**：它只在一个非常狭窄的“决策空间”内是常数，一旦扰动改变了 argmax 的位置，梯度**立即跳到完全不同的位置**，甚至可能从有梯度变成零梯度。

举个例子：一个 `2x2` 窗口内有两个很接近的值 `a` 和 `b`，当前 `a > b`，所以梯度指向 `a` 所在的位置。扰动 `a` 或 `b` 极小的量 `h` 就可能让 `b` 反超 `a`，此时：

- **解析梯度**仍然流向 `a` 的位置（因为我们是基于未扰动的参数计算梯度）
- **数值梯度**却流向 `b` 的位置（因为扰动后 max 变了）

这两个梯度的差异不是一个缩放因子，而是**梯度的空间位置都变了**，误差在 L2 范数下可以非常巨大（甚至超过 100%）。而且 max pooling 在一个批次的每个样本的每个区域都可能会发生这种 argmax 切换，总误差被急剧放大。

## spacial and group normalization

### spacial batch norm

前向传播：之前batch norm 的输入shape都是(N，D)，现在需要处理（N，C，H，W）的输入，只需要通过transpose+reshape方法将X变为($N*H*W$,C)，再调用batch norm即可

```python
# 1. 把 (N, C, H, W) 转置并 reshape 成 (N*H*W, C)
x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
```

反向传播：与前向传播类似，将维度调整至适配batch norm即可

### group norm

group norm概念参考[[Lecture06 CNN Architectures#slide9-14：Normalization]]，更接近layer norm

前向传播：将求和通道（H，W，C/G）通过两重reshape展开，然后调用layer norm

```python
    x_grouped = x.reshape(N, G, C//G, H, W)          # (N, G, C//G, H, W)
    x_flat = x_grouped.reshape(N * G, -1)            # (N*G, C//G * H * W)
```

反向传播同理