>声明：本章是因为没有成体系的资料支撑，只能依赖于slide进行整理，属于笔记整理的特殊情况

# part1：HOW to build CNNs？
## slide1-8：对前面卷积层、池化层和全连接层的recap

## slide9-14：Normalization

![[normalization.png]]

- **$N$ (Batch)**：一批次里有多少张图片。
    
- **$C$ (Channel)**：图片的通道数（如 RGB 3通道，或卷积层的特征图通道）。
    
- **$H, W$ (Height, Width)**：图片的长和宽，这里看作一个维度，做了展开。

> 蓝色平面：计算面，以batch norm为例，每个shape为(N,H,W)的立方体计算得到一组均值和方差

|归一化方法|统计量计算范围|统计量形状|一句话描述|
|---|---|---|---|
|**Batch Norm**|每个通道，跨所有样本和空间位置|(C,)|以通道为切面，横切所有样本|
|**Layer Norm**|每个样本，跨所有通道和空间位置|(N,)|以样本为切面，纵切所有通道|
|**Instance Norm**|每个样本的每个通道，跨空间位置|(N, C) 或 (C, N)|每个样本的每个通道独立归一化|
|**Group Norm**|每个样本的每组通道，跨组内通道和空间位置|(N, G)，其中 G = C / group_size|介于 Layer Norm 和 Instance Norm 之间，通道分组归一化|

## Slides15-20 dropout

### why dropout

1. 防止共适应(co-adaption):神经元之间可能产生某种微弱的依赖关系，即一个神经元依赖另一个神经元的输出。引入dropout之后，由于依赖随时可能被关闭，所以每个神经元被强迫独立学习
2. 大量子模型平均协商：引入dropout后，每个参数在每轮训练中都被不同的上下文(神经元开关组合)更新，所以最终的训练结果可以看作所有子模型(共享一套参数的不同神经元开关组合)协商的结果。把所有神经元的开关看作一个二元掩码，那么拥有n个神经元的模型可以看作对$2^n$种子模型的平均大模型

### 训练与测试的期望一致性

训练期间，每个神经元以概率 p 保留，以概率 1−p 被置为零。所以训练时，如果一个神经元的激活值是 a，那么它的**期望输出**是：
$$E[output]=p⋅a+(1−p)⋅0=p⋅a$$测试时，如果你把 Dropout 关掉（所有神经元都激活），神经元的输出就是 a，而不是 p⋅a。

为了保证一致性，有两种方式：
1. 标准dropout:测试时对测试结果\*p
2. Inverted Dropout训练时，每次dropout，对剩余的神经元激活值除以p

## Slides21-31 Activation function

### why activation

如果没有激活函数，且归一化层为predict模式（固定参数）的情况下，此时整个网络的卷积层/全连接层/归一化层都是线性的，也就是他只是线性变换的组合，仍然可以用一个线性运算表示，即有效深度只有1，层数堆叠没有意义。

从另一个视角来说，深度学习网络应该能够学习从低级边缘到高级语义的抽象，而这依赖于非线性因素的参与。如果没有非线性，再深的模型实质上也只是在学习低级的线性语义。

拓展：万能逼近定理[[cybenkoApproximationSuperpositionsSigmoidal1989]],简而言之，通过线性层+S形激活函数，可以无限逼近任何连续函数

### 激活函数对比

| 名称           | 函数方程                                                                                                  | 导数                                                                                            | 优点                                                   | 缺点                                      |
| ------------ | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------- |
| Sigmoid      | $\sigma(x) = \frac{1}{1 + e^{-x}}$                                                                    | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$                                                       | 输出在 $(0,1)$ 之间，适合做概率输出；平滑可微                          | 梯度饱和（梯度消失）；输出非零中心；计算含指数项开销大             |
| ReLU         | $\text{ReLU}(x) = \max(0, x)$                                                                         | $\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}$                     | 正区间梯度恒为 1，缓解梯度消失；计算极快；产生稀疏激活                         | 负区间梯度恒为 0，神经元一旦"死掉"不再更新（Dead ReLU）；非零中心 |
| Leaky ReLU   | $\text{LReLU}(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}$ 通常 $\alpha = 0.01$   | $\text{LReLU}'(x) = \begin{cases} 1, & x > 0 \\ \alpha, & x \leq 0 \end{cases}$               | 负区间有梯度，缓解 Dead ReLU 问题；计算同样高效                        | $\alpha$ 是超参数需要人工设定；负区间梯度线性，优势并不总是明显    |
| ELU          | $\text{ELU}(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \leq 0 \end{cases}$ 通常 $\alpha = 1$ | $\text{ELU}'(x) = \begin{cases} 1, & x > 0 \\ \text{ELU}(x) + \alpha, & x \leq 0 \end{cases}$ | 负区间饱和趋于 $-\alpha$，使均值接近零，加速收敛；对噪声更鲁棒                 | 计算含指数项，比 ReLU 慢；负区间饱和同样可能造成梯度小          |
| GELU         | $\text{GELU}(x) = x \cdot \Phi(x)$，$\Phi(x)$ 为标准正态 CDF；常用近似 $x \cdot \sigma(1.702x)$                  | $\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$，$\phi(x)$ 为标准正态 PDF                             | 在 NLP（BERT, GPT）中表现优异；结合 ReLU 和 Dropout 的思想；平滑且非单调   | 计算成本高于 ReLU；精确形式涉及误差函数，近似形式精度略损         |
| SiLU (Swish) | $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$                                           | $\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))$                       | 无上界有下界；非单调（在 $x \approx -1.28$ 处有轻微负值）；深层网络中常优于 ReLU | 计算含 Sigmoid，速度慢于 ReLU；梯度形式较复杂           |

[[激活函数的演进]]：可以从梯度消失、计算成本、平滑性、鲁棒性、均值是否产生偏移、稀疏性等角度思考，可以看看链接中的综述


## slides31-59 CNN architecture

### Case study：VGGNet

![[Pasted image 20260503210029.png|400]]

**（相比AlexNet）更深的网络，更小的卷积核**：VGGNet中连续三层3\*3网络,其感受野相当于一层7\*7网络。其优势有两点：1.**参数量更小**——假设输入输出维度深度都为C，则前者为$3^3*C^2$，后者为$7^2*C^2$，参数量减少了将近一半；2.**更多非线性**——每次卷积都附带一次激活函数，相当于进行了一次抽象，所以前者的抽象程度更高

### Case study：ResNet

神经网络在变深的时候训练误差和测试误差都升高，但这并不是过拟合导致的（废话，过拟合的话就不会训练误差升高了），为了解决这个问题，Kaiming He等人提出了残差块方法，参见[[heDeepResidualLearning2015]]

简而言之，对于ResNet而言，最小的单位并不是层数，而是Residential Block，每个残差块是两个带ReLU的卷积层的组合。通过残差块的设计，深层神经网络的梯度能够传到表层，并且损失景观更加平滑、非凸性更弱，使得更深层的神经网络成为可能。顺带一提，VGGNet后期研究已经发现全连接层没有必要，ResNet中不包含全连接层，这也使其虽然层数远深于VGGNet，但是参数量反而更少。

## slides60-66 Weight Initialization

### The target

权重矩阵初始化的目标是，在前向传播和反向传播的过程中，数值/梯度不应消失，也不应该爆炸，应该保持在一个合适的范围内。

而我们在初始化的过程中，可以看作调整两个变量——分布种类和方差项，因为均值是不言自明的，必然是0。高斯分布和正态分布其实没有太大影响，真正影响模型性能的是方差项，也就是weight scale参数。
### Xavier initialization

对于一个两层神经网络，前一层的输出为x，输出深度为$D_{in}$，均值为$E(x)$，方差为$Var(x)$。设第二层的输出为y，权重矩阵为w，忽略偏置项，则$$y = \sum_{i=1}^{D_{\text{in}}} w_i x_i$$
假设：
1. w 和 x 互相独立
2. 所有 $w_i$ 同分布、所有 $x_i$ 同分布
3. 均值都是 0

那么此时方差可加，即$$\mathrm{Var}(y)
= \mathrm{Var}\left(\sum_{i=1}^{n_{\text{in}}} w_i x_i\right)
= \sum_{i=1}^{n_{\text{in}}} \mathrm{Var}(w_i x_i)$$
由于w和x的均值为0，且相互独立，那么$E(wx)=E(w)*E(x)=0$，所以$Var(y)=E(y^2)-E(y)^2=E[(wx)^2]-0=E(w^2x^2)$

又可知$w^2$和$x^2$也相互独立，所以$E(w^2x^2)=E(w^2)*E(x^2)=Var(w)*Var(x)$

又由于在$n_in$个输入维度上方差可加，所以$Var(y)=n_in*Var(w)*Var(x)$

为了保持神经网络的稳定性，**我们希望每层输出方差不变**，所以对我希望$Var(w)=1/n_in$，所以$w \sim \mathcal{N}(0, 1/n_in)$，那么容易推导出`W=np.random.randn(Din,Dout)*np.sqrt(1/Nin)`

### Kaiming Initialization

对于包含ReLU激活函数的多层神经网络而言，由于ReLU会将负数输入截断，导致每层方差缩水一半，为了补偿方差损失，`W=np.random.randn(Din,Dout)*np.sqrt(2/Nin)`

这里是一个工程上的近似，经过ReLU截断之后，二阶矩变为一半，但是下一层的输出wx虽然均值依然为0，但是实际上不再满足高斯分布。虽然Xavier中的假设被打破，不过根据中心极限定理，当卷积核足够多，即$n_in$足够大时，下一层的输出wx仍然接近于高斯分布，所以实际上偏差并不大。

# part2：How to train CNNs

## Data Processing

逐通道0均值化+归一化：`norm_pixel[i,j,c]=(pixel[i,j,c]-np.mean(pixel[:,:,c]))/np.std(pixel[:,:,c])`

## Data Argumentation

### Why argumentation

数据增广有两个目的：
1. 减少过拟合：当数量量不够的时候，通过数据增强可以在不改变训练集的情况下增加数据量，减少过拟合
2. 增强覆盖能力：通过数据增广可以加强训练数据对真实分布的覆盖能力，为更强的泛化能力提供必要条件

### How to Argumentation

数据增广是指将原本的数据做几何变换，亮度变化或区域置零、交叉图像等方法，扩充数据量

## Transfer Learning

### Why transfer learning

我们希望在一个训练集上已经训练完的模型能够被利用在新模型的学习中。这个思路能成立，是因为卷积层学习到的特征具有一定的通用性，所以模型参数可以迁移。

### How to transfer

根据数据集的大小和相似度，采取不同的策略：

| 目标数据集大小 | 与源数据集（如 ImageNet）的相似度 | 推荐做法                           |
| ------- | --------------------- | ------------------------------ |
| 小       | 高                     | 只训练最后的线性分类器                    |
| 小       | 低                     | 很难搞，要么换别的预训练模型，要么收集更多data      |
| 大       | 高                     | 对整个网络进行微调（fine-tune），但使用较小的学习率 |
| 大       | 低                     | 对整个网络进行微调，甚至可以重新训练更多层          |

## Hyperparameter Selection

| 步骤 | 含义 |
|:---|:---|
| **Step 1**: Check initial loss | 检查初始化的损失值是否合理（例如，对于 softmax 分类器，使用 weight decay 时初始 loss 应为 $-\ln(1/\text{num\_classes})$ 附近） |
| **Step 2**: Overfit a small sample | 先用少量样本（如 5-10 个 minibatch）过拟合到 100% 训练精度，验证模型架构和梯度传播是否正确 |
| **Step 3**: Find LR that makes loss go down | 找到一个能让 loss 下降的学习率（learning rate），用全部数据训练一小段时间观察 |
| **Step 4**: Coarse **grid**, train for ~1-5 epochs | **粗略的网格搜索**：在超参数空间中选取几个稀疏的候选值（如学习率 $[1e{-3}, 1e{-4}, 1e{-5}]$ 与 weight decay $[1e{-4}, 1e{-5}]$ 的组合），训练几个 epoch 快速筛选 |
| **Step 5**: Refine **grid**, train longer | **精细化网格搜索**：根据粗网格的结果，在表现好的区域缩小范围，加密采样，训练更长时间 |
| **Step 6**: Look at loss and accuracy curves | 观察训练/验证的 loss 曲线和准确率曲线，判断是否过拟合、欠拟合等 |
| **Step 7**: GOTO step 5 | 反复迭代，直到找到满意的超参数 |




