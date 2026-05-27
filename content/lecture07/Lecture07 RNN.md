---
title: "Lecture 07: RNN"
publish: true
---

## 序列建模 (Sequence Modeling)

视频重点介绍了序列建模的四种主要模式，突破了固定尺寸输入输出的限制：

- **一对多 (One-to-Many)**：如图像描述（输入一幅图，输出一段变长的文字描述）。
- **多对一 (Many-to-One)**：如视频分类（输入多帧视频，输出一个类别标签）。
- **多对多 (Many-to-Many)**：包括输入输出长度不相等的情况，以及如视频逐帧分类等输入输出长度相等的情况。RNN structure
---
## RNN 的前向传播

- **核心机制**：RNN 拥有 **隐藏状态 (Hidden State)**，在处理序列时，它会根据当前输入和前一时刻的隐藏状态不断更新。
- **数学表达**：视频给出了 RNN 的递推公式，并解释了隐藏状态如何通过学习到的权重矩阵进行变换。
- **手算示例**：[[RNN concrete example]]：讲师展示了一个手动构建 RNN 的“玩具例子”，用于检测二进制序列中重复出现的数字，帮助学生理解权重矩阵的实际运作方式

<div style="text-align: center;">
    <img src="Pasted image 20260526000923.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 1：Many-to-Many 架构</div>
</div>

>图片展现的是many-to-many的RNN架构，为了使得每一个x产生一个对应的y输出，在单个RNN模块中，要通过上一步的隐藏状态和输入的x值，得到一个新的隐藏状态，即$h_t = f_W(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$；通过这个隐藏状态，计算得到输出层，即$y_t = f_Y(W_{yh}h_t + b_y)$

---
## RNN 的反向传播

### 随时间反向传播 (BPTT)


<div style="text-align: center;">
    <img src="Pasted image 20260526001409.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 2：many-to-many 架构下 BPTT 对 loss 的贡献</div>
</div>

BPTT的梯度计算非常直接：$$\delta_t = \frac{\partial L_t}{\partial h_t} + \left( \frac{\partial h_{t+1}}{\partial h_t} \right)^\top \delta_{t+1},
\quad t = k-1, \dots, 1$$
这里的k是当前时间步到最后一个时间步的距离，也就是说所有后续时间步都对当前时间步有贡献
### 截断反向传播 (Truncated BPTT)

但是当序列相当长的时候，由于BPTT的梯度计算依赖于每个y值（激活值），导致对GPU显存提出了很高的要求，所以提出了截断反向传播

<div style="text-align: center;">
    <img src="Pasted image 20260526010014.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 3：Truncated BPTT 示意图，注意梯度在块与块之间截断</div>
</div>

截断反向传播将序列分为n个块，对于每个块的最后一个时间步，其梯度为：$$ \delta_k = \frac{\partial L_k}{\partial h_k} $$
在同一个块内(设其大小为k），其前面的时间步的梯度计算公式为：

$$\delta_t = \frac{\partial L_t}{\partial h_t} + \left( \frac{\partial h_{t+1}}{\partial h_t} \right)^\top \delta_{t+1},\quad t = k-1, \dots, 1$$

> 这里体现出与BPTT的区别：BPTT中某一个时间步的梯度会传递到前序所有时间步，而截断BPTT只影响于所在块内前序时间步。也就是说，梯度不会跨块传递。
> 可以看出截断BPTT基于一个假设：间隔较远的时间步对梯度的贡献可以忽略不计。

对每个窗口内的梯度值进行累加：$$\frac{\partial L_{\text{block}}}{\partial W_{hh}} = \sum_{t=1}^{k} 
\left( \frac{\partial h_t}{\partial W_{hh}} \right)^\top \delta_t$$
> 这里有一个观察：在 many-to-one 的情境下，截断BPTT不适用，因为只有最后一个块有梯度，其他块的梯度全都消失。


### 梯度计算困境

<div style="text-align: center;">
    <img src="Pasted image 20260526212945.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 4：反向传播梯度流</div>
</div>

注意到反向传播中，对权重矩阵求偏导，且时间步跨度T较大时，会出现极大的连乘项：$$\frac{\partial L_T}{\partial W} = \frac{\partial L_T}{\partial h_T} \left( \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}} \right) \frac{\partial h_1}{\partial W}$$
当考虑激活函数tanh时，连乘项总是小于1，会导致梯度消失，这也是主要的问题

即使不考虑激活函数tanh，此时连乘项可以展开为更简单的形式，可以发现梯度传递高度依赖于$W_{hh}$的取值，很容易产生梯度爆炸或者消失：$$\frac{\partial L_T}{\partial W} = \frac{\partial L_T}{\partial h_T} W_{hh}^{T-1} \frac{\partial h_1}{\partial W}$$
### LSTM(long short term memory)

由于RNN出现的梯度消失/爆炸问题，出现了他的复杂变体LSTM，旨在解决梯度传递问题

<div style="text-align: center;">
    <img src="Pasted image 20260526215334.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 X：LSTM 示意图</div>
</div>
这里课程并没有详细展开LSTM，但是可以参考[ Understanding LSTM Networks -- colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)，Colah 的博客中详细解释了LSTM中各个门的机制

> 事实上，下面展示的 RNN 成果大多数都是由 LSTM 及其变体创造的

---
## RNN 成果

### 字符级RNN

<div style="text-align: center;">
    <img src="Pasted image 20260526134713.png" width="600" />
    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 5：字符级RNN示例</div>
</div>

字符级RNN是一个很标准的 Many-to-many RNN 模型，可以通过一个语料库来训练说话风格

python实现：[Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy](https://gist.github.com/karpathy/d4dee566867f8291f086)

#### Embedding layer

注意到这里相比之前的 RNN 架构多了一层 Embedding Layer ，它的作用是把离散、稀疏的独热编码向量转换成稠密、连续空间向量。如果没有Embedding layer，任何两个字符之间的几何距离和夹角都相同，也就是字符之间没有任何关联，然而在实际文本中，比如常见的 the，t-h-e之间应该有所关联，所以他们之间的几何距离应该相近。

经过Embedding layer之后，字符向量被映射到连续的、有语义的坐标系中，得到一个稠密的向量，再输入到 Hidden layer 中作为输入。

Embedding layer 在逻辑上就是一个矩阵E，与独热编码相乘得到稠密向量，但是把这个矩阵与隐藏层的权重矩阵 W 分割开来，有以下两点考虑：

1. 计算成本：对于一个高维、稀疏的独热编码而言，矩阵乘法太浪费了，所以实际实现中采用的是查表方式，直接从输入字符映射到对应向量
2. 逻辑解耦：矩阵E考虑的是字符之间的通用关系，而矩阵W考虑的是特定文本中的时序关系，所以经过大量文本训练的矩阵E可以迁移学习到其他 RNN 模型之中

### 其他 RNN 示例

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260526210318.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 6：Image caption : CNN提取特征后输入RNN</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260526210853.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 7：Visual Question Answering</div>
    </div>
</div>


<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
    <!-- 第一张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260526211154.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 8：VLN（视觉语言路线规划）</div>
    </div>
    <!-- 第二张图 -->
    <div style="text-align: center;">
        <img src="Pasted image 20260526212012.png" width="400" />
        <div style="font-size: 0.85em; color: #888; margin-top: 5px;">图 9：Multilayer RNN</div>
    </div>
</div>

---
## RNN 总结

RNN的优势：

1. 能处理任意长度的序列
2. 对当前时间步的运算理论上可以用到很多步之前的信息
3. 模型规模不会随着输入规模增加而增加
4. 对每个时间步的运算相同，带来较好的对称性

RNN的缺陷：

1. 训练阶段计算较慢，且传统BPTT需要存储激活值，对显存要求较高
2. 梯度消失或爆炸

---
### 补充资料

1. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
2. [deeplearningbook.org/contents/rnn.html](https://www.deeplearningbook.org/contents/rnn.html)
3. [NLP From Scratch: Classifying Names with a Character-Level RNN — PyTorch Tutorials 2.12.0+cu130 documentation](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
4. [CS231n Deep Learning for Computer Vision](https://cs231n.github.io/rnn/)