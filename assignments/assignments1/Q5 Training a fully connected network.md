### 模型健全性检查

其实健全性检查最早在[[Q3 two_layer_net]]中就已经出现过了，但是直到本次练习才注意到他的作用，即在实现了模型架构之后，如何通过小规模的数据验证其正确性

健全性检查有两种方法：
1. 检查初始损失及梯度计算是否正确
2. 确保神经网络能在较小数据集中达到过拟合

#### 1. 检查初始损失及梯度计算是否正确

需要在reg=0的条件下判断初始损失是否正确，此时没有正则化损失影响，初始损失完全由data loss决定，所以对于cifar-10这样的分类数为10的数据集而言，初始损失应该为-log(0.1)≈2.3

梯度计算的验证：以初始损失作为输入，通过解析法计算梯度，将其与模型计算所得梯度进行对比，相对误差应该小于1e-6

#### 2.确保神经网络能在较小数据集中达到过拟合

其实方法一已经完成了网络工作流中的前向传播和反向传播的正确性了，但是没有验证其在多轮训练下的参数调整能力，其实就是验证权重矩阵的优化是否正确，即`model.params['W1'] -= lr * grads['W1']`

所以方法二就是要在较小的数据集中达到过拟合，具体表现就是训练正确率达到100%，且loss持续下降

### 两种momentum递归表示

根据[[lecture04 backprop#动量]]中的公式推导实现之后，next_w和velocity error竟然达到了惊人的1，导致本人百思不得其解，遂比较两种递归表达式

经过证明，当v0=0时，两种表达式的$W_n$相同，即两种递归表示等价
![[conv等价性推导.jpg]]

但是当v0不为0时，产生了极大偏差，且随递归层数加深而放大，遂对v0!=0的情况进行推导：
$$w_n^{\text{Conv1}} 
= w_0 + \sum_{k=1}^n \beta^k v_0 
- \eta \sum_{k=1}^n \sum_{i=0}^{k-1} \beta^{k-1-i} dw_i$$$$w_n^{\text{Conv2}} 
= w_0 - \eta \sum_{k=1}^n \beta^k v_0 
- \eta \sum_{k=1}^n \sum_{i=0}^{k-1} \beta^{k-1-i} dw_i$$
容易发现两式当且仅当v0=0时相等，不为0时，${Δ =(1+\eta) \sum_{k=1}^n \beta^k v_0}$，误差自然爆炸

### 其他模型参数调整方法

以下方法都没有被实验使用，只作为补充资料提及：
[[Nestrov momentum]]:动量优化版
[学习率退火](https://cs231n.github.io/neural-networks-3/#anneal)：在epoch推进过程中降低学习率，从而在接近损失梯度谷底的时候减少调整步长
[二阶方法](https://cs231n.github.io/neural-networks-3/#second)：如果能够计算出Hessian矩阵(二阶偏导数矩阵)，将其逆矩阵左乘梯度矩阵，就能更好的调整梯度，使其在平缓的曲率方向更激进，在陡峭的方向更保守；但是由于直接计算开销过大，只能通过L-BFGS等方式模拟hassian矩阵
[adagrad](https://cs231n.github.io/neural-networks-3/#ada):第一个被介绍的自动调节学习率的算法，通过cache逐参数计算历史梯度平方和，将其开根后作为分母来调整dw矩阵，实际上就是逐参数调整学习率，使其在陡峭的方向上步长更小

```
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

但是Adagrad有一个缺陷，由于其对历史梯度直接取平方和，使其不能够反映某曲率方向上的最近情况，而且由于持续增加，很容易导致学习率停滞，所以我们引入RMSprop方法

RMSprop方法很简单，就是在Adagrad方法上，对$dw^2$做一个加权平均
```
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Adam算法只是引入动量思想，在RMSprop的基础上，对dx做加权平均;并引入迭代修正因子t，弥补迭代次数较小时m和v过小的问题
```
# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x += - learning_rate * mt / (np.sqrt(vt) + eps)
```







