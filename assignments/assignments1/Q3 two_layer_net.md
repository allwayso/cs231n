## db计算

先前的[[lecture04 backprop#反向传播的矩阵运算推导]]都是把偏置向量b作为W的一个维度计算，即认为$out=x*W$,所以并没有推导过db的计算方法

需要注意的是，虽然python代码可以写作`out=x.dot(W)+b`，但是矩阵运算中并不是加上了一个b向量，这里的广播机制使其等效于`out=x.dot(W)+b.dot(np.ones((N, 1)))`，这样的话矩阵运算推导就类似前半部分$x*W$了。利用维度对齐思想，把$row=1_N$和dout对齐为b的形状，可以推导出$db=dout.dot(row.T)$，相当于把dout按列求和，所以更简单的写法为`db=np.sum(dout,axis=0)`

```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_row=x.reshape(x.shape[0],-1)
    dw=x_row.T.dot(dout)
    dx_row=dout.dot(w.T)
    dx=dx_row.reshape(x.shape)
    db=np.sum(dout,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
```

## copy()

实现函数的时候一定要记住不要修改输入的参数，应该通过copy函数获取参数的副本再进行操作，以免后续检验或者二次使用时出现难以理解的误差

```python
N=x.shape[0]
shift_x=x-np.max(x,axis=1,keepdims=True)
exp_scores=np.exp(shift_x)
prob=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
loss=-np.sum(np.log(prob[np.arange(N),y]))/N
```

惨痛教训之直接使用`x-=np.max(x,axis=1,keepdims=True)`，而数值法求梯度中取一个极小步长h，计算前后loss差值来计算梯度，这个过程中调用了两次`softmax_loss(x,y)`，导致x早已不是初始值，引入爆炸误差

## softmax loss对x偏导

#### 1. 准备工作：

Softmax 函数：$p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}$

Loss 函数：$L_i = -\ln(p_y)$ 
 

#### 2. 链式法则

由链式法则：$$\frac{\partial L_i}{\partial s_k} = \frac{\partial L_i}{\partial p_y} \cdot \frac{\partial p_y}{\partial s_k}$$
前一项可以简单展开：$$\frac{\partial (-\ln p_y)}{\partial p_y} = -\frac{1}{p_y}$$

#### 3.分类讨论

情况 A：$k = y$（对正确类别得分求导）：$$\frac{\partial p_y}{\partial s_y} = \frac{e^{s_y} \cdot \sum e^{s_j} - e^{s_y} \cdot e^{s_y}}{(\sum e^{s_j})^2} = \frac{e^{s_y}}{\sum e^{s_j}} \cdot \frac{\sum e^{s_j} - e^{s_y}}{\sum e^{s_j}} = p_y(1 - p_y)$$
情况 B：$k \neq y$（对错误类别得分求导）：$$\frac{\partial p_y}{\partial s_k} = \frac{0 \cdot \sum e^{s_j} - e^{s_y} \cdot e^{s_k}}{(\sum e^{s_j})^2} = -\frac{e^{s_y}}{\sum e^{s_j}} \cdot \frac{e^{s_k}}{\sum e^{s_j}} = -p_y p_k$$

#### 4. 带入公式

- **如果 $k = y$（正确类位置）：**$$\frac{\partial L_i}{\partial s_y} = \left( -\frac{1}{p_y} \right) \cdot p_y(1 - p_y) = -(1 - p_y) = \mathbf{p_y - 1}$$
    
- **如果 $k \neq y$（错误类位置）：**$$\frac{\partial L_i}{\partial s_k} = \left( -\frac{1}{p_y} \right) \cdot (-p_y p_k) = \mathbf{p_k}$$

## rand or randn ？

- **`np.random.rand`**：生成的是 **[0, 1) 之间的均匀分布**  (Uniform distribution)。
    
- **`np.random.randn`**：生成的才是 **均值为 0，标准差为 1 的高斯分布** (Normal/Gaussian distribution)。

所以在生成初始权重矩阵时应该调用randn函数

### 