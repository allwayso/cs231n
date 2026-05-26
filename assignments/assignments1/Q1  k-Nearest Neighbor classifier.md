## 1.环境初始化

这里遇到python版本不兼容问题。

![[环境不兼容.png]]

简而言之，就是远古版本（大概为python3）中使用imp来执行文件导入，但是从python3.12开始，imp库由于过于老旧而被删除；而这里的`%load_ext autoreload`中的autoreload方法中包含了语句`--> 121 from imp import reload`,而python3.12环境中找不到标准库imp，导致爆出ModuleNotFoundError。

而解决方法也很神奇，由于更改python环境可能导致内核连接错误，所以这里用现代库模拟了imp的功能，并将其命名为imp，这样调用autoreload的时候，实际上就是在调用现代的sys.modules 逻辑

```python
import importlib, sys
from types import ModuleType

# 创建一个假的 imp 模块并注入到 sys.modules
fake_imp = ModuleType("imp")
fake_imp.reload = importlib.reload
sys.modules["imp"] = fake_imp
```


## 2.数据提取和训练

### list语法

```python
num_training = 5000
mask = list(range(num_training))//mask是一个长为5000的列表
X_train = X_train[mask]
y_train = y_train[mask]
```

list更接近cpp的vector而不是array，动态扩容，其余操作参考[[list的基本语法]]

## 3. 性能优化

其实三种算法的数学机制是一致的，都为以下矩阵运算：$$\text{Dists}^{(2)} = \underbrace{(\mathbf{x}_{norm} \cdot \mathbf{1}_N^T)}_{\text{项 A}} + \underbrace{(\mathbf{1}_M \cdot \mathbf{y}_{norm}^T)}_{\text{项 B}} - \underbrace{2XY^T}_{\text{项 C}}$$
这里还可以进行更严谨的展开：$$\mathbf{y_norm}^T = [ \|y_1\|_2^2, \|y_2\|_2^2, \dots, \|y_N\|_2^2 ]$$
但是他们的算法实现却大大不同

### 1. Two Loops: 零优化（最底线）

```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
    for j in range(num_train):
        diff = X[i, :].astype('float') - self.X_train[j, :].astype('float')
        square_sum=np.sum(np.square(diff))
        dists[i,j]=np.sqrt(square_sum)
return dists        

```
相当于每个$dists[i][j]$都是通过两个行向量相减运算求得的

在两个循环中，计算机每次只看**两个点**。

- **实现逻辑：** `dists[i, j] = sqrt(sum((X[i] - Y[j])**2))`
    
- **数学拆解：** 它实际上在内层循环里重复计算了每一项。
    
    - $\|x_i\|^2$ 被计算了 $N$ 次（针对每个训练点都算一遍）。
        
    - $\|y_j\|^2$ 被计算了 $M$ 次（针对每个测试点都算一遍）。
        
- **性能瓶颈：** 所有的工作都由 Python 解释器一行行解释，且没有任何数据复用。

### 2. One Loop: 局部向量化（广播优化的底线）

```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
for i in range(num_test):
	diff=self.X_train-X[i]
    dists[i, :] = np.sqrt(np.sum(np.square(diff), axis=1))
```

实际上对于每个测试点而言，他对每个训练点的操作是一致的，也就是说每一行$self.X_train[j]$都要减去$X[i]$，所以可以将该行放大使其维度与X一致，一次性进行运算，利用[广播机制broadcasting](https://cs231n.github.io/python-numpy-tutorial/#broadcasting)可以实现

当你减去一个循环，保留 `for i in range(num_test)` 时，你是在对**每一个测试点**做优化。

- **实现逻辑：** `diff = self.X_train - X[i]`
    
- **数学拆解：**
	$$\text{dists}[i, :] = \sqrt{\underbrace{\|x_i\|^2}_{\text{标量}} + \underbrace{\mathbf{y}_{norms}}_{\text{向量 (N,)}} - \underbrace{2(x_i \cdot Y^T)}_{\text{向量 (N,)}}}$$
    
- **优化点：** 虽然代码里没写公式展开，但 NumPy 的 `self.X_train - X[i]` 在底层通过**广播**（Broadcasting）一次性处理了 $X[i]$ 与整个训练集的减法。
    
- **为什么快：** 它把 $N$ 个训练点的像素减法交给了 C 语言内核并行处理。
    
- **性能底线：** 它依然有 $M$ 次 Python 循环开销，且如你之前发现的，如果 $N$ 很大，频繁申请 `diff` 这个大矩阵的内存反而会拖慢速度。

### 3. No Loops: 全局算子化

```python
def compute_distances_no_loops(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        test_sum=np.sum(X**2,axis=1,keepdims=True)
        train_sum=np.sum(self.X_train**2,axis=1)
        dot_product=np.dot(X,self.X_train.T)
        dists=np.sqrt(test_sum+train_sum-2*dot_product)
        return dists
```

这是完全抛弃“点”的概念，直接在“空间（矩阵）”层面操作。

- **实现逻辑：** 就是你写的那个三项合一公式。
    
- **数学拆解：** $\text{Dists}^{(2)} = A + B - C$
    
    - **项 A (test_sum):** 算出所有测试点的能量，只算 **1 次**。
        
    - **项 B (train_sum):** 算出所有训练点的能量，只算 **1 次**。
        
    - **项 C (dot_product):** 利用矩阵乘法一次性算出所有两两配对的交互项。
        
- **优化点：**
    
    1. **数据零冗余：** 每一项模长平方都只算了一次，然后通过广播“刷”满整个矩阵。
        
    2. **硬件利用率：** 矩阵乘法（GEMM）能触发 CPU 的所有核心和最宽的指令集（AVX/SIMD）。

### 三个算法的性能优化对比表

|**维度**|**Two Loops**|**One Loop**|**No Loops**|
|---|---|---|---|
|**数学形态**|标量运算 ($d_{ij}$)|向量运算 ($d_{i, \cdot}$)|矩阵运算 ($D$)|
|**Python 开销**|$M \times N$ 次循环调用|$M$ 次循环调用|**1 次** 函数调用|
|**内存申请**|极小，频繁申请|**巨大**（中间矩阵 `diff`）|**适中**（仅结果矩阵）|
|**核心优化手段**|无|广播 (Broadcasting)|**矩阵乘法 (BLAS)**|
## 交叉验证

交叉验证的基本思路就是，将训练数据分为若干个folds，每次选取一个作为测试集，其他作为训练集，观察正确率

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

# 神秘array_split函数，将数组分成若干个等大的folds，与 np.split 的区别就是能接受除不尽，比切片更优雅高效
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

for k in k_choices:
    # 1. 为当前的 k 初始化一个存放准确率的列表
    k_to_accuracies[k] = []

    for i in range(num_folds):
        # 2. 准备当前的训练集和验证集
        # 神秘concatenate函数，将不同列表进行组装，当axis=0时按行拼接，axis=1时按列拼接；
        # 如果使用+则将变为矩阵加法，所以本质不同
        X_train_cv = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        y_train_cv = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

        X_val_cv = X_train_folds[i]
        y_val_cv = y_train_folds[i]

        # 3. 实例化并训练模型（
        classifier.train(X_train_cv, y_train_cv)

        # 4. 预测
        dists = classifier.compute_distances_no_loops(X_val_cv)
        y_val_pred = classifier.predict_labels(dists, k=k)

        # 5. 计算准确率
        num_correct = np.sum(y_val_pred == y_val_cv)
        accuracy = float(num_correct) / len(y_val_cv)

        # 6. 保存结果
        k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
```

![[不同k值的交叉验证.png]]

## inline questions

**Inline Question 1**

Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

$\color{blue}{\textit Your Answer:}$ 
1. 这个测试点离所有的训练点都较远
2. 这个训练样本比较罕见，距离所有的测试点都较远，对预测基本没有贡献

**Inline Question 2**

We can also use other distance metrics such as L1 distance.
For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$,

the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
And the pixel-wise mean $\mu_{ij}$ across all images is
$$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.

Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply. To clarify, both training and test examples are preprocessed in the same way.

1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
5. Rotating the coordinate axes of the data, which means rotating all the images by the same angle. Empty regions in the image caused by rotation are padded with a same pixel value and no interpolation is performed.

$\color{blue}{\textit Your Answer:}$
1 2 3

$\color{blue}{\textit Your Explanation:}$
1.相当于Σ中的每一项同减μ，相互约去
2.正好约掉
3.不影响，相当于case1的距离值除以μ
4.会影响，这样使得图像中的每个像素权重统一了
5.L2具有旋转不变性，而L1不具备

**Inline Question 3**

Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than or equal to that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

$\color{blue}{\textit Your Answer:}$ 2 4


$\color{blue}{\textit Your Explanation:}$

1. 决策边界可以认为是分段线性的，但是总体上来说不是线性的，而且k越大，决策边界越零碎
2. 这是显然正确的，k=1时，因为测试点离他最近的一个点就是他本身，所以训练误差永远为0；而k=5时，如果某个测试点旁边有较多其他测试点，则会出现错误
3. 不一定，一般而言k越小，决策边界越平滑，越倾向于欠拟合；k越大，决策边界越零碎，越倾向于过拟合。但是测试误差与训练集和测试集都有关，在实验之前不能确定最佳k值
4. 正确的，k最邻近法采取懒训练策略，在训练过程中只做数据存储，不做预处理，导致预测时间随训练规模增长而增长

