## 序列依赖性

之前的讨论只涉及了[[lec01 线性回归#时间步特征]]，但是实际上[[lec01 线性回归#滞后特征]]也是时间序列模型的重要组成部分。

这里需要澄清两个概念：
1. 时间依赖性：事件与时间之间的关系，包括趋势，多项式，季节性，本质上都是与t相关的特征项
2. 序列依赖性：事件与前序事件之间的关系

![[Pasted image 20260518171329.png]]

以该图为例，上下两图的时间依赖性都不强，很难说有周期或者趋势，但是lag 1 滞后图能体现其序列依赖性

## 滞后图和滞后序列

滞后序列就是将创造当前时间的n个步长之前的副本

|          | y   | y_lag_1 | y_lag_2 |
| -------- | --- | ------- | ------- |
| Date  日期 |     |         |         |
| 1954-07  | 5.8 | NaN     | NaN     |
| 1954-08  | 6.0 | 5.8     | NaN     |
| 1954-09  | 6.1 | 6.0     | 5.8     |
| 1954-10  | 5.7 | 6.1     | 6.0     |
| 1954-11  | 5.3 | 5.7     | 6.1     |

滞后图就是把原序列作为x值，滞后序列作为y值绘制的图表

![[Pasted image 20260518172950.png]]

### 自相关性（Autocorrelation，ACF）

序列依赖度的常见度量为自相关性（autocorrelation），自相关性体现了时间序列与其某个滞后序列之间的相关性。

其计算见[AutoCorrelation - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/autocorrelation/)

这里做一个简单的阐述: 
$$ \hat{\rho}_k = \frac{ \frac{1}{n} \sum_{t=k+1}^{n} (X_t - \bar{X})(X_{t-k} - \bar{X}) }{ \frac{1}{n} \sum_{t=1}^{n} (X_t - \bar{X})^2 } = \frac{ \sum_{t=k+1}^{n} (X_t - \bar{X})(X_{t-k} - \bar{X}) }{ \sum_{t=1}^{n} (X_t - \bar{X})^2 }
$$
注意：一段长度为n的时间序列，当滞后步长为k时，最长的滞后序列长度为n-k，所以Σ的下标只能从k+1开始

> 公式中的协方差：[ Statistics By Jim --- Covariance: Formula, Definition & Example - Statistics By Jim](https://statisticsbyjim.com/basics/covariance/)
>![[Pasted image 20260518190322.png]]
很明显，协方差能够标度两个序列的相关方向，但是由于数值不统一，所以无法体现相关强度

### 偏自相关（Partial Autocorrelation，PACF）

当我们计算时间序列和k步滞后序列之间的自相关性时，实际上包含了时间序列、1步时间序列至k-1步滞后序列对相关性的贡献。如果希望单独得到两者之间的相关性，则需要计算PACF。

偏自相关的计算建立在Yule-Walker方程组上：
$$
\begin{bmatrix}
1      & \rho_1  & \rho_2  & \cdots & \rho_{k-1} \\
\rho_1 & 1       & \rho_1  & \cdots & \rho_{k-2} \\
\rho_2 & \rho_1  & 1       & \cdots & \rho_{k-3} \\
\vdots & \vdots  & \vdots  & \ddots & \vdots    \\
\rho_{k-1} & \rho_{k-2} & \rho_{k-3} & \cdots & 1
\end{bmatrix}
\begin{bmatrix}
\phi_{k1} \\ \phi_{k2} \\ \phi_{k3} \\ \vdots \\ \phi_{kk}
\end{bmatrix}
=
\begin{bmatrix}
\rho_1 \\ \rho_2 \\ \rho_3 \\ \vdots \\ \rho_k
\end{bmatrix}
$$
其中$\rho_n$表示自相关系数，$\phi_{kk}$含义是：在已经用了 \($X_{t-1}$, $X_{t-2}$, $\dots$, $X_{t-(k-1)}$\) 这些中间变量的情况下，\($X_{t-k}$\) 对 \($X_t$\) 还有多少独立的、净的线性影响

解得偏自相关之后，我们可以绘制相关图，即滞后k-偏自相关图：

![[Pasted image 20260518192846.png]]

需要注意的是，自相关性和偏自相关性都基于线性假设，在现实中往往很少有线性情况，此时可以通过滞后图判断

