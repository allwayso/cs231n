## 趋势

趋势是变化中较为持久而稳定的部分，是对变化规律的长期观察

## 移动平均图（moving average plots）

为了过滤掉短期的波动现象，我们创建一个滑动窗口，在窗口内部取平均值，这样就能过滤出相对稳定的变化

![[EZOXiPs.gif]]

利用移动平均图，我们可以判断出大致的趋势形状，从而给建模提供参考

```python
trend = food_sales.rolling(
    window=12,       # 12 month window
    center=True,      # puts the average at the center of the window
    min_periods=6,  # choose about half the window size
).mean()
```
## 工程趋势

```python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()
```

DeterministicProcess是一个建模函数，规定了dataframe的格式。其中constant为常数项，order=1为一次项，对于n次多项式只需要调节order=n即可

次数越高，在训练集上的拟合效果越精细，以order=3为例：

![[Pasted image 20260518163943.png]]


但是对于预测能力而言，并不是次数越高能力越强，相反，次数越高在训练集外越容易发散，以order=11为例

![[Pasted image 20260518164017.png]]
