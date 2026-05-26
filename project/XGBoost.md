[Introduction to Boosted Trees — xgboost 3.2.1 documentation](https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/model.html)

## 总流程

\[开始]
  │
  ├─► 【初始化】
  │     ├── 设定初始全局预测值
  │     └── 配置全局超参数：n_estimators=3, learning_rate=0.5
  │
  ├─► 【训练第 1 棵树】 
  │     ├── 1. 计算样本一阶导 $I_i$ 与二阶导 $H_i$
  │     ├── 2. 遍历特征与切分点，计算 $Gain$，通过精确贪心算法选择最优切分
  │     ├── 3. 达到 max_depth=2 或满足剪枝条件（$Gain < \gamma$）时停止生长
  │     ├── 4. 计算叶子节点输出权重：$w = -\frac{\sum I_i}{\sum H_i + \lambda}$
  │     │
  │     └───► 【冻结第 1 棵树】
  │           │
  │           └─► 更新当前总预测值并计算残差
  │
  ├─► 【创建第 2 棵树】
  │     ├── 1. 针对上一步更新后的残差计算新的 $I_i$ 和 $H_i$
  │     ├── 2. 重复特征选择与切分过程，限制最大深度为 max_depth=2
  │     ├── 3. 计算叶子节点输出权重（受超参数 $\lambda$ 与 $\gamma$ 约束）
  │     │
  │     └───► 【冻结第 2 棵树】
  │           │
  │           └─► 更新当前总预测值并计算残差
  │
  ├─► 【创建第 3 棵树】
  │     ├── 1. 针对最新残差计算 $I_i$ 和 $H_i$
  │     └── 2. 重复构建流程，达到 max_depth=2 后停止
  │     │
  │     └───► 【冻结第 3 棵树】
  │
\[结束] 达到指定的 n_estimators=3，训练终止

## 核心函数：

正则化损失函数：$$f_t(x) = w_{q(x)}, w \in R^T, q:R^d\rightarrow \{1,2,\cdots,T\} .$$   $$\omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
最优目标缩减函数,即最佳叶子权重：$$\begin{split}w_j^\ast &= -\frac{G_j}{H_j+\lambda}\\
\text{obj}^\ast &= -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda} + \gamma T\end{split}$$
分裂增益函数，只有当Gain>0时才能切分叶子节点：$$Gain = \frac{1}{2} \left[ \frac{(\sum_{i \in L} I_i)^2}{\sum_{i \in L} H_i + \lambda} + \frac{(\sum_{i \in R} I_i)^2}{\sum_{i \in R} H_i + \lambda} - \frac{(\sum_{i \in I} I_i)^2}{\sum_{i \in I} H_i + \lambda} \right] - \gamma$$
