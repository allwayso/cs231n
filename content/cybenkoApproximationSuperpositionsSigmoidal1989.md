# Approximation by superpositions of a sigmoidal function

> **作者:** G. Cybenko
> **发表:** Mathematics of Control, Signals and Systems (1989)
> **引用键:** `cybenkoApproximationSuperpositionsSigmoidal1989`
> **类型:** journalArticle

---

## 📋 摘要

In this paper we demonstrate that finite linear combinations of compositions of a fixed, univariate function and a set of affine functionals can uniformly approximate any continuous function ofn real variables with support in the unit hypercube; only mild conditions are imposed on the univariate function. Our results settle an open question about representability in the class of single hidden layer neural networks. In particular, we show that arbitrary decision regions can be arbitrarily well approximated by continuous feedforward neural networks with only a single internal, hidden layer and any continuous sigmoidal nonlinearity. The paper discusses approximation properties of other possible types of nonlinearities that might be implemented by artificial neural networks.

---

## 核心贡献

这篇论文要回答一个根本性问题：**单隐层前馈神经网络（即形如 $\sum_{j=1}^N \alpha_j \sigma(y_j^T x + \theta_j)$ 的函数）能否逼近任意连续函数？**

> 这里单隐层前馈神经网络指的是一层线性层+一层激活层（隐藏层）+线性加权，N为隐藏层宽度，shape=(N,x.shape\[0]);α为输出层权重

这里的 $\sigma$ 是**S形函数（sigmoidal function）**，满足：

$$
\sigma(t) \to 1 \quad \text{as } t \to +\infty, \qquad \sigma(t) \to 0 \quad \text{as } t \to -\infty
$$

论文证明：只要 $\sigma$ 是连续S形函数，这样的函数类在 $C(I_n)$（$n$ 维单位超立方体上的连续函数空间）中**稠密**

## 方法

泛函分析

## 学到什么

数学真难

## 疑问

数学证明完全看不懂


---

