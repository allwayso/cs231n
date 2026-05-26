**Nesterov Momentum** is a slightly different version of the momentum update that has recently been gaining popularity. It enjoys stronger theoretical converge guarantees for convex functions and in practice it also consistenly works slightly better than standard momentum.  
Nesterov Momentum 是动量更新的一种略有不同的变体，最近越来越受欢迎。它对于凸函数具有更强的理论收敛保证，在实践中也始终比标准动量表现稍好一些。

The core idea behind Nesterov momentum is that when the current parameter vector is at some position `x`, then looking at the momentum update above, we know that the momentum term alone (i.e. ignoring the second term with the gradient) is about to nudge the parameter vector by `mu * v`. Therefore, if we are about to compute the gradient, we can treat the future approximate position `x + mu * v` as a “lookahead” - this is a point in the vicinity of where we are soon going to end up. Hence, it makes sense to compute the gradient at `x + mu * v` instead of at the “old/stale” position `x`.  
Nesterov 动量的核心思想是，当当前参数向量处于某个位置 `x` 时，观察上面的动量更新公式，我们知道仅动量项（即忽略含梯度的第二项）就会将参数向量推动 `mu * v` 。因此，如果我们即将计算梯度，可以将未来的近似位置 `x + mu * v` 视为一个"前瞻"——这是我们很快将要到达的位置附近的一个点。因此，在 `x + mu * v` 而非"旧的/过时的"位置 `x` 处计算梯度是合理的。

![](https://cs231n.github.io/assets/nn3/nesterov.jpeg)

Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this "looked-ahead" position.  
Nesterov 动量。我们知道动量即将把我们带到绿色箭头的尖端，而不是在当前位置（红色圆圈）处评估梯度。因此，使用 Nesterov 动量时，我们在这个"前瞻"位置评估梯度。

That is, in a slightly awkward notation, we would like to do the following:  
也就是说，用一种稍微别扭的记号，我们希望执行以下操作：

```
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```

However, in practice people prefer to express the update to look as similar to vanilla SGD or to the previous momentum update as possible. This is possible to achieve by manipulating the update above with a variable transform `x_ahead = x + mu * v`, and then expressing the update in terms of `x_ahead` instead of `x`. That is, the parameter vector we are actually storing is always the ahead version. The equations in terms of `x_ahead` (but renaming it back to `x`) then become:  
然而，在实践中，人们更倾向于让更新式看起来尽可能与普通 SGD 或之前的动量更新相似。这可以通过对上述更新进行变量变换 `x_ahead = x + mu * v` 来实现，然后用 `x_ahead` 而非 `x` 来表达更新。也就是说，我们实际存储的参数向量始终是超前版本。那么，用 `x_ahead` 表示的方程（但将其重新命名为 `x` ）则变为：

```
v_prev = v # back this up
v = mu * v - learning_rate * dx # velocity update stays the same
x += -mu * v_prev + (1 + mu) * v # position update changes form
```

We recommend this further reading to understand the source of these equations and the mathematical formulation of Nesterov’s Accelerated Momentum (NAG):  
我们建议阅读以下资料，以理解这些方程的来源以及 Nesterov 加速动量（NAG）的数学形式：

- [Advances in optimizing Recurrent Networks](http://arxiv.org/pdf/1212.0901v2.pdf) by Yoshua Bengio, Section 3.5.  
    Yoshua Bengio 的《Advances in optimizing Recurrent Networks》，第 3.5 节。
- [Ilya Sutskever’s thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) (pdf) contains a longer exposition of the topic in section 7.2  
    Ilya Sutskever 的论文（pdf）在第 7.2 节对该主题有更详细的阐述。