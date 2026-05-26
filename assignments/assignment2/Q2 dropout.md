### 前向传播和反向传播

肥肠的简单：
```python
if mode == "train":
        mask=(np.random.rand(*x.shape)<p)/p
        out=x*mask
    elif mode == "test":
        out=x
...
cache = (dropout_param, mask)
out = out.astype(x.dtype, copy=False)
```

```python
if mode == "train":
        dx=dout*mask
    elif mode == "test":
        dx = dout
    return dx
```

唯一要注意的就是 vanilla dropout 和 inverse dropout 的区别，应该使用后者,参考[[Lecture06 CNN Architectures#训练与测试的期望一致性]]和[CS231n Deep Learning for Computer Vision](https://cs231n.github.io/neural-networks-2/#reg)
