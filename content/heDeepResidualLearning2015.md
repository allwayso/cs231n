# Deep Residual Learning for Image Recognition

> **作者:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
> **发表:**  (2015)
> **引用键:** `heDeepResidualLearning2015a`
> **类型:** preprint

---

## 📋 摘要

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

---

## 核心贡献

解决了神经网络深度变深时的训练问题

## 方法

### 残差块（Residential Block）

设理想映射为 $\mathcal{H}(x)$。传统网络直接学习 $\mathcal{H}(x)$。ResNet 将堆叠的层转而学习一个残差映射：

$$
\mathcal{F}(x) := \mathcal{H}(x) - x
$$

于是原始映射被重写为：

$$
\mathcal{H}(x) = \mathcal{F}(x) + x
$$

前向传播通过 shortcut connection 实现加法：

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

其中 $\mathcal{F}$ 通常包含两层或三层卷积。

### Why res block

aspect1：通过+x来保证反向传播时梯度不会消失，每次通过残差块至少会保留1

aspect2：恒等映射假设——如果目标函数接近恒等映射，那么神经网络逼近0比逼近x更为简单，因为如果逼近x

aspect3：加入残差块会使得损失景观更平滑，在多轮训练下不容易由凸性变为非凸性[[
## 学到什么

神秘啊，感觉这种重要的论文采用的方法其实挺简单的

## 疑问

他为什么会想到这个方法的


---

