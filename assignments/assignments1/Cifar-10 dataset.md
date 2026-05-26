## 1. 基本概况

CIFAR-10 是计算机视觉和机器学习领域最经典、使用最广泛的基准数据集之一。它由 Alex Krizhevsky, Vinod Nair 和 Geoffrey Hinton 收集并整理，通常用于评估图像分类算法的性能。

- 图像总量：60,000 张 32x32 彩色图像。
- 类别数量：10 个互斥类别（每类 6,000 张）。
- 数据集划分:
- 训练集：50,000 张图像（分为 5 个 batch 文件）。
- 测试集：10,000 张图像。
## 2. 数据分类

数据集包含以下 10 个类别：
1. 飞机 (airplane)
2. 汽车 (automobile)
3. 鸟 (bird)
4. 猫 (cat)
5. 鹿 (deer)
6. 狗 (dog)
7. 青蛙 (frog)
8. 马 (horse)
9. 船 (ship)
10. 卡车 (truck)

注：类别之间完全互斥，例如“汽车”包含轿车和 SUV，而“卡车”仅包含大卡车。
## 3. 数据结构与存储布局

在 Python 版本中，数据集以 pickle 字典格式存储。
### 字典键值对 (Key-Value)

- b'data':
- 类型：numpy.uint8 数组。
- 形状：(10000, 3072)。
- 布局：每一行存储一张图。前 1024 字节为红色通道 (R)，中间 1024 字节为绿色通道 (G)，最后 1024 字节为蓝色通道 (B)。
- b'labels':
- 类型：包含 10,000 个 0-9 整数的列表。
### 图像重构建议

由于数据是按通道分块存储的，在 Python 中通常需要将其重构为可视化格式：
将 3072 维向量重构为 (通道, 高, 宽)

image = data_row.reshape(3, 32, 32)
转换为 (高, 宽, 通道) 以便用 matplotlib 显示
image = image.transpose(1, 2, 0)
## 4. 相关资源

- 官方主页: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- 引用: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
