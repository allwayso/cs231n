
[CS231n Deep Learning for Computer Vision](https://cs231n.github.io/)[Course Website](http://cs231n.stanford.edu/)

Table of Contents:  目录：

- [Linear Classification  线性分类](https://cs231n.github.io/linear-classify/#linear-classification)
    - [Parameterized mapping from images to label scores  
        从图像到标签分数的参数化映射](https://cs231n.github.io/linear-classify/#parameterized-mapping-from-images-to-label-scores)
    - [Interpreting a linear classifier  
        解释线性分类器](https://cs231n.github.io/linear-classify/#interpreting-a-linear-classifier)
    - [Loss function  损失函数](https://cs231n.github.io/linear-classify/#loss-function)
        - [Multiclass Support Vector Machine loss  
            多类支持向量机损失](https://cs231n.github.io/linear-classify/#multiclass-support-vector-machine-loss)
    - [Practical Considerations  
        实际考虑](https://cs231n.github.io/linear-classify/#practical-considerations)
    - [Softmax classifier  Softmax 分类器](https://cs231n.github.io/linear-classify/#softmax-classifier)
    - [SVM vs. Softmax  SVM 与 Softmax](https://cs231n.github.io/linear-classify/#svm-vs-softmax)
    - [Interactive web demo  交互式网络演示](https://cs231n.github.io/linear-classify/#interactive-web-demo)
    - [Summary  摘要](https://cs231n.github.io/linear-classify/#summary)
    - [Further Reading  进一步阅读](https://cs231n.github.io/linear-classify/#further-reading)

## Linear Classification  线性分类

In the last section we introduced the problem of Image Classification, which is the task of assigning a single label to an image from a fixed set of categories. Moreover, we described the k-Nearest Neighbor (kNN) classifier which labels images by comparing them to (annotated) images from the training set. As we saw, kNN has a number of disadvantages:  
在上一节中，我们介绍了图像分类问题，这是一个从一组固定的类别中为图像分配单个标签的任务。此外，我们还描述了 k-近邻（kNN）分类器，它通过将图像与（带注释的）训练集中的图像进行比较来为图像打标签。如我们所见，kNN 存在许多缺点：

- The classifier must _remember_ all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size.  
    分类器必须记住所有训练数据，并将其存储下来，以便将来与测试数据进行比较。这是空间效率低下的，因为数据集很容易就达到吉字节的大小。
- Classifying a test image is expensive since it requires a comparison to all training images.  
    对一个测试图像进行分类是昂贵的，因为它需要与所有训练图像进行比较。

**Overview**. We are now going to develop a more powerful approach to image classification that we will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks. The approach will have two major components: a **score function** that maps the raw data to class scores, and a **loss function** that quantifies the agreement between the predicted scores and the ground truth labels. We will then cast this as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function.  
概述。我们现在将开发一种更强大的图像分类方法，最终会自然地将其扩展到整个神经网络和卷积神经网络。该方法将包含两个主要组件：一个将原始数据映射到类别分数的评分函数，以及一个量化预测分数与真实标签之间一致性的损失函数。然后我们将将其视为一个优化问题，在这个问题中，我们将通过最小化损失函数来优化评分函数的参数。

### Parameterized mapping from images to label scores  
从图像到标签分数的参数化映射

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. We will develop the approach with a concrete example. As before, let’s assume a training dataset of images xi∈RD, each associated with a label yi. Here i=1…N and yi∈1…K. That is, we have **N** examples (each with a dimensionality **D**) and **K** distinct categories. For example, in CIFAR-10 we have a training set of **N** = 50,000 images, each with **D** = 32 x 32 x 3 = 3072 pixels, and **K** = 10, since there are 10 distinct classes (dog, cat, car, etc). We will now define the score function f:RD↦RK that maps the raw image pixels to class scores.  
这种方法的第一步是定义一个分数函数，该函数将图像的像素值映射到每个类别的置信度分数。我们将通过一个具体的例子来开发这种方法。和之前一样，我们假设一个训练数据集包含图像 xi∈RD ，每个图像都关联一个标签 yi 。这里 i=1…N 和 yi∈1…K 。也就是说，我们有 N 个示例（每个示例的维度为 D）和 K 个不同的类别。例如，在 CIFAR-10 中，我们有一个包含 N = 50,000 张图像的训练集，每张图像的维度为 D = 32 x 32 x 3 = 3072 个像素，且 K = 10，因为有 10 个不同的类别（狗、猫、汽车等）。现在我们将定义分数函数 f:RD↦RK ，该函数将原始图像像素映射到类别分数。

**Linear classifier.** In this module we will start out with arguably the simplest possible function, a linear mapping:  
线性分类器。在这个模块中，我们将从一个可能的最简单的函数开始，即线性映射：

f(xi,W,b)=Wxi+b

In the above equation, we are assuming that the image xi has all of its pixels flattened out to a single column vector of shape [D x 1]. The matrix **W** (of size [K x D]), and the vector **b** (of size [K x 1]) are the **parameters** of the function. In CIFAR-10, xi contains all pixels in the i-th image flattened into a single [3072 x 1] column, **W** is [10 x 3072] and **b** is [10 x 1], so 3072 numbers come into the function (the raw pixel values) and 10 numbers come out (the class scores). The parameters in **W** are often called the **weights**, and **b** is called the **bias vector** because it influences the output scores, but without interacting with the actual data xi. However, you will often hear people use the terms _weights_ and _parameters_ interchangeably.  
在上述公式中，我们假设图像 xi 的所有像素被展平成一个形状为 [D x 1] 的单列向量。矩阵 W（大小为 [K x D]）和向量 b（大小为 [K x 1]）是该函数的参数。在 CIFAR-10 中， xi 包含第 i 张图像的所有像素展平成一个 [3072 x 1] 的单列，W 是 [10 x 3072]，b 是 [10 x 1]，因此有 3072 个数字输入到函数中（原始像素值），并输出 10 个数字（类别分数）。W 中的参数通常被称为权重，b 被称为偏置向量，因为它会影响输出分数，但不会与实际数据 xi 交互。然而，你经常会听到人们将权重和参数这两个词互换使用。

There are a few things to note:  
有几个要点需要注意：

- First, note that the single matrix multiplication Wxi is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of **W**.  
    首先，请注意单个矩阵乘法 Wxi 实际上是在并行评估 10 个不同的分类器（每个类别一个），其中每个分类器是 W 的一行。
- Notice also that we think of the input data (xi,yi) as given and fixed, but we have control over the setting of the parameters **W,b**. Our goal will be to set these in such way that the computed scores match the ground truth labels across the whole training set. We will go into much more detail about how this is done, but intuitively we wish that the correct class has a score that is higher than the scores of incorrect classes.  
    请注意，我们还认为输入数据 (xi,yi) 是给定且固定的，但我们控制着参数 W、b 的设置。我们的目标是将这些参数设置得使计算出的分数在整个训练集中与真实标签相匹配。我们将更详细地讨论如何做到这一点，但直观上我们希望正确类别的分数比错误类别的分数更高。
- An advantage of this approach is that the training data is used to learn the parameters **W,b**, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.  
    这种方法的优点是训练数据用于学习参数 W、b，但一旦学习完成，我们可以丢弃整个训练集，只保留学习到的参数。这是因为新的测试图像可以简单地通过函数传递，并根据计算出的分数进行分类。
- Lastly, note that classifying the test image involves a single matrix multiplication and addition, which is significantly faster than comparing a test image to all training images.  
    最后，请注意，对测试图像进行分类涉及一次矩阵乘法和加法，这比将测试图像与所有训练图像进行比较要快得多。

> Foreshadowing: Convolutional Neural Networks will map image pixels to scores exactly as shown above, but the mapping ( f ) will be more complex and will contain more parameters.  
> 预示：卷积神经网络将像上面所示那样将图像像素映射到分数，但映射（f）将更复杂，并且包含更多参数。

### Interpreting a linear classifier  
解释线性分类器

Notice that a linear classifier computes the score of a class as a weighted sum of all of its pixel values across all 3 of its color channels. Depending on precisely what values we set for these weights, the function has the capacity to like or dislike (depending on the sign of each weight) certain colors at certain positions in the image. For instance, you can imagine that the “ship” class might be more likely if there is a lot of blue on the sides of an image (which could likely correspond to water). You might expect that the “ship” classifier would then have a lot of positive weights across its blue channel weights (presence of blue increases score of ship), and negative weights in the red/green channels (presence of red/green decreases the score of ship).  
请注意，线性分类器将一个类别的分数计算为其所有像素值在 3 个颜色通道上的加权和。根据我们为这些权重设定的具体值，该函数有能力在图像的特定位置喜欢或不喜欢某些颜色（取决于每个权重的符号）。例如，你可以想象如果图像的两侧有很多蓝色（这很可能对应于水），那么“船”类可能会更有可能。你可能期望“船”分类器在其蓝色通道权重上有很多正权重（蓝色的存在会增加船的分数），而在红色/绿色通道上有负权重（红色/绿色的存在会降低船的分数）。

![](https://cs231n.github.io/assets/imagemap.jpg)

An example of mapping an image to class scores. For the sake of visualization, we assume the image only has 4 pixels (4 monochrome pixels, we are not considering color channels in this example for brevity), and that we have 3 classes (red (cat), green (dog), blue (ship) class). (Clarification: in particular, the colors here simply indicate 3 classes and are not related to the RGB channels.) We stretch the image pixels into a column and perform matrix multiplication to get the scores for each class. Note that this particular set of weights W is not good at all: the weights assign our cat image a very low cat score. In particular, this set of weights seems convinced that it's looking at a dog.  
将图像映射到类别分数的一个例子。为了可视化，我们假设图像只有 4 个像素（4 个单色像素，为了简洁起见，本例中不考虑颜色通道），并且有 3 个类别（红色（猫）、绿色（狗）、蓝色（船）类别）。（说明：这里的具体颜色仅表示 3 个类别，与 RGB 通道无关。）我们将图像像素拉直成一列，并进行矩阵乘法以得到每个类别的分数。请注意，这组特定的权重 W 非常糟糕：权重将我们的猫图像的猫分数分配得非常低。特别是，这组权重似乎确信它正在看一只狗。

**Analogy of images as high-dimensional points.** Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space (e.g. each image in CIFAR-10 is a point in 3072-dimensional space of 32x32x3 pixels). Analogously, the entire dataset is a (labeled) set of points.  
将图像视为高维点的类比。由于图像被拉伸成高维列向量，我们可以将每张图像解释为该空间中的一个点（例如，CIFAR-10 中的每张图像都是 3072 维空间中 32x32x3 像素的一个点）。类似地，整个数据集是一个（标记的）点集。

Since we defined the score of each class as a weighted sum of all image pixels, each class score is a linear function over this space. We cannot visualize 3072-dimensional spaces, but if we imagine squashing all those dimensions into only two dimensions, then we can try to visualize what the classifier might be doing:  
由于我们将每个类别的分数定义为所有图像像素的加权总和，因此每个类别的分数是空间上的线性函数。我们无法可视化 3072 维空间，但如果我们将所有这些维度压缩到只有两个维度，我们就可以尝试想象分类器可能正在做什么：

![](https://cs231n.github.io/assets/pixelspace.jpeg)

Cartoon representation of the image space, where each image is a single point, and three classifiers are visualized. Using the example of the car classifier (in red), the red line shows all points in the space that get a score of zero for the car class. The red arrow shows the direction of increase, so all points to the right of the red line have positive (and linearly increasing) scores, and all points to the left have a negative (and linearly decreasing) scores.  
图像空间的卡通表示，其中每张图像是一个点，并可视化了三个分类器。以汽车分类器（红色）为例，红线显示了空间中所有对汽车类别得分为零的点。红箭头显示了增加的方向，因此红线右侧的所有点都有正（且线性增加）的分数，而左侧的所有点都有负（且线性减少）的分数。

As we saw above, every row of W is a classifier for one of the classes. The geometric interpretation of these numbers is that as we change one of the rows of W, the corresponding line in the pixel space will rotate in different directions. The biases b, on the other hand, allow our classifiers to translate the lines. In particular, note that without the bias terms, plugging in xi=0 would always give score of zero regardless of the weights, so all lines would be forced to cross the origin.  
如上所述， W 的每一行都是一个分类器，用于分类中的一个类别。这些数字的几何解释是，当我们改变 W 的一行时，像素空间中的对应线会向不同方向旋转。另一方面，偏差 b 允许我们的分类器平移这些线。特别地，请注意如果没有偏差项，无论权重如何，将 xi=0 代入总会得到零分，因此所有线都会被强制穿过原点。

**Interpretation of linear classifiers as template matching.** Another interpretation for the weights W is that each row of W corresponds to a _template_ (or sometimes also called a _prototype_) for one of the classes. The score of each class for an image is then obtained by comparing each template with the image using an _inner product_ (or _dot product_) one by one to find the one that “fits” best. With this terminology, the linear classifier is doing template matching, where the templates are learned. Another way to think of it is that we are still effectively doing Nearest Neighbor, but instead of having thousands of training images we are only using a single image per class (although we will learn it, and it does not necessarily have to be one of the images in the training set), and we use the (negative) inner product as the distance instead of the L1 or L2 distance.  
线性分类器的解释作为模板匹配。权重 W 的另一种解释是 W 的每一行对应于一个类别的模板（有时也称为原型）。然后通过逐个使用内积（或点积）将每个模板与图像进行比较，找到最匹配的那个，从而得到图像每个类别的分数。用这种术语来说，线性分类器正在进行模板匹配，其中模板是学习得到的。另一种思考方式是，我们仍然在有效地进行最近邻分类，但我们只使用每个类别的一幅图像（尽管我们会学习它，并且它不一定必须是训练集中的图像），而使用（负）内积作为距离，而不是 L1 或 L2 距离。

![](https://cs231n.github.io/assets/templates.jpg)

Skipping ahead a bit: Example learned weights at the end of learning for CIFAR-10. Note that, for example, the ship template contains a lot of blue pixels as expected. This template will therefore give a high score once it is matched against images of ships on the ocean with an inner product.  
稍微跳过一些内容：CIFAR-10 学习结束时的示例学习权重。请注意，例如，船模板中包含很多蓝色像素，这是符合预期的。因此，当这个模板与海洋中船只的图像进行内积匹配时，会给出高分。

Additionally, note that the horse template seems to contain a two-headed horse, which is due to both left and right facing horses in the dataset. The linear classifier _merges_ these two modes of horses in the data into a single template. Similarly, the car classifier seems to have merged several modes into a single template which has to identify cars from all sides, and of all colors. In particular, this template ended up being red, which hints that there are more red cars in the CIFAR-10 dataset than of any other color. The linear classifier is too weak to properly account for different-colored cars, but as we will see later neural networks will allow us to perform this task. Looking ahead a bit, a neural network will be able to develop intermediate neurons in its hidden layers that could detect specific car types (e.g. green car facing left, blue car facing front, etc.), and neurons on the next layer could combine these into a more accurate car score through a weighted sum of the individual car detectors.  
此外，请注意马模板似乎包含了一匹双头马，这是由于数据集中同时存在面向左和面向右的马。线性分类器将数据中的这两种马模式合并为一个模板。类似地，汽车分类器似乎也将几种模式合并为一个模板，该模板需要识别来自所有方向、所有颜色的汽车。特别是，这个模板最终变成了红色，这表明在 CIFAR-10 数据集中红色汽车比其他任何颜色的汽车都多。线性分类器太弱，无法正确处理不同颜色的汽车，但正如我们稍后将看到的，神经网络将允许我们完成这项任务。稍微向前看，神经网络将能够在其隐藏层中发展出中间神经元，这些神经元可以检测特定的汽车类型（例如，左侧的绿色汽车、前方的蓝色汽车等），而下一层的神经元可以通过对各个汽车检测器的加权求和将这些信息组合起来，从而得到更准确的汽车分数。

**Bias trick.** Before moving on we want to mention a common simplifying trick to representing the two parameters W,b as one. Recall that we defined the score function as:  
偏差技巧。在继续之前，我们想提到一个常见的简化技巧，将两个参数 W,b 表示为一个。回想一下，我们定义了得分函数为：

f(xi,W,b)=Wxi+b

As we proceed through the material it is a little cumbersome to keep track of two sets of parameters (the biases b and weights W) separately. A commonly used trick is to combine the two sets of parameters into a single matrix that holds both of them by extending the vector xi with one additional dimension that always holds the constant 1 - a default _bias dimension_. With the extra dimension, the new score function will simplify to a single matrix multiply:  
随着我们学习这些材料，分别跟踪两组参数（偏差 b 和权重 W ）会有些繁琐。一个常用的技巧是将两组参数合并成一个矩阵，该矩阵同时包含它们，通过扩展向量 xi 来增加一个额外的维度，该维度始终包含常数 1 —— 一个默认的偏差维度。有了这个额外的维度，新的得分函数将简化为单个矩阵乘法：

f(xi,W)=Wxi

With our CIFAR-10 example, xi is now [3073 x 1] instead of [3072 x 1] - (with the extra dimension holding the constant 1), and W is now [10 x 3073] instead of [10 x 3072]. The extra column that W now corresponds to the bias b. An illustration might help clarify:  
以我们的 CIFAR-10 示例为例， xi 现在是[3073 x 1]而不是[3072 x 1]——（额外的维度存储常数 1），而 W 现在是[10 x 3073]而不是[10 x 3072]。 W 现在对应的额外列是偏差 b 。一个图示或许能帮助说明：

![](https://cs231n.github.io/assets/wb.jpeg)

Illustration of the bias trick. Doing a matrix multiplication and then adding a bias vector (left) is equivalent to adding a bias dimension with a constant of 1 to all input vectors and extending the weight matrix by 1 column - a bias column (right). Thus, if we preprocess our data by appending ones to all vectors we only have to learn a single matrix of weights instead of two matrices that hold the weights and the biases.  
偏置技巧的图示。进行矩阵乘法然后加上偏置向量（左侧）等同于给所有输入向量添加一个偏置维度，其常数为 1，并将权重矩阵扩展 1 列——即偏置列（右侧）。因此，如果我们通过在所有向量后添加 1 来预处理数据，我们只需学习一个权重矩阵，而不是两个分别存储权重和偏置的矩阵。

**Image data preprocessing.** As a quick note, in the examples above we used the raw pixel values (which range from [0…255]). In Machine Learning, it is a very common practice to always perform normalization of your input features (in the case of images, every pixel is thought of as a feature). In particular, it is important to **center your data** by subtracting the mean from every feature. In the case of images, this corresponds to computing a _mean image_ across the training images and subtracting it from every image to get images where the pixels range from approximately [-127 … 127]. Further common preprocessing is to scale each input feature so that its values range from [-1, 1]. Of these, zero mean centering is arguably more important but we will have to wait for its justification until we understand the dynamics of gradient descent.  
图像数据预处理。简单来说，在上述示例中我们使用了原始像素值（范围在[0…255]）。在机器学习中，对输入特征进行归一化是一种非常常见的做法（对于图像来说，每个像素被视为一个特征）。特别是，通过从每个特征中减去均值来对数据进行中心化非常重要。对于图像来说，这对应于计算训练图像的均值图像，并从每个图像中减去它，以得到像素值范围大约在[-127 … 127]的图像。进一步常见的预处理是对每个输入特征进行缩放，使其值范围在[-1, 1]。在这些方法中，零均值中心化可能更为重要，但我们只有在理解梯度下降的动态之后才能对其合理性进行解释。

### Loss function  损失函数

In the previous section we defined a function from the pixel values to class scores, which was parameterized by a set of weights W. Moreover, we saw that we don’t have control over the data (xi,yi) (it is fixed and given), but we do have control over these weights and we want to set them so that the predicted class scores are consistent with the ground truth labels in the training data.  
在上一节中，我们定义了一个从像素值到类分数的函数，该函数由一组权重 W 参数化。此外，我们看到我们无法控制数据 (xi,yi) （它是固定的且已给出），但我们确实可以控制这些权重，并且我们希望设置它们，使得预测的类分数与训练数据中的真实标签一致。

For example, going back to the example image of a cat and its scores for the classes “cat”, “dog” and “ship”, we saw that the particular set of weights in that example was not very good at all: We fed in the pixels that depict a cat but the cat score came out very low (-96.8) compared to the other classes (dog score 437.9 and ship score 61.95). We are going to measure our unhappiness with outcomes such as this one with a **loss function** (or sometimes also referred to as the **cost function** or the **objective**). Intuitively, the loss will be high if we’re doing a poor job of classifying the training data, and it will be low if we’re doing well.  
例如，回到之前猫的示例图像及其在“猫”、“狗”和“船”这三个类别上的得分，我们看到该示例中的特定权重集表现非常糟糕：我们输入了描绘猫的像素，但猫的得分却非常低（-96.8），与其他类别相比（狗的得分 437.9 和船的得分 61.95）。我们将用损失函数（有时也称为成本函数或目标函数）来衡量我们对这类结果的不满程度。直观上，如果我们对训练数据的分类工作做得不好，损失就会很高；如果我们做得好，损失就会很低。

#### Multiclass Support Vector Machine loss  
多类支持向量机损失

There are several ways to define the details of the loss function. As a first example we will first develop a commonly used loss called the **Multiclass Support Vector Machine** (SVM) loss. The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. Notice that it’s sometimes helpful to anthropomorphise the loss functions as we did above: The SVM “wants” a certain outcome in the sense that the outcome would yield a lower loss (which is good).  
定义损失函数的细节有几种方法。作为第一个例子，我们将首先开发一个常用的损失函数，称为多类支持向量机（SVM）损失。SVM 损失被设置成使得 SVM“希望”每个图像的正确类别得分比错误类别高出一定的固定间隔 Δ 。请注意，有时将损失函数拟人化（如上所述）是有帮助的：SVM“希望”得到某种结果，因为这种结果会产生较低的损失（这是好的）。

Let’s now get more precise. Recall that for the i-th example we are given the pixels of image xi and the label yi that specifies the index of the correct class. The score function takes the pixels and computes the vector f(xi,W) of class scores, which we will abbreviate to s (short for scores). For example, the score for the j-th class is the j-th element: sj=f(xi,W)j. The Multiclass SVM loss for the i-th example is then formalized as follows:  
现在让我们更加精确地说明。回想一下，对于第 i 个示例，我们被给出图像 xi 的像素和指定正确类别索引的标签 yi 。分数函数接收像素并计算类别分数向量 f(xi,W) ，我们将它缩写为 s （简称分数）。例如，第 j 个类别的分数是第 j 个元素： sj=f(xi,W)j 。因此，第 i 个示例的多类支持向量机损失可以形式化为如下：

Li=∑j≠yimax(0,sj−syi+Δ)

**Example.** Lets unpack this with an example to see how it works. Suppose that we have three classes that receive the scores s=[13,−7,11], and that the first class is the true class (i.e. yi=0). Also assume that Δ (a hyperparameter we will go into more detail about soon) is 10. The expression above sums over all incorrect classes (j≠yi), so we get two terms:  
示例。我们用一个例子来解析它是如何工作的。假设我们有三个类别，它们得到的分数分别是 s=[13,−7,11] ，并且第一个类别是真实类别（即 yi=0 ）。同时假设 Δ （一个我们很快会详细讨论的超参数）是 10。上面的表达式对所有错误类别（ j≠yi ）求和，所以我们得到两项：

Li=max(0,−7−13+10)+max(0,11−13+10)

You can see that the first term gives zero since [-7 - 13 + 10] gives a negative number, which is then thresholded to zero with the max(0,−) function. We get zero loss for this pair because the correct class score (13) was greater than the incorrect class score (-7) by at least the margin 10. In fact the difference was 20, which is much greater than 10 but the SVM only cares that the difference is at least 10; Any additional difference above the margin is clamped at zero with the max operation. The second term computes [11 - 13 + 10] which gives 8. That is, even though the correct class had a higher score than the incorrect class (13 > 11), it was not greater by the desired margin of 10. The difference was only 2, which is why the loss comes out to 8 (i.e. how much higher the difference would have to be to meet the margin). In summary, the SVM loss function wants the score of the correct class yi to be larger than the incorrect class scores by at least by Δ (delta). If this is not the case, we will accumulate loss.  
你可以看到，第一项为零，因为 [-7 - 13 + 10] 得到一个负数，然后通过 max(0,−) 函数被阈值化为零。这对的损失为零，因为正确类别的分数（13）比错误类别的分数（-7）至少大 10 的间隔。实际上差异是 20，这远大于 10，但 SVM 只关心差异至少为 10；超过间隔的任何额外差异都会通过 max 操作被钳位为零。第二项计算 [11 - 13 + 10]，结果为 8。也就是说，尽管正确类别的分数比错误类别高（13 > 11），但它没有达到期望的 10 的间隔。差异只有 2，这就是为什么损失为 8（即差异需要高出多少才能满足间隔）。总之，SVM 损失函数希望正确类别的分数 yi 比错误类别的分数至少大 Δ （delta）。如果不是这样，我们就会累积损失。

Note that in this particular module we are working with linear score functions ( f(xi;W)=Wxi ), so we can also rewrite the loss function in this equivalent form:  
请注意，在这个特定模块中我们使用的是线性评分函数（ f(xi;W)=Wxi ），因此我们也可以将损失函数以这种等价形式重写：

Li=∑j≠yimax(0,wTjxi−wTyixi+Δ)

where wj is the j-th row of W reshaped as a column. However, this will not necessarily be the case once we start to consider more complex forms of the score function f.  
wj 是 W 重构为列的 j-行。然而，一旦我们开始考虑更复杂的分数函数 f 的形式，情况就不一定是这样。

A last piece of terminology we’ll mention before we finish with this section is that the threshold at zero max(0,−) function is often called the **hinge loss**. You’ll sometimes hear about people instead using the squared hinge loss SVM (or L2-SVM), which uses the form max(0,−)2 that penalizes violated margins more strongly (quadratically instead of linearly). The unsquared version is more standard, but in some datasets the squared hinge loss can work better. This can be determined during cross-validation.  
在本节结束前，我们还要提到最后一个术语，即零阈值 max(0,−) 函数通常被称为铰链损失。你有时会听到人们使用平方铰链损失 SVM（或 L2-SVM）的情况，它使用形式 max(0,−)2 来更强烈地惩罚违反边界的部分（二次方而不是线性）。未平方的版本更标准，但在某些数据集上平方铰链损失可能效果更好。这可以在交叉验证过程中确定。

> The loss function quantifies our unhappiness with predictions on the training set  
> 损失函数量化了我们对训练集上预测的不满程度

![](https://cs231n.github.io/assets/margin.jpg)

The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.  
多类支持向量机“希望”正确类别的分数比所有其他分数至少高 delta。如果任何类别的分数在红色区域内（或更高），则会有累积损失。否则损失将为零。我们的目标将是找到能够同时满足训练数据中所有样本的约束条件的权重，并使总损失尽可能低。  

**Regularization**. There is one bug with the loss function we presented above. Suppose that we have a dataset and a set of parameters **W** that correctly classify every example (i.e. all scores are so that all the margins are met, and Li=0 for all i). The issue is that this set of **W** is not necessarily unique: there might be many similar **W** that correctly classify the examples. One easy way to see this is that if some parameters **W** correctly classify all examples (so loss is zero for each example), then any multiple of these parameters λW where λ>1 will also give zero loss because this transformation uniformly stretches all score magnitudes and hence also their absolute differences. For example, if the difference in scores between a correct class and a nearest incorrect class was 15, then multiplying all elements of **W** by 2 would make the new difference 30.  
正则化。我们上面给出的损失函数存在一个缺陷。假设我们有一个数据集和一组参数 W，它们能够正确分类每个样本（即所有得分都满足所有间隔条件，并且对于所有 i， Li=0 ）。问题在于，这组 W 不一定唯一：可能存在许多相似的 W 也能正确分类样本。一个简单的方法是，如果某些参数 W 能正确分类所有样本（即每个样本的损失为零），那么这些参数的任何倍数 λW （其中 λ>1 ）也会给出零损失，因为这种变换会均匀地拉伸所有得分的大小，从而也拉伸了它们的绝对差值。例如，如果正确类别和最近错误类别之间的得分差是 15，那么将 W 的所有元素乘以 2 会使新的差值变为 30。

In other words, we wish to encode some preference for a certain set of weights **W** over others to remove this ambiguity. We can do so by extending the loss function with a **regularization penalty** R(W). The most common regularization penalty is the squared **L2** norm that discourages large weights through an elementwise quadratic penalty over all parameters:  
换句话说，我们希望对一组权重 W 相对于其他权重赋予一定的偏好，以消除这种模糊性。我们可以通过扩展损失函数添加正则化惩罚 R(W) 来实现这一点。最常见的正则化惩罚是平方 L2 范数，它通过在所有参数上对每个元素施加二次惩罚来抑制大的权重：

R(W)=∑k∑lW2k,l

In the expression above, we are summing up all the squared elements of W. Notice that the regularization function is not a function of the data, it is only based on the weights. Including the regularization penalty completes the full Multiclass Support Vector Machine loss, which is made up of two components: the **data loss** (which is the average loss Li over all examples) and the **regularization loss**. That is, the full Multiclass SVM loss becomes:  
在上述表达式中，我们对 W 的所有平方元素求和。请注意，正则化函数不是数据函数，它仅基于权重。包含正则化惩罚完成了完整的多元支持向量机损失，该损失由两个部分组成：数据损失（即所有样本的平均损失 Li ）和正则化损失。也就是说，完整的多元 SVM 损失变为：

L=1N∑iLidata loss+λR(W)regularization loss

Or expanding this out in its full form:  
或者以完整形式展开：

L=1N∑i∑j≠yi[max(0,f(xi;W)j−f(xi;W)yi+Δ)]+λ∑k∑lW2k,l

Where N is the number of training examples. As you can see, we append the regularization penalty to the loss objective, weighted by a hyperparameter λ. There is no simple way of setting this hyperparameter and it is usually determined by cross-validation.  
其中 N 是训练样本的数量。如您所见，我们将正则化惩罚附加到损失目标上，其权重由超参数 λ 调整。没有简单的方法来设置这个超参数，它通常通过交叉验证来确定。

In addition to the motivation we provided above there are many desirable properties to include the regularization penalty, many of which we will come back to in later sections. For example, it turns out that including the L2 penalty leads to the appealing **max margin** property in SVMs (See [CS229](http://cs229.stanford.edu/notes/cs229-notes3.pdf) lecture notes for full details if you are interested).  
除了我们上面提供的动机之外，包含正则化惩罚还有许多理想特性，其中许多我们将在后面的章节中再次讨论。例如，事实证明包含 L2 惩罚会导致支持向量机（SVM）中引人入胜的最大间隔特性（如果您感兴趣，可以参考 CS229 课程笔记了解详细信息）。

The most appealing property is that penalizing large weights tends to improve generalization, because it means that no input dimension can have a very large influence on the scores all by itself. For example, suppose that we have some input vector x=[1,1,1,1] and two weight vectors w1=[1,0,0,0], w2=[0.25,0.25,0.25,0.25]. Then wT1x=wT2x=1 so both weight vectors lead to the same dot product, but the L2 penalty of w1 is 1.0 while the L2 penalty of w2 is only 0.25. Therefore, according to the L2 penalty the weight vector w2 would be preferred since it achieves a lower regularization loss. Intuitively, this is because the weights in w2 are smaller and more diffuse. Since the L2 penalty prefers smaller and more diffuse weight vectors, the final classifier is encouraged to take into account all input dimensions to small amounts rather than a few input dimensions and very strongly. As we will see later in the class, this effect can improve the generalization performance of the classifiers on test images and lead to less _overfitting_.  
最吸引人的特性是惩罚大的权重往往会提高泛化能力，因为这意味着没有任何一个输入维度能够单独对分数产生非常大的影响。例如，假设我们有一个输入向量 x=[1,1,1,1] 和两个权重向量 w1=[1,0,0,0] 、 w2=[0.25,0.25,0.25,0.25] 。那么 wT1x=wT2x=1 ，所以这两个权重向量导致相同的点积，但 w1 的 L2 惩罚是 1.0，而 w2 的 L2 惩罚只有 0.25。因此，根据 L2 惩罚，权重向量 w2 会被优先选择，因为它实现了更低的正则化损失。直观上，这是因为 w2 中的权重更小且更分散。由于 L2 惩罚倾向于更小、更分散的权重向量，最终的分类器被鼓励对所有的输入维度都考虑少量，而不是对少数几个输入维度非常强烈。正如我们将在课程后面看到的，这种效果可以提高分类器在测试图像上的泛化性能，并减少过拟合。

Note that biases do not have the same effect since, unlike the weights, they do not control the strength of influence of an input dimension. Therefore, it is common to only regularize the weights W but not the biases b. However, in practice this often turns out to have a negligible effect. Lastly, note that due to the regularization penalty we can never achieve loss of exactly 0.0 on all examples, because this would only be possible in the pathological setting of W=0.  
需要注意的是，偏差的影响并不相同，因为它们不像权重那样控制输入维度的强弱影响。因此，通常只正则化权重 W 而不正则化偏差 b 。然而，在实践中，这通常被证明几乎没有效果。最后，需要注意的是，由于正则化惩罚，我们永远无法在所有示例上实现精确的 0.0 损失，因为这种情况只有在 W=0 的病态设置下才可能发生。

**Code**. Here is the loss function (without regularization) implemented in Python, in both unvectorized and half-vectorized form:  
代码。以下是 Python 中实现的损失函数（不含正则化），包括未向量化版本和半向量化版本：

```
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in range(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
```

The takeaway from this section is that the SVM loss takes one particular approach to measuring how consistent the predictions on training data are with the ground truth labels. Additionally, making good predictions on the training set is equivalent to minimizing the loss.  
本节的重点在于，SVM 损失采用了一种特定的方法来衡量训练数据上的预测与真实标签的一致性。此外，在训练集上做出良好的预测等同于最小化损失。

> All we have to do now is to come up with a way to find the weights that minimize the loss.  
> 我们现在只需要想出一个方法来找到使损失最小化的权重。

### Practical Considerations  
实际考虑

**Setting Delta.** Note that we brushed over the hyperparameter Δ and its setting. What value should it be set to, and do we have to cross-validate it? It turns out that this hyperparameter can safely be set to Δ=1.0 in all cases. The hyperparameters Δ and λ seem like two different hyperparameters, but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective. The key to understanding this is that the magnitude of the weights W has direct effect on the scores (and hence also their differences): As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become higher. Therefore, the exact value of the margin between the scores (e.g. Δ=1, or Δ=100) is in some sense meaningless because the weights can shrink or stretch the differences arbitrarily. Hence, the only real tradeoff is how large we allow the weights to grow (through the regularization strength λ).  
设置 Delta。请注意，我们之前略过了超参数 Δ 及其设置。它应该设置为多少值，我们是否需要进行交叉验证？事实证明，这个超参数在所有情况下都可以安全地设置为 Δ=1.0 。超参数 Δ 和 λ 看起来像是两个不同的超参数，但实际上它们都控制着同一个权衡：目标函数中数据损失和正则化损失之间的权衡。理解这个问题的关键在于权重 W 的幅度对分数（因此也对其差异）有直接影响：当我们缩小 W 内的所有值时，分数差异会变得较低，而当我们放大权重时，分数差异会全部变得较高。因此，分数之间的边界的确切值（例如 Δ=1 ，或 Δ=100 ）在某种程度上是无意义的，因为权重可以任意地缩小或拉伸差异。因此，唯一真正的权衡是我们允许权重增长多大（通过正则化强度 λ ）。

**Relation to Binary Support Vector Machine**. You may be coming to this class with previous experience with Binary Support Vector Machines, where the loss for the i-th example can be written as:  
与二元支持向量机的关系。你可能带着之前学习二元支持向量机的经验来到这个课程，其中第 i 个样本的损失可以表示为：

Li=Cmax(0,1−yiwTxi)+R(W)

where C is a hyperparameter, and yi∈{−1,1}. You can convince yourself that the formulation we presented in this section contains the binary SVM as a special case when there are only two classes. That is, if we only had two classes then the loss reduces to the binary SVM shown above. Also, C in this formulation and λ in our formulation control the same tradeoff and are related through reciprocal relation C∝1λ.  
其中 C 是一个超参数，而 yi∈{−1,1} 。你可以证明本节中我们提出的公式包含了当只有两个类别时作为特例的二分类支持向量机。也就是说，如果我们只有两个类别，那么损失函数将简化为上面所示的二分类支持向量机。此外，本公式中的 C 和我们公式中的 λ 控制着相同的权衡，并通过互反关系 C∝1λ 相关联。

**Aside: Optimization in primal**. If you’re coming to this class with previous knowledge of SVMs, you may have also heard of kernels, duals, the SMO algorithm, etc. In this class (as is the case with Neural Networks in general) we will always work with the optimization objectives in their unconstrained primal form. Many of these objectives are technically not differentiable (e.g. the max(x,y) function isn’t because it has a _kink_ when x=y), but in practice this is not a problem and it is common to use a subgradient.  
旁注：原始问题中的优化。如果你带着对 SVMs 的先前知识来上这门课，你可能也听说过核函数、对偶问题、SMO 算法等。在这门课中（就像神经网络通常的情况一样），我们将始终使用无约束的原始形式优化目标。其中许多目标在技术上不可微（例如，max(x,y)函数不可微，因为它在 x=y 时有尖点），但在实践中这不是问题，通常使用次梯度。

**Aside: Other Multiclass SVM formulations.** It is worth noting that the Multiclass SVM presented in this section is one of few ways of formulating the SVM over multiple classes. Another commonly used form is the _One-Vs-All_ (OVA) SVM which trains an independent binary SVM for each class vs. all other classes. Related, but less common to see in practice is also the _All-vs-All_ (AVA) strategy. Our formulation follows the [Weston and Watkins 1999 (pdf)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf) version, which is a more powerful version than OVA (in the sense that you can construct multiclass datasets where this version can achieve zero data loss, but OVA cannot. See details in the paper if interested). The last formulation you may see is a _Structured SVM_, which maximizes the margin between the score of the correct class and the score of the highest-scoring incorrect runner-up class. Understanding the differences between these formulations is outside of the scope of the class. The version presented in these notes is a safe bet to use in practice, but the arguably simplest OVA strategy is likely to work just as well (as also argued by Rikin et al. 2004 in [In Defense of One-Vs-All Classification (pdf)](http://www.jmlr.org/papers/volume5/rifkin04a/rifkin04a.pdf)).  
附注：其他多类 SVM 公式。值得注意的是，本节介绍的多类 SVM 是少数几种多类 SVM 公式之一。另一种常用形式是 One-Vs-All（OVA）SVM，它为每个类别与其他所有类别分别训练一个独立的二分类 SVM。相关但实践中较少见的是 All-vs-All（AVA）策略。我们的公式遵循 Weston 和 Watkins 1999 年（pdf）版本，该版本比 OVA 更强大（在某种意义上，你可以构建多类数据集，该版本可以实现零数据损失，而 OVA 不能。如感兴趣可查阅论文了解详情）。你可能还会看到一种结构化 SVM，它最大化正确类别得分与得分最高的错误类别次优得分之间的间隔。理解这些公式的差异超出了本课程的范围。这些笔记中介绍的版本在实践中是安全的，但 arguably 最简单的 OVA 策略可能同样有效（正如 Rikin 等人 2004 年在《捍卫 One-Vs-All 分类》（pdf）中所论证的那样）。

### Softmax classifier  Softmax 分类器

It turns out that the SVM is one of two commonly seen classifiers. The other popular choice is the **Softmax classifier**, which has a different loss function. If you’ve heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes. Unlike the SVM which treats the outputs f(xi,W) as (uncalibrated and possibly difficult to interpret) scores for each class, the Softmax classifier gives a slightly more intuitive output (normalized class probabilities) and also has a probabilistic interpretation that we will describe shortly. In the Softmax classifier, the function mapping f(xi;W)=Wxi stays unchanged, but we now interpret these scores as the unnormalized log probabilities for each class and replace the _hinge loss_ with a **cross-entropy loss** that has the form:  
结果表明，SVM 是两种常见的分类器中的一种。另一种流行的选择是 Softmax 分类器，它具有不同的损失函数。如果你之前听说过二元逻辑回归分类器，那么 Softmax 分类器就是它的多类别推广。与将输出 f(xi,W) 视为（未校准且可能难以解释）的类别分数的 SVM 不同，Softmax 分类器给出一个稍微更直观的输出（归一化类别概率），并且具有我们稍后将描述的概率解释。在 Softmax 分类器中，映射函数 f(xi;W)=Wxi 保持不变，但我们现在将这些分数解释为每个类别的未归一化对数概率，并将铰链损失替换为具有以下形式的交叉熵损失：

Li=−log(efyi∑jefj)or equivalentlyLi=−fyi+log∑jefj

where we are using the notation fj to mean the j-th element of the vector of class scores f. As before, the full loss for the dataset is the mean of Li over all training examples together with a regularization term R(W). The function fj(z)=ezj∑kezk is called the **softmax function**: It takes a vector of arbitrary real-valued scores (in z) and squashes it to a vector of values between zero and one that sum to one. The full cross-entropy loss that involves the softmax function might look scary if you’re seeing it for the first time but it is relatively easy to motivate.  
我们在使用 fj 表示类分数向量 f 的第 j 个元素。和之前一样，数据集的完整损失是所有训练样本的 Li 的平均值，再加上正则化项 R(W) 。函数 fj(z)=ezj∑kezk 被称为 softmax 函数：它将一个任意实数值分数向量（在 z 中）压缩成一个值在 0 和 1 之间且总和为 1 的向量。涉及 softmax 函数的完整交叉熵损失，如果你是第一次看到它可能会觉得吓人，但它相对容易解释。

**Information theory view**. The _cross-entropy_ between a “true” distribution p and an estimated distribution q is defined as:  
信息论视角。真实分布 p 与估计分布 q 之间的交叉熵定义为：

H(p,q)=−∑xp(x)logq(x)

The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ( q=efyi/∑jefj as seen above) and the “true” distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. p=[0,…1,…,0] contains a single 1 at the yi -th position.). Moreover, since the cross-entropy can be written in terms of entropy and the Kullback-Leibler divergence as H(p,q)=H(p)+DKL(p||q), and the entropy of the delta function p is zero, this is also equivalent to minimizing the KL divergence between the two distributions (a measure of distance). In other words, the cross-entropy objective _wants_ the predicted distribution to have all of its mass on the correct answer.  
因此，Softmax 分类器是在最小化估计的类别概率（如上所见）与“真实”分布之间的交叉熵，在这个解释中，“真实”分布是指所有概率质量都在正确类别上的分布（即 p=[0,…1,…,0] 在 yi -th 位置上包含一个 1）。此外，由于交叉熵可以用熵和 Kullback-Leibler 散度表示为 H(p,q)=H(p)+DKL(p||q) ，而 delta 函数 p 的熵为零，因此这也等价于最小化这两个分布之间的 KL 散度（距离的度量）。换句话说，交叉熵目标函数希望预测分布的所有质量都在正确答案上。

**Probabilistic interpretation**. Looking at the expression, we see that  
概率解释。观察这个表达式，我们看到

P(yi∣xi;W)=efyi∑jefj

can be interpreted as the (normalized) probability assigned to the correct label yi given the image xi and parameterized by W. To see this, remember that the Softmax classifier interprets the scores inside the output vector f as the unnormalized log probabilities. Exponentiating these quantities therefore gives the (unnormalized) probabilities, and the division performs the normalization so that the probabilities sum to one. In the probabilistic interpretation, we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing _Maximum Likelihood Estimation_ (MLE). A nice feature of this view is that we can now also interpret the regularization term R(W) in the full loss function as coming from a Gaussian prior over the weight matrix W, where instead of MLE we are performing the _Maximum a posteriori_ (MAP) estimation. We mention these interpretations to help your intuitions, but the full details of this derivation are beyond the scope of this class.  
可以解释为在给定图像 xi 和参数 W 的情况下，分配给正确标签 yi 的（归一化）概率。要理解这一点，请记住 Softmax 分类器将输出向量 f 中的分数解释为未归一化的对数概率。对这些量取指数因此给出（未归一化）概率，而除法执行归一化操作，使得概率之和为 1。在概率解释中，因此我们是在最小化正确类别的负对数似然，这可以解释为执行最大似然估计（MLE）。这种观点的一个优点是，我们现在还可以将完整损失函数中的正则化项 R(W) 解释为来自权重矩阵 W 上的高斯先验，其中我们不是执行 MLE，而是执行最大后验（MAP）估计。我们提及这些解释是为了帮助你的直觉，但这个推导的完整细节超出了本课程的范围。

**Practical issues: Numeric stability**. When you’re writing code for computing the Softmax function in practice, the intermediate terms efyi and ∑jefj may be very large due to the exponentials. Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. Notice that if we multiply the top and bottom of the fraction by a constant C and push it into the sum, we get the following (mathematically equivalent) expression:  
实际问题：数值稳定性。在实际编写计算 Softmax 函数的代码时，由于指数运算，中间项 efyi 和 ∑jefj 可能会非常大。除大数可能导致数值不稳定，因此使用归一化技巧非常重要。注意，如果我们将分数的上下乘以一个常数 C 并把它移入求和，我们会得到以下（数学上等价）的表达式：

efyi∑jefj=CefyiC∑jefj=efyi+logC∑jefj+logC

We are free to choose the value of C. This will not change any of the results, but we can use this value to improve the numerical stability of the computation. A common choice for C is to set logC=−maxjfj. This simply states that we should shift the values inside the vector f so that the highest value is zero. In code:  
我们可以自由选择 C 的值。这不会改变任何结果，但我们可以利用这个值来提高计算的数值稳定性。 C 的一个常见选择是设置 logC=−maxjfj 。这简单地表明我们应该将向量 f 中的值进行平移，使得最大值为零。在代码中：

```
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

```

**Possibly confusing naming conventions**. To be precise, the _SVM classifier_ uses the _hinge loss_, or also sometimes called the _max-margin loss_. The _Softmax classifier_ uses the _cross-entropy loss_. The Softmax classifier gets its name from the _softmax function_, which is used to squash the raw class scores into normalized positive values that sum to one, so that the cross-entropy loss can be applied. In particular, note that technically it doesn’t make sense to talk about the “softmax loss”, since softmax is just the squashing function, but it is a relatively commonly used shorthand.  
可能令人困惑的命名惯例。准确地说，SVM 分类器使用的是 hinge loss，有时也称为 max-margin loss。Softmax 分类器使用的是 cross-entropy loss。Softmax 分类器得名于 softmax 函数，该函数用于将原始类别分数压缩成归一化的正值，这些值之和为 1，以便可以应用交叉熵损失。特别要注意的是，从技术上讲，谈论“softmax loss”没有意义，因为 softmax 只是一个压缩函数，但它是一个相对常用的简称。

### SVM vs. Softmax  SVM 与 Softmax

A picture might help clarify the distinction between the Softmax and SVM classifiers:  
一张图可能有助于澄清 Softmax 和 SVM 分类器之间的区别：

![](https://cs231n.github.io/assets/svmvssoftmax.png)

Example of the difference between the SVM and Softmax classifiers for one datapoint. In both cases we compute the same score vector **f** (e.g. by matrix multiplication in this section). The difference is in the interpretation of the scores in **f**: The SVM interprets these as class scores and its loss function encourages the correct class (class 2, in blue) to have a score higher by a margin than the other class scores. The Softmax classifier instead interprets the scores as (unnormalized) log probabilities for each class and then encourages the (normalized) log probability of the correct class to be high (equivalently the negative of it to be low). The final loss for this example is 1.58 for the SVM and 1.04 (note this is 1.04 using the natural logarithm, not base 2 or base 10) for the Softmax classifier, but note that these numbers are not comparable; They are only meaningful in relation to loss computed within the same classifier and with the same data.  
SVM 和 Softmax 分类器在单个数据点上的差异示例。在两种情况下，我们都计算相同的得分向量 f（例如，在本节中通过矩阵乘法计算）。差异在于对 f 中得分的解释：SVM 将这些得分解释为类别得分，其损失函数鼓励正确类别（蓝色类别 2）的得分比其他类别得分高出一个间隔。Softmax 分类器则将得分解释为每个类别的（未归一化）对数概率，然后鼓励正确类别的（归一化）对数概率较高（等效地，其负值较低）。此示例中，SVM 的最终损失为 1.58，Softmax 分类器的最终损失为 1.04（注意，这是使用自然对数计算的，不是以 2 或 10 为底的对数），但请注意，这些数字不可比较；它们仅在相同分类器和相同数据计算出的损失中才有意义。

**Softmax classifier provides “probabilities” for each class.** Unlike the SVM which computes uncalibrated and not easy to interpret scores for all classes, the Softmax classifier allows us to compute “probabilities” for all labels. For example, given an image the SVM classifier might give you scores [12.5, 0.6, -23.0] for the classes “cat”, “dog” and “ship”. The softmax classifier can instead compute the probabilities of the three labels as [0.9, 0.09, 0.01], which allows you to interpret its confidence in each class. The reason we put the word “probabilities” in quotes, however, is that how peaky or diffuse these probabilities are depends directly on the regularization strength λ - which you are in charge of as input to the system. For example, suppose that the unnormalized log-probabilities for some three classes come out to be [1, -2, 0]. The softmax function would then compute:  
Softmax 分类器为每个类别提供“概率”。与 SVM 计算未校准且难以解释的所有类别的分数不同，Softmax 分类器允许我们为所有标签计算“概率”。例如，对于一张图像，SVM 分类器可能会为“猫”、“狗”和“船”这三个类别给出分数[12.5, 0.6, -23.0]。而 Softmax 分类器则可以计算这三个标签的概率为[0.9, 0.09, 0.01]，这使你能够理解它在每个类别中的置信度。然而，我们之所以将“概率”一词加引号，是因为这些概率的峰值或分散程度直接取决于正则化强度 λ ——这是你作为系统输入负责的。例如，假设某些三个类别的未归一化对数概率为[1, -2, 0]。那么 Softmax 函数将计算：

[1,−2,0]→[e1,e−2,e0]=[2.71,0.14,1]→[0.7,0.04,0.26]

Where the steps taken are to exponentiate and normalize to sum to one. Now, if the regularization strength λ was higher, the weights W would be penalized more and this would lead to smaller weights. For example, suppose that the weights became one half smaller ([0.5, -1, 0]). The softmax would now compute:  
所采取的步骤是进行指数化并归一化以使其和为 1。现在，如果正则化强度 λ 更高，权重 W 会受到更大的惩罚，这将导致权重变小。例如，假设权重减小了一半（[0.5, -1, 0]）。此时，softmax 函数会计算：

[0.5,−1,0]→[e0.5,e−1,e0]=[1.65,0.37,1]→[0.55,0.12,0.33]

where the probabilites are now more diffuse. Moreover, in the limit where the weights go towards tiny numbers due to very strong regularization strength λ, the output probabilities would be near uniform. Hence, the probabilities computed by the Softmax classifier are better thought of as confidences where, similar to the SVM, the ordering of the scores is interpretable, but the absolute numbers (or their differences) technically are not.  
其中概率现在更加分散。此外，在正则化强度 λ 非常强的情况下，权重趋于极小数的极限中，输出概率会趋于均匀分布。因此，Softmax 分类器计算出的概率最好被理解为置信度，类似于 SVM，分数的排序是可解释的，但其绝对数值（或其差异）在技术上是不确定的。

**In practice, SVM and Softmax are usually comparable.** The performance difference between the SVM and Softmax are usually very small, and different people will have different opinions on which classifier works better. Compared to the Softmax classifier, the SVM is a more _local_ objective, which could be thought of either as a bug or a feature. Consider an example that achieves the scores [10, -2, 3] and where the first class is correct. An SVM (e.g. with desired margin of Δ=1) will see that the correct class already has a score higher than the margin compared to the other classes and it will compute loss of zero. The SVM does not care about the details of the individual scores: if they were instead [10, -100, -100] or [10, 9, 9] the SVM would be indifferent since the margin of 1 is satisfied and hence the loss is zero. However, these scenarios are not equivalent to a Softmax classifier, which would accumulate a much higher loss for the scores [10, 9, 9] than for [10, -100, -100]. In other words, the Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better. However, the SVM is happy once the margins are satisfied and it does not micromanage the exact scores beyond this constraint. This can intuitively be thought of as a feature: For example, a car classifier which is likely spending most of its “effort” on the difficult problem of separating cars from trucks should not be influenced by the frog examples, which it already assigns very low scores to, and which likely cluster around a completely different side of the data cloud.  
在实践中，SVM 和 Softmax 通常表现相当。SVM 和 Softmax 之间的性能差异通常非常小，不同的人可能会有不同的看法，认为哪种分类器效果更好。与 Softmax 分类器相比，SVM 是一个更局部的目标，这可以被视为一个缺点或优点。考虑一个得分为 [10, -2, 3] 的例子，其中第一个类别是正确的。SVM（例如，期望的边界为 Δ=1 ）会看到正确类别的得分已经高于其他类别的得分，并且会计算损失为零。SVM 不关心各个得分的细节：如果得分是 [10, -100, -100] 或 [10, 9, 9]，SVM 都会保持不变，因为边界为 1，所以损失为零。然而，这些情况与 Softmax 分类器并不等价，Softmax 分类器对于得分 [10, 9, 9] 的损失会远高于得分 [10, -100, -100] 的损失。 换句话说，Softmax 分类器对其产生的分数永远不会完全满意：正确类别的概率总是可能更高，错误类别的概率总是可能更低，而损失函数也总是能进一步改善。然而，SVM 一旦满足边界条件就会感到满意，并且不会在这一点之外对精确分数进行微观管理。这可以直观地看作是一个特点：例如，一个汽车分类器很可能将大部分“精力”用于解决从汽车中区分卡车的难题，它不应该受到青蛙示例的影响，这些示例它已经分配了非常低的分数，并且它们很可能聚集在数据云完全不同的另一侧。

### Interactive web demo  交互式网络演示

[![](https://cs231n.github.io/assets/classifydemo.jpeg)](http://vision.stanford.edu/teaching/cs231n/linear-classify-demo)

We have written an interactive web demo to help your intuitions with linear classifiers. The demo visualizes the loss functions discussed in this section using a toy 3-way classification on 2D data. The demo also jumps ahead a bit and performs the optimization, which we will discuss in full detail in the next section.  
我们编写了一个交互式网络演示，以帮助您理解线性分类器的直观概念。该演示使用二维数据上的三分类任务，可视化了本节讨论的损失函数。该演示还提前展示了优化过程，我们将在下一节详细讨论优化方法。

### Summary  摘要

In summary,  总之，

- We defined a **score function** from image pixels to class scores (in this section, a linear function that depends on weights **W** and biases **b**).  
    我们定义了一个从图像像素到类别分数的评分函数（在本节中，这是一个依赖于权重 W 和偏差 b 的线性函数）。
- Unlike kNN classifier, the advantage of this **parametric approach** is that once we learn the parameters we can discard the training data. Additionally, the prediction for a new test image is fast since it requires a single matrix multiplication with **W**, not an exhaustive comparison to every single training example.  
    与 kNN 分类器不同，这种参数化方法的优点在于一旦我们学习到参数就可以丢弃训练数据。此外，对新测试图像的预测非常快，因为它只需要与 W 进行一次矩阵乘法，而不是与每个训练样本进行彻底的比较。
- We introduced the **bias trick**, which allows us to fold the bias vector into the weight matrix for convenience of only having to keep track of one parameter matrix.  
    我们介绍了偏差技巧，它允许我们将偏差向量合并到权重矩阵中，以便只需跟踪一个参数矩阵。
- We defined a **loss function** (we introduced two commonly used losses for linear classifiers: the **SVM** and the **Softmax**) that measures how compatible a given set of parameters is with respect to the ground truth labels in the training dataset. We also saw that the loss function was defined in such way that making good predictions on the training data is equivalent to having a small loss.  
    我们定义了一个损失函数（我们介绍了两种常用的线性分类器损失：支持向量机（SVM）和 Softmax），该函数用于衡量给定参数集与训练数据集中真实标签的兼容程度。我们还了解到，损失函数是这样定义的：在训练数据上做出良好预测等同于损失值较小。

We now saw one way to take a dataset of images and map each one to class scores based on a set of parameters, and we saw two examples of loss functions that we can use to measure the quality of the predictions. But how do we efficiently determine the parameters that give the best (lowest) loss? This process is _optimization_, and it is the topic of the next section.  
我们现在看到了一种将图像数据集映射到基于参数集的类别分数的方法，并且看到了两种可以用来衡量预测质量的损失函数示例。但是，我们如何高效地确定能给出最佳（最低）损失的参数？这个过程是优化，也是下一节的主题。

### Further Reading  进一步阅读

These readings are optional and contain pointers of interest.  
这些阅读材料是可选的，包含了一些有趣的相关链接。

- [Deep Learning using Linear Support Vector Machines](https://arxiv.org/abs/1306.0239) from Charlie Tang 2013 presents some results claiming that the L2SVM outperforms Softmax.  
    使用线性支持向量机进行深度学习，来自 Charlie Tang 的 2013 年研究展示了一些结果，声称 L2SVM 比 Softmax 表现更优。

-  [cs231n](https://github.com/cs231n)
-  [cs231n](https://twitter.com/cs231n)