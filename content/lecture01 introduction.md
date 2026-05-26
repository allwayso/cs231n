网课链接：[Stanford CS231N Deep Learning for Computer Vision | Spring 2025 | Lecture 1: Introduction - YouTube](https://www.youtube.com/watch?v=2fq9wYslV0A&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)


---

### **第一部分：李飞飞教授 (Prof. Li Fei-Fei) —— 愿景与历史**

#### **1. 课程开场与 AI 中的视觉定位 (00:00 - 03:00)**

- **核心内容**：介绍教学团队；阐述计算机视觉是智能的“基石”；定义计算机视觉、机器学习与深度学习的交汇关系,。
- **关键概念**：跨学科性 (Interdisciplinary)、核心交汇点 (Core intersection)。
![[Pasted image 20260413105058.png|500]]
#### **2. 视觉的生物进化背景 (估计：03.00 - 6:30)**

- **核心内容**：5.4 亿年前的“寒武纪大爆发”；光敏细胞（原始针孔眼）如何驱动智能进化；人类视觉系统的复杂性。
- **关键概念**：寒武纪大爆发 (Cambrian Explosion)、主动感知。

>这里直接从Cambrain跳跃到人类社会，主要谈及人类使用机器来“产生视觉”的方式，比如相机。不过"eyes are not enough for seeing",生成影像距离产生visual intelligence还有很遥远的距离，所以引入了第三章，深度学习和计算机视觉的结合。
#### **3. 计算机视觉的学术起源 (1950s - 1970s) (6:30 - 19:30)**

- **核心内容**：Hubel & Wiesel 的猫实验（发现层级化处理）；Larry Roberts 的博士论文；MIT 的夏季视觉项目；David Marr 的 3D 表示理论；Generalized cylinders & pictorial structure。
- **关键概念**：感受野 (Receptive Field)、初始简图 (Primal Sketch)。

>Hubel & Wiesel 的猫实验：因其对神经系统视觉处理的研究而获得诺贝尔奖 1. 每个神经元都有独立的[[感受野]]（actually并不是这个实验中发现的） 2. 视觉通路是分层的：深层的神经元具有更复杂的感受野 
>受限于篇幅，这里prof Li的讲解并不完全准确，可以参考[[Hubel & Wiesel 的猫实验]]补充说明

>Larry Roberts 的博士论文:CV领域的第一篇论文,研究了如何通过计算机对图片进行三维模型识别，以及如何将三维模型投影到二维平面上[[《Machine Perception of Three-Dimensional Solids》.pdf]] 中文概括+个人理解见[[中文浓缩版MPTHS]]

>David Marr：现代CV开创者，其大致工作见[[David Marr：现代cv体系教父]]，将神经和认知系统的研究与计算机识别相结合，但是仍然延续了Larry Roborts的3D重建思路

>[[广义圆柱体Generalized cylinders]] & [[PS模型pictorial structure]]：对现实对象的两种建模方式，前者是David Marr对构建普适性3D模型的尝试，而后者舍弃3D模型，在2D平面中进行判别和定位。这标志着cv从几何重建转向了模式识别，从严格几何计算转向了概率匹配，为后续全自动深度学习打下基础。

#### **4. AI 寒冬下的探索与认知科学 (1980s - 2000s) (估计：20:00 - 30:00)**

- **核心内容**：认知科学的爆发；认知科学对视觉速度的测量（150 毫秒）；早期的人脸检测成功案例，互联网和早期数据集的出现。
- **关键概念**：AI 寒冬 (AI Winter)、物体识别 (Object Recognition)。

>这一时期cv相关的技术发展相对放缓，但是认知科学、深度学习以及互联网等技术正在悄然萌芽
#### **5. 深度学习革命与 ImageNet (估计：30:00 - 45:00)**

- **核心内容**：Fukushima 的 Neocognitron；1986 年的back propagation突破；数据的关键性；ImageNet 挑战赛与 2012 年 AlexNet 的历史性转折。
- **关键概念**：反向传播 (Back propagation)、AlexNet、模型过拟合。

>[[Neocognitron：CNN的直接祖先]]，人工设计的精巧设计

>[[back propagation：深度网络自行学习]]

>ImageNet：大规模数据视觉识别挑战，第一个开源大规模图片数据集

>[[AlexNet：数据驱动深度学习时代的开端]]
#### **6. 现代应用、硬件与社会影响 (估计：45:00 - 55:00)**

- **核心内容**：视觉任务的爆炸式增长（分割、医疗影像、生成式 AI）；Nvidia GPU 的硬件驱动；AI 系统中的人类偏见与道德伦理,,,。
- **关键概念**：生成式 AI、计算力 (Compute)、社会偏见。

---

### **第二部分：Isan Adeli 教授 —— 课程架构与技术路线**

#### **1. 深度学习基础 (Deep Learning Basics)**

- **图像分类是核心**：这是计算机视觉最基础的任务，即给图像打上标签（如“猫”）。
- **从线性到非线性**：课程将从**线性分类器**（通过超平面切分数据）讲起，探讨其局限性，并引入**神经网络**。通过堆叠多个操作层，模型可以处理复杂的非线性函数。
- **模型优化**：为了让模型具备良好的**泛化能力**（在未见过的数据上表现良好），将深入讲解**正则化 (Regularization)** 和**优化 (Optimization)** 算法，以解决过拟合或欠拟合问题。

#### **2. 感知与理解视觉世界 (Perceiving and Understanding)**

- **任务的进阶**：除了简单的分类，课程还会涵盖更精细的任务：
    - **语义分割 (Semantic Segmentation)**：为图像中的每个像素打标签。
    - **目标检测 (Object Detection)**：用边界框定位物体。
    - **实例分割 (Instance Segmentation)**：最精细的任务，结合了检测与分割，为每个物体实例生成掩码。
- **维度扩展**：包括**视频分类**、**多模态视频理解**（结合视觉与声音）以及通过**注意力机制 (Attention)** 来可视化模型到底在“看”什么。
- **架构演进**：将学习 **CNN (卷积神经网络)**、**RNN (循环神经网络)** 以及目前主流的 **Transformers** 框架。

>将基本的图像分类任务拓展到语义分割、视频理解等更复杂的任务，从基本的MLP架构拓展到CNN、Transformers等现代框架
#### **3. 大规模分布式训练 (Large-scale Distributed Training)**

- **本学期新增内容**：针对目前大语言模型 (LLM)和大视觉模型 (LVM)的趋势，新增了关于如何训练大规模模型的讲座。
- **技术策略**：涵盖**数据并行化 (Data Parallelization)**、**模型并行化 (Model Parallelization)** 以及训练过程中的同步与效率挑战。

#### **4. 生成式与交互式视觉智能 (Generative and Interactive)**

- **自监督学习 (Self-supervised Learning)**：利用无标签的海量数据让模型自我学习，这是近年来视觉突破的关键。
- **生成模型**：涵盖**风格迁移**、**扩散模型 (Diffusion Models)**（如 DALL-E 的原理），你甚至会在作业中尝试用文本生成表情符号。
- **3D 视觉与具身智能 (Embodied AI)**：探索从 2D 图像重建 3D 空间，以及如何让 AI 代理在物理世界中感知、规划并执行任务（如机器人控制）。

#### **5. 以人为本的 AI (Human-centered AI)**

- **社会影响**：讨论 AI 带来的**人类偏见**、社会影响以及在**医疗护理**（如服务老年群体）中的积极应用。
- **行业认可**：提到深度学习先驱（如 Hinton, LeCun 等）获得的图灵奖和诺贝尔奖，强调该领域的深远意义。