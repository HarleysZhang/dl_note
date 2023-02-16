## 卷积神经网络基本部件

学习完深度神经网络的基本架构后，就可以学习卷积神经网络中的重要组成部件（或模块）。

一个简单的卷积神经网络（`ConvNet`）是由各种网络层按照顺序排列组成的，网络中的每个层都使用一个可微分的函数将激活数据从一个层传递到另一个层。`ConvNet` 架构主要由三种类型的层构成：卷积层，池化（`Pooling`）层和全连接层（全连接层和常规神经网络中的一样）。堆叠这些层就可以构建一个完整的 `ConvNet` 架构。

### 2.1，端到端思想

深度学习提供了一种端到端的学习方式（`paradigm` 范式），整个学习流程并不进行人为的子问题划分， 而是完全交给深度学习模型**直接学习从原始输入到期望输出的映射**。相比分治策略，“端到端”的学习方式具有协同增效的优势，有更大可能获得全局最优解。

深度学习的模型的训练过程可以简单抽象为从模型输出向最终目标的直接"拟合"，而中间的部件则起到了将原始输入数据映射为特征（即特征学习）随后再映射为样本标记（即目标任务，如分类、检测）的作用。

### 2.6，全连接层

全连接层（`fully connected layers`）在整个卷积神经网络中起到的是 “分类器” 的作用（图像分类任务）。如果说卷积层、池化层和激活函数层等操作是将原始数据映射到隐层特征空间的话，那么全连接层则起到是将学到的**特征表示**映射到样本的标记空间的作用。

## 参考资料

1. [Visualizing the Feature Maps and Filters by Convolutional Neural Networks](https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e)
2. [What is a Convolutional Neural Network?](https://poloclub.github.io/cnn-explainer/)
3. [Visual Interpretability for Convolutional Neural Networks](https://towardsdatascience.com/visual-interpretability-for-convolutional-neural-networks-2453856210ce)
4. [Convolutional Neural Network: Feature Map and Filter Visualization](https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c)