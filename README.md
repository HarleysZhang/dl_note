- [项目概述](#项目概述)
- [一，数学基础专栏](#一数学基础专栏)
- [二，神经网络基础部件](#二神经网络基础部件)
- [三，深度学习炼丹](#三深度学习炼丹)
- [四，深度学习模型压缩](#四深度学习模型压缩)
- [五，模型推理部署](#五模型推理部署)
- [六，异构计算](#六异构计算)
- [七，进阶课程](#七进阶课程)
- [参考资料](#参考资料)

## 项目概述

本仓库项目是个人总结的深度学习炼丹、优化及部署落地笔记，包含深度学习数学基础知识、深度学习基础知识、神经网络基础部件详解、构建 `CNN` 网络总结、深度学习炼丹策略、深度学习模型压缩算法、以及深度学习推理框架代码解析及动手实战。

## 一，数学基础专栏

- [深度学习数学基础-概率与信息论](./1-math_ml_basic/深度学习数学基础-概率与信息论.md)
- [深度学习基础-机器学习基本原理](./1-math_ml_basic/深度学习基础-机器学习基本原理.md)
- [随机梯度下降法的数学基础](./1-math_ml_basic/随机梯度下降法的数学基础.md)
## 二，神经网络基础部件

1，**神经网络基础部件**：

1. [神经网络基础部件-卷积层详解](./2-deep_learning_basic/神经网络基础部件-卷积层详解.md)
2. [神经网络基础部件-BN 层详解](./2-deep_learning_basic/神经网络基础部件-BN层详解.md)
3. [神经网络基础部件-激活函数详解](./2-deep_learning_basic/神经网络基础部件-激活函数详解.md)
4. [神经网络基础部件-Transformer 详解](./2-deep_learning_basic/神经网络基础部件-Transformer详解.md)

2，**深度学习基础**：
- [反向传播与梯度下降详解](2-deep_learning_basic/反向传播与梯度下降详解.md)
- [深度学习基础-参数初始化详解](./2-deep_learning_basic/深度学习基础-参数初始化详解.md)
- [深度学习基础-损失函数详解](./2-deep_learning_basic/深度学习基础-损失函数详解.md)
- [深度学习基础-优化算法详解](./2-deep_learning_basic/深度学习基础-优化算法详解.md)

## 三，深度学习炼丹

1. [深度学习炼丹-数据标准化](./3-deep_learning_alchemy/深度学习炼丹-数据标准化.md)
2. [深度学习炼丹-数据增强](./3-deep_learning_alchemy/深度学习炼丹-数据增强.md)
3. [深度学习炼丹-不平衡样本的处理](./3-deep_learning_alchemy/深度学习炼丹-不平衡样本的处理.md)
4. [深度学习炼丹-超参数设定](./3-deep_learning_alchemy/深度学习炼丹-超参数调整.md)
5. [深度学习炼丹-正则化策略](./3-deep_learning_alchemy/深度学习炼丹-正则化策略.md)

## 四，深度学习模型压缩

1. [深度学习模型压缩算法综述](./4-model_compression/深度学习模型压缩方法概述.md)
2. [模型压缩-轻量化网络设计与部署总结](./4-model_compression/模型压缩-轻量化网络详解.md)
3. [模型压缩-剪枝算法详解](./4-model_compression/模型压缩-剪枝算法详解.md)
4. [模型压缩-知识蒸馏详解](./4-model_compression/模型压缩-知识蒸馏详解.md)
5. [模型压缩-量化算法详解](./4-model_compression/模型压缩-网络量化概述.md)

## 五，模型推理部署

1，模型部署：

- [卷积神经网络复杂度分析](./5-model_deploy/卷积神经网络复杂度分析.md)
- [模型压缩部署概述](./5-model_deploy/模型压缩部署概述.md)
- [FasterTransformer 速览](./5-model_deploy/FasterTransformer速览.md)

2，模型推理：

- [矩阵乘法详解](./5-model_deploy/卷积算法优化.md)
- [模型推理加速技巧-融合卷积和BN层](./5-model_deploy/模型推理加速技巧-融合卷积和BN层.md)

3，`ncnn` 框架源码解析：

- [ncnn 源码解析-sample 运行](5-model_deploy/ncnn源码解析/ncnn源码解析-sample运行.md)
- [ncnn 源码解析-Net 类](5-model_deploy/ncnn源码解析/ncnn源码解析-Net类.md)
- [ncnn 源码解析-Layer 层](5-model_deploy/ncnn源码解析/ncnn源码解析-Layer层.md)
- [ncnn 源码解析-常见算子](../5-model_deploy/ncnn源码解析/ncnn源码解析-常见算子.md)

## 六，异构计算

1. 移动端异构计算：`neon` 编程
2. GPU 端异构计算：`cuda` 编程

通用矩阵乘法 `gemm` 算法解析与优化、`neon`、`cuda` 编程等内容，以及 `ncnn` 框架代码解析总结。

## 七，进阶课程

1，推荐几个比较好的深度学习模型压缩与加速的仓库和课程资料：

1. [awesome-emdl](https://github.com/EMDL/awesome-emdl): 嵌入式与移动端深度学习研究资料合集。
2. [AI-System](https://github.com/microsoft/AI-System/tree/main/Textbook): 深度学习系统。
3. [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)

2，一些笔记好的博客链接：

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): 国内比较好的博客大都参考这篇文章。
- [C++ 并发编程（从C++11到C++17）](https://paul.pub/cpp-concurrency/): 不错的 C++ 并发编程教程。 
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [annotated_deep_learning_paper_implementations
](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
## 参考资料

- 《深度学习》
- 《机器学习》
- 《动手学深度学习》
- [《AI-EDU》](https://microsoft.github.io/ai-edu/index.html)
- [《AI-System》](https://github.com/microsoft/AI-System/tree/main/Textbook)
- [《PyTorch_tutorial_0.0.5_余霆嵩》](https://github.com/TingsongYu/PyTorch_Tutorial)
- [《动手编写深度学习推理框架 Planer》](https://github.com/Image-Py/planer)

**chatglm-6b 模型**

- 单机单卡
- 测试设备 T4
- 测试 promot: 
    - 你是现代诗人，用'红包、美好、表白、夕阳、月光、慢慢'关键词生成2首表白唯美打油诗
    - 写一篇500字的武侠小说，主角名字为李纯白

测试框架：HuggingFace + Transformers + DeepSpeed 

| Batch_size | 数据类型 | 显存占用 | GPU使用率 | 性能（tokens per second） |
| ---------- | -------- | -------- | --------- | ---------------------|
| 1          | `FP16`   | 13046MiB | 83%       | 14.0~14.7            |

测试框架：HuggingFace + **Transformers**

使用 chatglm 自带的量化函数进行量化，虽然对显存的要求低了，但是性能 `tps` 下降了很多，原因还在分析。

| Batch_size | 数据类型 | 显存占用 | GPU使用率 | 性能（tokens per second） |
| ---------- | -------- | -------- | --------- | -------------------- |
| 1          | `FP16`   | 13027MiB | 82%       | 13.23~14.25          |
| 1          | `INT8`   | 7008MiB  | 89%       | 14.02                |
