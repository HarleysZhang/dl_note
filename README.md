- [项目概述](#项目概述)
- [一，数学基础专栏](#一数学基础专栏)
- [二，神经网络基础部件](#二神经网络基础部件)
- [三，深度学习炼丹](#三深度学习炼丹)
- [四，深度学习模型压缩](#四深度学习模型压缩)
- [五，模型推理部署](#五模型推理部署)
- [八，进阶课程](#八进阶课程)
- [九，其他](#九其他)
- [Star History](#star-history)
- [参考资料](#参考资料)

## 项目概述

本仓库项目是个人总结的计算机视觉和大语言模型学习笔记，包含深度学习基础知识、神经网络基础部件详解、深度学习炼丹策略、深度学习模型压缩算法、深度学习推理框架代码解析及动手实战，以及 `LLM` 基础及推理优化的专栏笔记。

## 一，数学基础专栏

- [深度学习数学基础-概率与信息论](./1-math_ml_basic/深度学习数学基础-概率与信息论.md)
- [深度学习基础-机器学习基本原理](./1-math_ml_basic/深度学习基础-机器学习基本原理.md)
- [随机梯度下降法的数学基础](./1-math_ml_basic/随机梯度下降法的数学基础.md)
- [Python 编程思维导航](./1-math_ml_basic/python_learn_xmind)

## 二，神经网络基础部件

1，**神经网络基础部件**：

1. [神经网络基础部件-卷积层详解](./2-deep_learning_basic/神经网络基础部件-卷积层详解.md)
2. [神经网络基础部件-BN 层详解](./2-deep_learning_basic/神经网络基础部件-BN层详解.md)
3. [神经网络基础部件-激活函数详解](./2-deep_learning_basic/神经网络基础部件-激活函数详解.md)

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
5. [模型压缩-量化算法详解](./4-model_compression/模型压缩-量化算法概述.md)

## 五，模型推理部署

1，模型部署：

- [卷积神经网络复杂度分析](./5-model_deploy/卷积神经网络复杂度分析.md)
- [模型压缩部署概述](./5-model_deploy/模型压缩部署概述.md)

2，模型推理：

- [矩阵乘法详解](./5-model_deploy/卷积算法优化.md)
- [模型推理加速技巧-融合卷积和BN层](./5-model_deploy/模型推理加速技巧-融合卷积和BN层.md)

3，`ncnn` 框架源码解析：

- [ncnn 源码解析-sample 运行](5-model_deploy/ncnn源码解析/ncnn源码解析-sample运行.md)
- [ncnn 源码解析-Net 类](5-model_deploy/ncnn源码解析/ncnn源码解析-Net类.md)
- [ncnn 源码解析-Layer 层](5-model_deploy/ncnn源码解析/ncnn源码解析-Layer层.md)
- [ncnn 源码解析-常见算子](../5-model_deploy/ncnn源码解析/ncnn源码解析-常见算子.md)

**5，AI/NPU/GPU 芯片特性**：

- [英伟达 GPU 架构特性详解](5-model_deploy/英伟达GPU架构详解.md)

6，异构计算

1. 移动端异构计算：`neon` 编程
2. GPU 端异构计算：`cuda` 编程

通用矩阵乘法 `gemm` 算法解析与优化、`neon`、`cuda` 编程等内容，以及 `ncnn` 框架代码解析总结。



## 八，进阶课程

1，推荐几个比较好的深度学习模型压缩与加速的仓库和课程资料：

1. [神经网络基本原理教程](https://github.com/microsoft/ai-edu/blob/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.1-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%89%8D%E5%90%91%E8%AE%A1%E7%AE%97%E5%8E%9F%E7%90%86.md)
2. [AI-System](https://microsoft.github.io/AI-System/): 深度学习系统，主要从底层方向讲解深度学习系统等原理、加速方法、矩阵成乘加计算等。
3. [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)：很好的 pytorch 深度学习教程。

2，一些笔记好的博客链接：

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): 国内比较好的博客大都参考这篇文章。
- [C++ 并发编程（从C++11到C++17）](https://paul.pub/cpp-concurrency/): 不错的 C++ 并发编程教程。
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

## 九，其他

最后，持续高质量创作不易，有 `5` 秒空闲时间的，**可以扫码关注我的公众号-嵌入式视觉**，记录 CV 算法工程师成长之路，分享技术总结、读书笔记和个人感悟。
> 公众号不会写标题党文章，也不输出给大家带来的焦虑的内容！

![qcode](images/others/qcode.png)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HarleysZhang/deep_learning_system&type=Date)](https://star-history.com/#HarleysZhang/deep_learning_system&Date)

## 参考资料

- 《深度学习》
- 《机器学习》
- 《动手学深度学习》
- [《机器学习系统：设计和实现》](https://openmlsys.github.io/index.html)
- [《AI-EDU》](https://ai-edu.openai.wiki/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/index.html)
- [《AI-System》](https://github.com/microsoft/AI-System/tree/main/Textbook)
- [《PyTorch_tutorial_0.0.5_余霆嵩》](https://github.com/TingsongYu/PyTorch_Tutorial)
- [《动手编写深度学习推理框架 Planer》](https://github.com/Image-Py/planer)
- [distill：知识精要和在线可视化](https://distill.pub/)
- [LLVM IR入门指南](https://github.com/Evian-Zhang/llvm-ir-tutorial)
- [nanoPyC](https://github.com/vesuppi/nanoPyC/tree/master)
