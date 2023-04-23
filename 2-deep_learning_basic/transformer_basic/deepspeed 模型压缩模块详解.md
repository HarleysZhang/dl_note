

## ZeroQuant-零成本高效训练后量化

`DS-Compression` 模块支持零成本的 `INT8` 量化和 `INT4/INT8` 混合精度量化，算法原理可以参考作者的[论文](https://www.deepspeed.ai/tutorials/model-compression/#2-tutorial-for-zeroquant-efficient-and-affordable-post-training-quantization)。

### 什么是 ZeroQuant

`ZeroQuant` 是一种高效的训练后量化方法，它包括:

1. 针对权重和激活的细粒度硬件友好量化方案，可以显着降低量化误差；
2. 一种新的负担得起的逐层知识蒸馏算法 (`LKD`)，且无需原始训练集；
3. 一个高度优化的量化系统后端支持，以消除量化/反量化开销。

通过上述这些技术，ZeroQuant 能够将模型量化为 INT8 而无需任何成本（1），以及将模型量化为 INT4/INT8 混合精度量化（2），而资源需求最少（例如，基于 BERT 的量化需要 31s）。

### 什么时候使用 ZeroQuant

1. 当进行训练中量化（quantization aware training，QAT）非常消耗 GPU 资源时；
2. 原始训练数据集无法使用。

当存在以上两种情况，可以考虑将 transformer 架构的模型量化成 `INT8` 或 `INT4/INT8` 格式。

### 如何使用 ZeroQuant



## 参考资料

- [DeepSpeed Compression: A composable library for extreme compression and zero-cost quantization](https://www.microsoft.com/en-us/research/blog/deepspeed-compression-a-composable-library-for-extreme-compression-and-zero-cost-quantization/)
- [Extreme Compression for Pre-trained Transformers Made Simple and Efficient](https://arxiv.org/pdf/2206.01859.pdf)
- [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://www.microsoft.com/en-us/research/publication/zeroquant-efficient-and-affordable-post-training-quantization-for-large-scale-transformers/)