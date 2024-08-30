# LightLLM 推理框架静态性能测试及分析

算子占比分析、通信时间对比分析、不同阶段的算子分析。

1，之前的问题是因为 Prometheus 工具是 Python 工具没办法统计异步时间，而 cuda 流是异步的，所以统计的 kernel 时间不对。启用 CUDA_LAUNCH_BLOCKING 变量后，可以得到更准确的算子维度的性能数据。
2，A40 机器的简单拓扑互联结构图，卡间互联通信速度对比，注意 A40 机器的 NVLINK 是阉割版的！理论值为 112.5 GB/s。

![A40_Topology](../../images/lightllm_analysis/A40_Topology.png)

3，备注：集群不稳定的两束证据，尤其是 decode 阶段的 all_duce 通信时间，这个问题很重要！！！如果集群不解决，我们集群的性能数据都得重复测好几次，且结果不稳定。

![cluster_problem](../../images/lightllm_analysis/cluster_problem.png)

## 一、Prefill/Decode 阶段算子执行时间占比情况

1，将 LLaMA2-70B 模型分别在 4/8 卡 A100-40GB 上部署，设置 batch_size=20, input_length=1024, output_length=1024，统计 prefill/decode 阶段各算子的执行时间占比：

![pie_chart](../../images/lightllm_analysis/pie_chart.png)

2，将 LLaMA2-70B 模型分别在 4/8 卡 A40-48GB 上部署，设置 batch_size=20, input_length=1024, output_length=1024，统计 prefill/decode 阶段各算子的执行时间占比：

![pie_chart2](../../images/lightllm_analysis/pie_chart2.png)

3，将 LLaMA2-7B 模型分别在 4/8 卡 T4-16GB 上部署，设置 batch_size=32, input_length=1024, output_length=1024，统计 prefill/decode 阶段各算子的执行时间占比：

![pie_chart3](../../images/lightllm_analysis/pie_chart3.png)

**算子开销实验结论：对于 Prefill 阶段**：

- 如果在使用 NVLink 互联的卡（A100）上部署模型，则无论 4 卡还是 8 卡，都是 ffn 算子的执行时间占比最高；
- 如果在使用 PCIe 互联的卡（T4）上部署模型，则无论 4 卡还是 8 卡，都是 all_reduce 算子的执行时间占比最高；
- 如果在两两一组 NVLink 互联、其余均为 PCIe 互联的卡（A40）上部署模型，4 卡时是 ffn 算子的执行时间占比最高，而使用 8 卡时转变为 all_reduce  算子的执行时间占比最高。

理论 FLOPs：对于 Prefill 阶段的每个 decoder 层的算子计算量：

- MHA (attention) 层，占计算量大头的是线性变换计算 Q、K、V 和输出投影矩阵 O: FLOPs = 3 \times 2sh^2 + 2sh^2 = 8 sh^2。
- Feed-forward（MLP/FFN）层的计算量分析：FLOPs = 16 sh^2。

## 二、4 卡 A100 和 4 卡 A40 对比

**实验设计**：LLaMA2-70b 模型，输入/输出 tokens 长度都为 1024，batch_size 设为 20，比较 4 卡 A100 和 4 卡 A40 的性能差异。

### 2.2、推理性能分析结果

![4a100_4a40](../../images/lightllm_analysis/4a100_4a40.png)

结果分析：在推理层，无论是 prefill 阶段还是 decode 阶段，4 卡 A100 的推理性能都优于 4 卡 A40 的性能。这是符合预期的，因为 A100 在算力、显存带宽、卡间通信方面都优于 A40。

### 2.3、算子性能分析结果

`A100` 和 `A40` 卡的各算子的总耗时情况对比如下图所示，图中也展示了 A100 相对于 A40 的[加速比](https://www.jendow.com.tw/wiki/%E5%8A%A0%E9%80%9F%E6%AF%94)。

![算子性能分析结果](../../images/lightllm_analysis/op_perf1.png)

结果分析：

1. prefill 阶段 A100 的 all_reduce【卡间互联通信时间】时间比 a40 快了 300%，理论值的对比是 600/112.5 = 5.33，基本符合理论值对比；decode 阶段  all_reduce 时间接近。
2. prefill/decode 阶段 a100 ffn【纯矩阵运算】和 a40 快了 226%/172%，理论上 a100 算力比 a40 快 312/150 = 2.08，实验结果和理论预估几乎一致。

> prefill 阶段 sequence 和 batch_size 都大于 1，算子非访存密集型算子，有利于发挥 Tensor 性能。

## 三、4 卡 A40 和 8 卡 A40 对比

实验设计：llama2-70b 模型，输入/输出 tokens 长度都为 1024，batch_size 设为 20，比较 4 卡 A40 和 8 卡 A40 的性能差异。

### 3.2、推理性能分析结果

![4a40_8a40](../../images/lightllm_analysis/4a40_8a40.png)

### 3.3、算子性能分析结果

![算子性能分析结果](../../images/lightllm_analysis/op_perf2.png)

**结果分析**:

1. 上述表格可以看出 4 卡 A40 的 all_reduce 时间比 8 卡 A40 少一半，且 4 卡和 8卡 A40 的 prefill 阶段的 ffn + all_reduce 时间几乎一致，这也直接解释了 8 卡 A40 的最大并发跟 4 卡 A40 的最大并发差不多的原因。

## 四、4 卡 A100 和 8 卡 A100 对比

**实验设计**：LLaMA2-70b 模型，输入/输出 tokens 长度均为 1024，batch_size 设为 20，比较 4 卡 A100 和 8 卡 A100 的性能差异。

### 4.1、推理性能分析结果

![4a100_8a100](../../images/lightllm_analysis/4a100_8a100.png)

### 4.2、算子性能分析结果

![算子性能分析结果](../../images/lightllm_analysis/op_perf3.png)

**结果分析**：

- 在 A100 上增加 TP 数在 prefill 阶段对于大多计算密集型算子上都有性能提升：在 prefill 阶段，对于纯计算算子（线性层等），使用 8 张 A100 相对于 4 张 A100 都能带来接近 2x 的性能提升，这符合预期，因为 8 卡 A100 的算力大约是 4 卡 A100 的两倍；而 decode  阶段，因为sequence = 1，所以算术（计算）强度比较低，性能受内存带宽限制而不是算力限制，GPU 无法被充分利用，这也是 decode 阶段通过增加算力给纯计算算子带来的性能提升并不明显的原因。
- 在 A100 上增加 TP 数不会带来大量的额外通信开销：8 卡 A100 和 4 卡 A100 相比，all_reduce 的通信总耗时几乎持平。在总通信量相同的情况下，8 卡的通信次数增加，而单次通信量减少，这并没有带来较大的性能损耗。所以，对于使用 NVLink 互联（NV Switch）的 A100 来说，卡间通信并不构成并发的性能瓶颈。

## 五、2 卡 A40 上 PCIe 和 NVLink 互联对比

**实验设计**：LLaMA-7B 模型，输入/输出 tokens 长度都为 1024，batch_size 设为 64，比较 A40 使用 NVLink 通信和使用 PCIe 通信的 2 卡的性能差异。

**实验过程**：在 8 卡 A40 机器上，GPU0 和 GPU1 是两两一组的 NVLink 通信，但是 GPU0 和 GPU3 之间没有 NVLink 通信，而是使用 PCIE4.0 通信。因此，分别在 A40 的 0,1 号卡和 0,3 号卡上进行测试。

### 5.1、推理性能分析结果

![2a40_pcie_nvlink](../../images/lightllm_analysis/2a40_pcie_nvlink.png)

### 5.2、算子性能分析结果

![算子性能分析结果](../../images/lightllm_analysis/op_perf4.png)

**结果分析**：

- Prefill 阶段通信量很大， 对于 all_reduce 算子来说，相比于 PCIe 互联卡，NVLink 互联卡的实际加速比约为 2.07x，而卡间互联通信的理论加速比是 1.76x；之所以实际加速比超过理论值，是因为，A40 机器的 GPU0 和 GPU3 是 PXB 连接方式【跨过多个 PCIE bridges】，其通信速度低于 PIX  模式，即低于理论的 PCIE4.0  的通信带宽。实验结果和理论预估几乎一致。
- Decode 阶段通信量很小，PCIe 互联卡的 all_reduce 时间是 NVLink 互联卡的时间的 1.2 倍，NVLINK 通信带来的性能提升不明显。

## 六、总结-lightllm 推理框架性能瓶颈分析实验结论

- **nvlink 通信在 prefill 阶段有绝对优势！但是在 decode 阶段优势不明显，也意味着目前的推理框架在 decode 阶段的卡间互联通信上有优化空间**。
- 带 PCIE 互联的卡（如 T4 卡和 8张 A40 卡），**prefill 阶段的 all_reduce 时间占比是最大的**。
- `decode` 阶段，因为 sequence = 1 ，所以算子的算术（计算）强度比较低，性能受内存带宽限制而不是算力限制（GPU 算力无法被充分利用），增加 TP 数给算子带来的性能提升并不明显。

> 奇怪点：8卡/4卡 A100，单个卡的通信和访存变少了，但是为什么 decode 阶段没有性能提升？

## 参考资料

- [a40](https://www.nvidia.com/en-us/data-center/a40/)
- [a100](https://www.nvidia.com/en-us/data-center/a100/)