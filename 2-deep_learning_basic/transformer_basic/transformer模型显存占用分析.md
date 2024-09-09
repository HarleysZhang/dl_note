- [一 KV cache 优化](#一-kv-cache-优化)
  - [1.1 KV cache 优化原理](#11-kv-cache-优化原理)
  - [1.2 LLM 推理的两个阶段](#12-llm-推理的两个阶段)
- [二 显存占用量理论分析](#二-显存占用量理论分析)
  - [2.1 推理过程中显存占用分析](#21-推理过程中显存占用分析)
  - [2.2 训练过程中显存占用分析](#22-训练过程中显存占用分析)
  - [2.3 推理过程中显存占用量理论计算](#23-推理过程中显存占用量理论计算)
  - [2.4 显存占用的定性分析和定量分析](#24-显存占用的定性分析和定量分析)
- [参考资料](#参考资料)

## 一 KV cache 优化

因果建模模型、自回归模型、生成式 generative 模型所代表的意义几乎一致。

### 1.1 KV cache 优化原理

**背景**：生成式模型的推理过程很有特点，**推理生成 `tokens` 的过程是迭代式的**。简单来说就是，用户给一个输入文本，模型会输出一个回答（长度为 $N$），但该过程中实际执行了 $N$ 次模型前向传播过程。即 `GPT` 类模型一次推理只输出一个 `token`，**当前轮输出的 token 与输入 tokens 拼接，并作为下一轮的输入 tokens**，反复多次直到遇到终止符 `EOS` 或生成的 `token` 数目达到设置的 `max_new_token` 才会停止。

可以看出第 $i$ 轮输入数据只比第 $i+1$ 轮输入数据新增了一个 `token`，其他全部相同！因此第 $i+1$ 轮推理时必然包含了第 $i$ 轮的部分计算。`KV Cache` 优化的起因就在这里，**缓存当前轮可重复利用的计算结果**，下一轮计算时直接读取缓存结果，原理很简单，**本质就是用空间换时间**。

另外，**之所以每一层 decode layer 都有需要缓存 $K$ 和 $V$，因为每层的 `attention` 运算是独立的，即第 $L$ 层的 $K_L$ 和 $V_L$ 是独立的、与其他层不同的**。

如果不缓存每一层的 $K$ 和 $V$，在生成下一个 token 时，模型就需要重新计算之前所有 `token` 的 $K$ 和 $V$，这将导致大量冗余计算。通过缓存，避免了重复计算 $K$ 和 $V$，从而加速了生成过程。

### 1.2 LLM 推理的两个阶段
结合前面 `KV cache` 优化的原理，可以总结出一个典型的自回归模型的生成式推理过程包含了两个阶段：

1. **预填充阶段**（prefill phase）：输入一个 prompt 序列，为每个 transformer 层生成 key cache 和 value cache（KV cache）。因为 `QKV` 矩阵的计算过程是线性的，因此，对于同一个 `seq` 样本，每个 `token` 可以并行计算无时序依赖。简单来说就是，训练、推理的 prefill 阶段（输入 prompt 的计算）过程是**高度并行化的**。
2. **解码阶段**（decoding phase）：使用并更新 KV cache，一个接一个地生成词（**无并行性**），当前生成的词依赖于之前已经生成的词。该阶段的推理计算分两部分：更新 KV cache 和计算 decoder layers 的输出。

这两个阶段的差别在于 $Q$ 的维度不同。在第一个阶段时，用户输入的所有 token 都需要参与运算，所以此时 Q 的维度为 [batch_size, seq_len, d<sub>model</sub>]。在第二个阶段时，新生成的 token 作为第二次迭代过程的输入，所以此时 Q 的维度为 [batch_size, 1, d<sub>model</sub>]，即**只有新 token 作为 Q**。

> 为什么不缓存 Q ?
> 这是由 decoder-only 的模型原理决定的。在每次迭代过程中，只有新生成 token 的 Q 需要参与运算（新生成 token 是由前文推理而出，所以其 Q 向量包含了前文信息），之前的 Q 无需再参与运算，所以每次迭代过程结束时就可以将 Q 丢弃。

## 二 显存占用量理论分析

### 2.1 推理过程中显存占用分析

深度学习模型推理任务中，占用 GPU 显存的主要包括三个部分：**模型权重、输入输出以及中间结果**。（该结论来源[论文](https://www.usenix.org/conference/osdi20/presentation/gujarati)）

1. **模型权重**：神经网络模型都是由相似的 layer 堆叠而成，例如 cnn 模型的卷积层、池化层、全连接层等；以及 transformer 模型的 self-attention 层、全连接层、layer_norm 层等。**模型权重默认使用 `fp16` 存储**。
2. **中间结果**：
   - **输出激活值**: 前向传播计算过程中，前一层的输出就是后一层的输入，**相邻两层的中间结果也是需要 gpu 显存来保存的**，中间结果变量也叫激活内存，值相对很小。
   - **多头自注意力机制的中间计算结果**：包括 Query、Key、Value 及它们的计算结果，尤其是点积和注意力权重计算，这些结果在每一层都需要暂时存储。
3. **输入输出**：和模型权重参数所占用的显存相比，输入输出所占用的显存就很小了。

`batch_size(bs)` 是一个很重要的参数，`batch_size` 的增加能带来近乎线性的 `throughput` 增加。虽然 `batch_size` 的增加会带来输入输出和中间结果显存的增加，但**在 < 某个阈值时，推理阶段显存占大头的是模型权重**。

### 2.2 训练过程中显存占用分析

**在模型训练过程中，设备内存中除了需要模型权重之外，还需要存储中间变量（激活）、梯度和化器状态动量**，后者显存占用量与 `batch size` 成正比。

$$训练总内存 = 模型内存 + 优化器内存 + 激活内存 + 梯度内存$$

在模型训练过程中，**存储前向传播的所有中间变量（激活）结果**，称为 `memory_activations`，用以在反向传播过程中计算梯度时使用。而模型中梯度的数量通常等于中间变量的数量，所以 `memory_activations = memory_gradients`。

假设 `memory_modal` 是指存储模型所有参数所需的内存、`memory_optimizer` 是优化器状态变量所需内存。综上，模型训练过程中，显存占用量的理论计算公式为：

```bash
total_memory = memory_modal + 2 * memory_activations + memory_optimizer
```

值得注意的是，**对于 LLM 训练而言，现代 GPU 通常受限于内存瓶颈，而不是算力**。因此，**激活重计算** (`activation recomputation`，或称为激活检查点 (`activation checkpointing`) ) 就成为一种非常流行的**以计算换内存**的方法。

**激活重计算**主要的做法是**重新计算某些层的激活而不是把它们存在 GPU 内存中，从而减少内存的使用量**，内存的减少量取决于我们选择清除哪些层的激活。

### 2.3 推理过程中显存占用量理论计算

1，存储模型权重参数所需的显存计算公式（`params` 是模型参数量）：

$$\text{memory\_model} = \text{params} * 2 = [n(12h^2 + 4h) + Vh] * 2$$

2，中间结果显存占用

和模型训练需要存储前向传播过程中的中间变量结果不同，模型推理过程中并不需要存储中间变量，因此推理过程中涉及到的**中间结果**内存会很小（中间结果用完就会释放掉），一般指**相邻两层的中间结果**或者算子内部的中间结果，这里我们只考虑主要算子中最大的中间结果部分即可。

这里我们假设其占用的显存为 `memory_intermediate`，`heads` 数量用符号 $n_\text{head}$ 表示，假设输入数据的形状为 $[b,s]$。
- 每个 self-attention 头需要计算 Query 和 Key 的点积，每个头的 $QK^T$ 矩阵乘法的输入输出形状为 $[b, head\_num, s, h//n\_head] \times [b, head\_num, h//n\_head, s] \rightarrow [b, head\_num, s, s]$，所以占用显存大小为 $2bs^2n_{head}$；
- `mlp` 块中，第一个线性层的输出结果形状为 $[b, s, 4h]$，所以占用显存大小为 $8bsh$。

计算 `MHA`和 `MLP` 的 `memory_intermediate` 的伪代码如下:

```bash
memory_intermediate of attention(qk^t output) = 2 * batch_size * n_head * square_of(sequence_length)
memory_intermediate of mlp(fc1 layer1 output) = 2 * batch_size * s * 4h
```

又因为 $h \gg s$，所以 `memory_intermediate of mlp` 远大于 `memory_intermediate of attention`。所以:

$$\text{memory\_intermediate} = 8bsh$$

3，`kv cache` 显存占用

`LLM` 推理优化中 `kv cache` 是常见的方法，本质是用空间换时间。假设输入序列的长度为 $s$ ，输出序列的长度为 $o$，decoder layers 数目为 $n$，以 `float16` 来保存 `KV cache`，那么 `KV cache` 的峰值显存占用计算公式为:

$$\text{memory\_kv-cache} = b(s+o)h*n * 2*2 = 4nbh(s+o)$$

上式，第一个 `2` 表示 K/V cache，第二个 `2`表示 float16 占 2 个 bytes。

4，最后，通过前面的分析可知，**模型推理阶段总的显存消耗计算公式如下**:

$$\text{inference\_memory} = 1.2*12nh^2 + 4nbh(s+o)$$

### 2.4 显存占用的定性分析和定量分析

1. 模型推理阶段，当输入输出上下文长度之和比较小的时候，占用显存的大头主要是模型参数，但是当输入输出上下文长度之和很大的时候，占用显存的大头主要是 `kv cache`。
2. 每个 `GPU` `kv cache` 显存所消耗的量和**输入 + 输出序列长度**成正比，和 `batch_size` 也成正比。
3. 有[文档](https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model)指出，`13B` 的 `LLM` 推理时，每个 `token` 大约消耗 `1MB` 的显存。

以 A100-40G GPU 为例，llama-13b 模型参数占用了 26GB，如果忽略中间显存，那么剩下的 14GB 显存中大约可以容纳 14,000 个 token。在部署项目中，如果将输入序列长度限制为 512，那么该硬件下最多只能同时处理大约 `28` 个序列。



## 参考资料

1. [优化故事: BLOOM 模型推理](https://huggingface.co/blog/zh/bloom-inference-optimization)
2. [使用 DeepSpeed 和 Accelerate 进行超快 BLOOM 模型推理](https://huggingface.co/blog/zh/bloom-inference-pytorch-scripts)
3. [如何使用 Megatron-LM 训练语言模型](https://huggingface.co/blog/zh/megatron-training)
4. [Transformer Deep Dive: Parameter Counting](https://orenleung.com/transformer-parameter-counting)
5. [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/pdf/2207.00032.pdf)
6. [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
7. [Estimating memory requirements of transformer networks](https://www.linkedin.com/pulse/estimating-memory-requirements-transformer-networks-schartz-rehan/?trackingId=q8AzwkgCSK6DhhcafunTgA%3D%3D)
8. [Formula to compute approximate memory requirements of Transformer models](https://stats.stackexchange.com/questions/563919/formula-to-compute-approximate-memory-requirements-of-transformer-models)
9.  [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
10. [如何估算transformer模型的显存大小](https://avoid.overfit.cn/post/6724eec842b740d482f73386b1b8b012)
11. [大模型推理性能优化之KV Cache解读](https://zhuanlan.zhihu.com/p/630832593)