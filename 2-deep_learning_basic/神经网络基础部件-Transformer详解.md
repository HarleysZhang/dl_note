## 背景知识

### Word Embedding Vector

Word Embedding Vector 是将单词映射到向量空间的一种方式，它可以**将单词的语义信息转化为向量形式**，并且可以在向量空间中计算单词之间的相似性，简单理解就是，将单词从原始文本格式转换为神经网络可以理解和处理的数字格式的一种方式。在深度学习中，我们通常使用 Word2Vec、GloVe 或 FastText 等算法来生成单词的嵌入向量。

下面是一个简单的 Python 示例，用于使用 PyTorch 生成单词的嵌入向量：

```python
import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from collections import Counter

text = "hello world hello" # 假设我们有一个文本
counter = Counter(text.split()) # 统计每个单词的出现次数
vocab = Vocab(counter) # 创建词汇表

# 构建嵌入层
vocab_size = 1000
embedding_dim = 100
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 假设我们有一个输入单词列表，每个单词都是从词汇表中随机选择的
input_words = ["hello", "world", "this", "is", "a", "test"]

# 将单词转换为索引列表
word_indexes = [vocab.index(word) for word in input_words]

# 将索引列表转换为PyTorch张量
word_indexes_tensor = torch.LongTensor(word_indexes)

# 将索引列表输入嵌入层以获取嵌入向量
word_embeddings = embedding_layer(word_indexes_tensor)

# 输出嵌入向量
print(word_embeddings)
```

1.  首先通过 `Vocab` 和输入文本字符串创建词汇表 `vocab`，并通过 `nn.Embedding` 创建了一个大小为 $1000\times 100$ 的嵌入层 `embedding_layer`，表示**词汇表 **中有 `1000` 个单词，每个单词用一个 `100` 维的向量表示。
2. 然后，我们创建一个输入单词列表 input_words，并将每个单词转换为词汇表中对应的索引，和将索引列表转换为 PyTorch 张量。
3. 最后将张量 `word_indexes_tensor` 输入到嵌入层中，以获取每个单词的 100 维嵌入向量。

值得注意的是，嵌入向量的大小和维度通常需要根据任务进行调整。通常，词汇表越大，嵌入向量的维度就越高。此外，**嵌入向量的大小通常需要与模型中的其他参数相匹配，例如隐藏层的大小和注意力头的数量**。

### nn.Linear

在 PyTorch 中，`nn.Linear` 是一个模块，它实现了一个全连接层，它将输入张量的每个元素都乘以一个可学习的权重矩阵，并加上一个可学习的偏置向量，最后输出一个新的张量。全连接层的数学表达式为：
$$
\text{output} = \text{input} \times \text{weight}^\text{T} + \text{bias} \nonumber
$$
`nn.Linear` 的构造函数如下：

```python
nn.Linear(in_features: int, out_features: int, bias: bool = True)
```

各参数解释如下：

- `in_features` 表示输入张量的特征数，也就是输入张量的最后一维的大小，
- `out_features` 表示输出张量的特征数，也就是输出张量的最后一维的大小，
- `bias` 是一个布尔值，表示是否添加偏置向量。如果 `bias` 为 `False`，则不会添加偏置向量。

下面是一个示例代码，可直接运行，其创建了一个输入张量，并通过 层进行维度的线性变换。

```python
import torch.nn as nn
import torch
import time

# 创建一个输入张量
input_tensor = torch.randn(10, 20)

# 创建一个 Linear 层对象，将输入维度从 20 变换到 30
linear_layer = nn.Linear(20, 30)

# 将输入张量传递给 Linear 层进行变换
start_time = time.time()
output_tensor = linear_layer(input_tensor)
end_time = time.time()

# 打印输出张量的形状
print(f"Output shape: {output_tensor.shape}") # Output shape: torch.Size([10, 30])
print(f"Time taken: {end_time - start_time:.6f} seconds") # Time taken: 0.002543 seconds
```

## 一，Transformer 架构

论文中给出用于中英文翻译任务的 `Transformer` 整体架构如下图所示：

<img src="../images/transfomer/transformer_architecture.png" style="zoom:50%;" />

可以看出 Transformer 架构**由 Encoder 和 Decoder 两个部分组成**：其中 Encoder 和 Decoder 都是由 N=6 个相同的层堆叠而成。

## 二，Multi-Head Attention 结构

Encoder 和 Decoder 结构中公共的 `layer` 之一是 `Multi-Head Attention`，其是由多个 `Self-Attention` 并行组成的。Encoder block 只包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。

<img src="../images/transfomer/multi-head-attention2.png" alt="multi-head-attention" style="zoom:60%;" />

### 2.1，Self-Attention 结构

`Self-Attention` 中文翻译为**自注意力机制**，论文中叫作 `Scale Dot Product Attention`，它是 Transformer 架构的核心，其结构如下图所示：

![../images/transfomer/scale_dot_product_attention.jpeg](../images/transfomer/scale_dot_product_attention.jpeg)

那么重点来了，Self-Attention 结构的最初输入 **Q(查询), K(键值), V(值)** 这三个矩阵怎么理解呢？其代表什么，通过什么计算而来？

在 Self-Attention 中，Q、K、V 是在同一个输入（比如序列中的一个单词）上计算得到的三个向量。具体来说，我们可以通过对原始输入**词的 embedding** 进行线性变换（比如使用一个全连接层），来得到 Q、K、V。这三个向量的维度通常都是一样的，取决于模型设计时的决策。

在计算 Self-Attention 时，Q、K、V 被用来**计算注意力分数**，即用于表示**当前位置和其他位置之间的关系**。注意力分数可以通过 Q 和 K 的点积来计算，然后将分数除以 8，再经过一个 softmax 归一化处理，得到每个位置的权重。然后用这些权重来加权计算 V 的加权和，即得到当前位置的输出。

> 将分数除以 8 的操作，对应图中的 `Scale` 层，这个参数 8 是 K 向量维度 64 的平方根结果。

### 2.2，Self-Attention 实现

在 Self-Attention 层中，对于每个单词的 Word Embedding Vector，我们都会创建一个查询向量 Q、一个键向量 K 和一个值向量 V，这些新向量的维度会小于 embedding vector。在论文中 Q、K 和 V 的向量维度是 64，另外 Embedding Vector 和 Encoder-Decoder 输入输出的维度都是 512。

Self-Attention 层的计算过程用数学公式可表达为:
$$
\text{Attention}(Q, K, V) = \text{softmax} (\frac{QK^T}{\sqrt{d_k}})V \nonumber
$$
在 Self-Attention 层中，每个输入序列中的每个词语都会被映射到 Q、K 和 V 三个向量，它们尺寸相同。假设输入序列的长度为 seq_len，则 Q、K 和 V 的形状为（seq_len，d_k），其中，$\text{d}_{\text{k}}$ 表示每个词或向量的维度。

$d_k$ 是 $Q$、$K$ 矩阵的列数，即向量的维度。

以下是一个示例代码，它创建了一个 ScaleDotProductAttention 层，并将 Q、K、V 三个张量传递给它进行计算：

```python
class ScaleDotProductAttention(nn.Module):
    def __init__(self, ):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置  
        d_k = Q.size(-1)
        # 1, 计算 Q, K^T 矩阵的点积，再除以 sqrt(d_k) 得到注意力分数矩阵
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        # 2, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 3, 对注意力分数矩阵按照最后一个维度进行 softmax 操作，得到注意力权重矩阵，值范围为 [0, 1]
        attn_weights = self.softmax(scores)
        # 4, 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# 创建 Q、K、V 三个张量
Q = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)
K = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)
V = torch.randn(5, 10, 64)  # (batch_size, sequence_length, d_k)

# 创建 ScaleDotProductAttention 层
attention = ScaleDotProductAttention()

# 将 Q、K、V 三个张量传递给 ScaleDotProductAttention 层进行计算
output, attn_weights = attention(Q, K, V)

# 打印输出矩阵和注意力权重矩阵的形状
print(f"ScaleDotProductAttention output shape: {output.shape}") # torch.Size([5, 10, 64])
print(f"attn_weights shape: {attn_weights.shape}") # torch.Size([5, 10, 10])
```

### 2.3，Multi-Head Attention

Multi-Head Attention (`MHA`) 是基于 Self-Attention (`SA`) 的一种变体。在 SA 中，我们将输入的序列中的每个元素视为 Query、Key 和 Value，计算它们之间的注意力得分，然后将这些得分应用于 Value 来计算输出。MHA 在 SA 的基础上引入了**“多头”机制**，将输入拆分为多个子空间，每个子空间分别执行 `SA`，最后将多个子空间的输出拼接在一起并进行线性变换，从而得到最终的输出。

对于 `MHA`，之所以需要对 Q、K、V 进行多头（`head`）划分，其目的是为了增强模型对不同信息的关注。具体来说，多组 Q、K、V 分别计算 Self-Attention，每个头自然就会有独立的 Q、K、V 参数，从而让模型同时关注多个不同的信息，这有些类似 `CNN` 架构模型的**多通道机制**。

下图是论文中 Multi-Head Attention 的结构图。

![multi_head_attention](../images/transfomer/multi-head-attention3.png)

从图中可以看出， `MHA` 结构的计算过程可总结为下述步骤:

1. 将输入 Q、K、V 张量进行线性变换（`Linear` 层），其尺寸为 [batch_size, seq_len, d_model]；
2. 将前面步骤输出的张量，按照头的数量（`n_head`）拆分为 `n_head` 子张量，其尺寸为 [batch_size, n_head, seq_len, d_model//n_head]；
3. 每个子张量并行**计算注意力分数**，即执行 dot-product attention 层，输出张量尺寸为 [batch_size, n_head, seq_len, d_model//n_head]；
4. 将这些子张量进行拼接 `concat` ，并经过线性变换得到最终的输出张量，尺寸为 [batch_size, seq_len, d_model]。

总结：因为 `GPU` 的并行计算特性，步骤2中的张量拆分和步骤4中的张量拼接，其实都是通过 `review` 算子来实现的。同时，也能发现`SA` 和 `MHA` 模块的输入输出矩阵维度都是一样的。

### 2.4，Multi-Head Attention 实现

Multi-Head Attention 层的输入同样也是三个张量：**查询（Query）、键（Key）和值（Value）**，其用数学公式可表达为:
$$
\text{MultiHead(Q, K, V )} = \text{Concat}(\text{head}_{1}, ..., \text{head}_{\text{h}})W^O \\
 \text{where head}_{\text{i}} = \text{Attention}(QW_i^Q , KW_i^K , VW_i^V )
$$
一般用 `d_model` 表示输入嵌入向量的维度， `n_head` 表示分割成多少个头，因此，`d_model//n_head` 自然表示每个头的输入和输出维度，在论文中 d_model = 512，n_head = 8，d_model//n_head = 64。值得注意的是，由于每个头的维数减少，总计算成本与具有全维的单头注意力是相似的。

Multi-Head Attention 层的 `Pytorch` 实现代码如下所示：

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer
    Args:
        d_model: Dimensions of the input embedding vector, equal to input and output dimensions of each head
        n_head: number of heads, which is also the number of parallel attention layers
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)  # Q 线性变换层
        self.w_k = nn.Linear(d_model, d_model)  # K 线性变换层
        self.w_v = nn.Linear(d_model, d_model)  # V 线性变换层
        self.fc = nn.Linear(d_model, d_model)   # 输出线性变换层
        
    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # size is [batch_size, seq_len, d_model]
        # 2, split by number of heads(n_head) # size is [batch_size, n_head, seq_len, d_model//n_head]
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3, compute attention
        sa_output, attn_weights = self.attention(q, k, v, mask)
        # 4, concat attention and linear transformation
        concat_tensor = self.concat(sa_output)
        mha_output = self.fc(concat_tensor)
        
        return mha_output
    
    def split(self, tensor):
        """
        split tensor by number of head(n_head)

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_model//n_head], 输出矩阵是四维的，第二个维度是 head 维度
        
        # 将 Q、K、V 通过 reshape 函数拆分为 n_head 个头
        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, n_head, d_model // n_head)
        k = k.reshape(batch_size, seq_len, n_head, d_model // n_head)
        v = v.reshape(batch_size, seq_len, n_head, d_model // n_head)
        """
        
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        
        return split_tensor
    
    def concat(self, sa_output):
        """ merge multiple heads back together

        Args:
            sa_output: [batch_size, n_head, seq_len, d_tensor]
            return: [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_tensor = sa_output.size()
        d_model = n_head * d_tensor
        concat_tensor = sa_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return concat_tensor
```

## 三，Encoder 结构

Encoder 结构由 $\text{N} = 6$ 个相同的 encoder block 堆叠而成，每一层（ layer）主要有两个子层（sub-layers）: 第一个子层是多头注意力机制（`Multi-Head Attention`），第二个是简单的位置全连接前馈网络（`Positionwise Feed Forward`）。

Encoder block 输入矩阵和输出矩阵维度是一样的，都是 $n\times d$。

## 四，Decoder 结构

## 参考资料

1. https://github.com/hyunwoongko/transformer
2. [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
