# -*- coding  : utf-8 -*-
# Author: honggao.zhang + chatgpt
# Create: 2023-03-27
# Version     : 0.1.0
# Description: transformer 模型的 PyTorch 实现

import torch.nn as nn
import torch
import time
import math

def timer(func):
    """decorator: print the cost time of run function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

################## 1，nn.Linear 层理解 ##################
# 创建一个输入张量
input_tensor = torch.randn(10, 20)

# 创建一个 Linear 层，将输入维度从 20 变换到 30
linear_layer = nn.Linear(20, 30)

# 将输入张量传递给 Linear 层进行变换
start_time = time.time()
output_tensor = linear_layer(input_tensor)
end_time = time.time()

# 打印输出张量的形状
print(f"Output shape: {output_tensor.shape}")
print(f"Time taken: {end_time - start_time:.6f} seconds")

################## 2，词向量层算法 ##################
# 定义一个包含10个单词的词表，每个单词的嵌入维度为5
vocab_size = 10
embedding_dim = 5

# 定义一个词向量层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 假设输入句子为 "hello world"
input_sentence = ["hello", "world"]

# 将输入句子转化为词向量表示
input_embedding = embedding(torch.LongTensor([0, 1]))
# 最后，得到输入句子的词向量表示，尺寸为 (2, 5)。
print(input_embedding)
"""输入句子的词向量表示:
tensor([[ 0.1137, -0.4891, -0.2531, -0.6857, -0.2842],
        [-0.0712,  0.2119, -2.2433, -0.2108,  0.3078]],
       grad_fn=<EmbeddingBackward0>)
"""

################## word embedding ##################
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

################## 3，ScaleDotProductAttention 层实现 ##################
class ScaleDotProductAttention(nn.Module):
    """compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    
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

################## 4, MultiHeadAttention 层实现 ##################
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