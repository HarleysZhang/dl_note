# -*- coding  : utf-8 -*-
# Author: honggao.zhang + chatgpt
# Create: 2023-03-27
# Version     : 0.1.0
# Description: transformer 模型的 PyTorch 实现

import torch.nn as nn
import torch
import time

def timer(func):
    """decorator: print the cost time of run function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

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

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):