# -*- coding  : utf-8 -*-
# Author: honggao.zhang + chatgpt

import torch
import time

import numpy as np
import time

class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((output_channels, 1))

    def forward(self, input):
        batch_size, input_channels, height, width = input.shape
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.kernel_size // 2, self.kernel_size // 2),
                                      (self.kernel_size // 2, self.kernel_size // 2)), mode='constant')
        output = np.zeros((batch_size, self.output_channels, height, width))
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for r in range(height):
                    for c in range(width):
                        # kernel 矩阵和 input 矩阵的, 默认 array1*array2 就是对应元素的乘积
                        padded_input[b, :, r: r+self.kernel_size, c: c+self.kernel_size] * self.weights[oc, :, :, :]
                        output[b, oc, r, c] = np.sum(
                            padded_input[b, :, r:r+self.kernel_size, c:c+self.kernel_size]
                            * self.weights[oc]) + self.bias[oc]
        return output

stride = 1
kernel_size = 3
for bs in range(batch_size):
    for oc in range(output_channels):
        output[bs, oc, oh, ow] += bias[oc]
        for ic in range(input_channels):
            for oh in range(height):
                for ow in range(width):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            output[bs, oc, oh, ow] += input[bs, ic, oh+kh, ow+kw] * weights[oc, ic, kh, kw]
batch_size = 32
input_channels = 3
output_channels = 16
height = 224
width = 224
kernel_size = 3

input = np.random.randn(batch_size, input_channels, height, width)

conv = Conv2D(input_channels, output_channels, kernel_size)

start_time = time.time()
output = conv.forward(input)
end_time = time.time()

print(f"Output shape: {output.shape}")
print(f"Time taken: {end_time - start_time:.6f} seconds")

class Conv2D(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        return self.conv(x)

batch_size = 32
input_channels = 3
output_channels = 16
height = 224
width = 224
kernel_size = 3

input = torch.randn(batch_size, input_channels, height, width)

conv = Conv2D(input_channels, output_channels, kernel_size)

start_time = time.time()
output = conv(input)
end_time = time.time()

print(f"Output shape: {output.shape}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
