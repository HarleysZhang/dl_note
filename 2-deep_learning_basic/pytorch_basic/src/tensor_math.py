import torch

# 创建一个 2D 张量
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("原始张量：\n", b)

# 沿第 0 维计算累积乘积
cumprod_b_dim0 = torch.cumprod(b, dim=0)
print("沿第 0 维的累积乘积结果：\n", cumprod_b_dim0)

# 沿第 1 维计算累积乘积
cumprod_b_dim1 = torch.cumprod(b, dim=1)
print("沿第 1 维的累积乘积结果：\n", cumprod_b_dim1)