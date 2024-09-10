import numpy as np
import time

# 设置数组的维度
size = 1000

# 创建两个随机数组
a = np.random.rand(size, size)
b = np.random.rand(size, size)

# 记录开始时间
start_time = time.time()

# 执行矩阵乘法
c = np.dot(a, b)

# 记录结束时间
end_time = time.time()

# 计算计算时间
elapsed_time = end_time - start_time

# 计算浮点操作数（FLOP）
# 矩阵乘法的浮点操作数 = 2 * size^3
flops = 2 * (size ** 3)

# 计算每秒的浮点运算数（GFLOPS）
gflops = flops / elapsed_time / 1e9

print(f"Attainable GFLOPS/sec: {gflops:.2f}")