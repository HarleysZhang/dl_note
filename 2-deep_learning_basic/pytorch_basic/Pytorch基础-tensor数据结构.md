- [一 Tensor 概述](#一-tensor-概述)
  - [1.1 Tensor 数据类型](#11-tensor-数据类型)
  - [1.2 Tensor 的属性](#12-tensor-的属性)
- [二 创建 Tensor](#二-创建-tensor)
  - [2.1 传数据的方法创建 Tensor](#21-传数据的方法创建-tensor)
  - [2.2 传 size 的方法创建 Tensor](#22-传-size-的方法创建-tensor)
  - [2.3 其他创建 tensor 的方法](#23-其他创建-tensor-的方法)
  - [创建张量方法总结](#创建张量方法总结)
- [参考资料](#参考资料)

## 一 Tensor 概述

`torch.Tensor` 是一种包含**单一数据类型**元素的多维矩阵，类似于 numpy 的 `array`。

![tensor](../../images/tensor_datastructure/tensor.png)

1，指定数据类型的 tensor 可以通过传递参数 `torch.dtype` 和/或者 `torch.device` 到构造函数生成：
> 注意为了改变已有的 tensor 的 torch.device 和/或者 torch.dtype, 考虑使用 `to()` 方法.

```python
>>> torch.ones([2,3], dtype=torch.float64, device="cuda:0")
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0', dtype=torch.float64)
>>> torch.ones([2,3], dtype=torch.float32)
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

2，Tensor 的内容可以通过 Python 索引或者切片访问以及修改，用法和 `ndarray` 的操作一致：

```python
>>> matrix = torch.tensor([[2,3,4],[5,6,7]])
>>> print(matrix[1][2])
tensor(7)
>>> matrix[1][2] = 9
>>> print(matrix)
tensor([[2, 3, 4],
        [5, 6, 9]])
```

3，使用 `torch.Tensor.item()` 或者 `int()` 方法从**只有一个值的 Tensor**中获取 Python 数值对象：

```python
>>> x = torch.tensor([[4.5]])
>>> x
tensor([[4.5000]])
>>> x.item()
4.5
>>> int(x)
4
```

4，Tensor可以通过参数 `requires_grad=True` 创建, 这样 `torch.autograd` 会记录相关的运算实现自动求导：

```python
>>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
>>> out = x.pow(2).sum()
>>> out.backward()
>>> x.grad
tensor([[ 2.0000, -2.0000],
 [ 2.0000,  2.0000]])
```

5，每一个 tensor都有一个相应的 `torch.Storage` 保存其数据。tensor 类提供了一个多维的、strided 视图, 并定义了数值操作。

6，张量和 numpy 数组。可以用 `.numpy()` 方法从 Tensor 得到 numpy 数组，也可以用 `torch.from_numpy` 从 numpy 数组得到Tensor。这两种方法关联的 Tensor 和 numpy 数组是共享数据内存的。可以用张量的 `clone`方法拷贝张量，中断这种关联。

```python
arr = np.random.rand(4,5)
print(type(arr))
tensor1 = torch.from_numpy(arr)
print(type(tensor1))
arr1 = tensor1.numpy()
print(type(arr1))
"""
<class 'numpy.ndarray'>
<class 'torch.Tensor'>
<class 'numpy.ndarray'>
"""
```

7，2，`item()` 方法和 `tolist()` 方法可以将张量转换成 Python 数值和数值列表

```python
# item方法和tolist方法可以将张量转换成Python数值和数值列表
scalar = torch.tensor(5)  # 标量
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(3,2)  # 矩阵
t = tensor.tolist()
print(t)
print(type(t))
"""
1.0
<class 'float'>
[[0.8211846351623535, 0.20020723342895508], [0.011571824550628662, 0.2906131148338318]]
<class 'list'>
"""
```

### 1.1 Tensor 数据类型

Torch 定义了七种 CPU Tensor 类型和八种 GPU Tensor 类型：

![tensor数据类型](../../images/tensor_datastructure/tensor_data_types.png)

`torch.Tensor` 是默认的 tensor 类型（`torch.FloatTensor`）的简称，即 `32` 位浮点数数据类型。

### 1.2 Tensor 的属性

Tensor 有很多属性，包括数据类型、Tensor 的维度、Tensor 的尺寸。

+ **数据类型**：可通过改变 torch.tensor() 方法的 `dtype` 参数值，来设定不同的 `Tensor` 数据类型。
+ **维度**：不同类型的数据可以用不同维度(dimension)的张量来表示。标量为 `0` 维张量，向量为 `1` 维张量，矩阵为 `2` 维张量。彩色图像有 `rgb` 三个通道，可以表示为 `3` 维张量。视频还有时间维，可以表示为 `4` 维张量，有几个中括号 `[` 维度就是几。**可使用 `dim() 方法` 获取 `tensor` 的维度**。
+ **尺寸**：可以使用 `shape属性`或者 `size()方法`查看张量在每一维的长度，可以使用 `view()方法`或者`reshape() 方法`改变张量的尺寸。Pytorch 框架中四维张量形状的定义是 `(N, C, H, W)`。
+ **张量元素总数**：numel() 方法返回（输入）张量元素的总数。
+ **设备**：`.device` 返回张量所在的设备。

> 关于如何理解 Pytorch 的 Tensor Shape 可以参考 stackoverflow 上的这个 [回答](https://stackoverflow.com/questions/52370008/understanding-pytorch-tensor-shape)。

样例代码如下：

```python
matrix = torch.tensor([[[1,2,3,4],[5,6,7,8]],
                       [[5,4,6,7], [5,6,8,9]]], dtype = torch.float64)
print(matrix)               # 打印 tensor
print(matrix.dtype)     # 打印 tensor 数据类型
print(matrix.dim())     # 打印 tensor 维度
print(matrix.size())     # 打印 tensor 尺寸
print(matrix.shape)    # 打印 tensor 尺寸
matrix2 = matrix.view(4, 2, 2) # 改变 tensor 尺寸
print(matrix2)
print(matrix.numel())
```
```python
>>> matrix = torch.tensor([[[1,2,3,4],[5,6,7,8]],
...                        [[5,4,6,7], [5,6,8,9]]], dtype = torch.float64)
>>> matrix
tensor([[[1., 2., 3., 4.],
         [5., 6., 7., 8.]],

        [[5., 4., 6., 7.],
         [5., 6., 8., 9.]]], dtype=torch.float64)
>>> matrix.dtype        # tensor 数据类型
torch.float64
>>> matrix.dim()        # tensor 维度
3
>>> matrix.size()       # tensor 尺寸
torch.Size([2, 2, 4])
>>> matrix.shape        # tensor 形状
torch.Size([2, 2, 4])
>>> matrix.numel()      # tensor 元素总数
16
>>> matrix.device
device(type='cpu')
```

## 二 创建 Tensor

创建 tensor ，可以传入数据或者维度，torch.tensor() 方法只能传入数据，torch.Tensor() 方法既可以传入数据也可以传维度，强烈建议 tensor() 传数据，Tensor() 传维度，否则易搞混。

具体来说，一般使用 torch.tensor() 方法将 python 的 `list` 或 numpy 的 `ndarray` 转换成 Tensor 数据，生成的是`dtype`  默认是 `torch.FloatTensor`，和 torch.float32 或者 torch.float 意义一样。

通过 torch.tensor() 传入数据的方法创建 tensor 时，`torch.tensor()` 总是拷贝 data 且一般不会改变原有数据的数据类型 `dtype`。如果你有一个 tensor data 并且仅仅想改变它的 `requires_grad` 属性，可用 `requires_grad_()` 或者 `detach()` 来避免拷贝。如果你有一个 `numpy` 数组并且想避免拷贝，请使用 `torch.as_tensor()`。

### 2.1 传数据的方法创建 Tensor

1，`torch.tensor()`。

将 python 的 `list` 或 numpy 的 `ndarray` 转换成 Tensor 数据。
```python
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
```
参数解释：
- `data`: 数据，可以是 list，ndarray
- `dtype`: 数据类型，默认与 data 的一致
- `device`: 所在设备，cuda/cpu
- `requires_grad`: 是否需要梯度
- `pin_memory`: 是否存于锁页内存

代码示例：

```python
>>> a = np.arange(12).reshape(3,4)
>>> b = torch.tensor(a)
>>> b        # 打印张量 b 数据
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>> a.dtype
dtype('int64')
>>> b.dtype # 张量 b 元素的数据类型（和 numpy 数组 a 一致 ）
torch.int64
>>> a.shape 
(3, 4)
>>> b.shape # 张量 b 的形状
torch.Size([3, 4])
>>> b.dim() # 张量 b 的维度
2
>>> b.numel() # 张量 b 的元素个数
12
```

2，`torch.from_numpy(ndarray)`

从 numpy 创建 tensor。利用这个方法创建的 tensor 和原来的 ndarray 共享内存（不会拷贝数据，节省内存和时间），当修改其中一个数据，另外一个也会被改动。

```python
>>> arr = np.arange(1, 7).reshape(2, 3)
>>> ta = torch.from_numpy(arr)
>>> ta[1][2] = 100 # 修改第 2 行 第 3 列元素值为 100
>>> ta
tensor([[  1,   2,   3],
        [  4,   5, 100]])
>>> arr
array([[  1,   2,   3],
       [  4,   5, 100]])
```

3，`torch.empty_like`、`torch.zeros_like`、`torch.ones_like`、`torch.randint_like()`等
```python
torch.empty_like(input, *, dtype=None,) -> Tensor
```
根据 input（tensor 数据） 形状创建空、全 0 和全 1 的张量。

```python
arr = np.arange(20).reshape(5, 4)
a_tensor = torch.from_numpy(arr)
b_like = torch.empty_like(a_tensor)
c_like = torch.zeros_like(a_tensor)
assert a_tensor.shape == b_like.shape == c_like.shape == torch.Size([5,4])
print(c_like.shape)
print(c_like)
"""
torch.Size([5, 4])
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])
"""
```

### 2.2 传 size 的方法创建 Tensor

1，`torch.empty`、`torch.zeros`、`torch.ones` 等方法。

直接传入张量形状即可创建形状为 `size` 的空、全 0 和全 1 的张量。

```python
torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

代码示例：
```python
>>> torch.zeros(4,5,6)
tensor([[[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]]])
```

### 2.3 其他创建 tensor 的方法

1，`torch.arange()`

功能和参数 `np.arange()` 几乎一样，默认创建区间为 `[0, end)` 公差为 $1$ 的 1维张量。

```python
torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```
参数解释：
- start: 数列起始值
- end: 数列结束值，开区间，取不到结束值
- step: 数列公差，默认为 `1`

2，`torch.normal()`

根据给定的均值 (mean=0) 和标准差 (std=1) 从正态分布中抽取随机样本。每个生成的元素都是独立的随机变量，遵循相同的正态分布。

值的注意的是，当我们生成一个有限大小的样本（例如 3x3 共9个元素）时，样本的均值和标准差并不一定严格等于总体的均值和标准差。这是因为：

- 抽样误差（Sampling Error）: 由于样本量有限，样本统计量（如均值、标准差）可能会偏离总体参数。这种偏差是自然的随机现象，随着样本量的增加，样本统计量会更接近总体参数。
- 有限样本量的波动: 在小样本中，随机波动对统计量的影响较大。例如，在仅有 9 个样本的情况下，个别极端值可能显著影响均值和标准差。

**参数说明**：
- `mean` (float 或 Tensor): 正态分布的均值。如果是张量，必须与 std 的形状一致。
- `std` (float 或 Tensor): 正态分布的标准差。如果是张量，必须与 mean 的形状一致。
- `size` (tuple): 输出张量的形状。
- `generator` (torch.Generator, 可选): 用于生成随机数的随机数生成器。
- `out` (Tensor, 可选): 用于存储输出结果的张量。
返回一个张量，张量中的随机数从各自的正态分布中抽取，这些正态分布的均值和标准差是给定的。

这个函数有 4 种模式，这里给出常见 `torch.normal(mean=float, std=float, size, *)` 的用法示例。

```python
torch.normal(mean, std, size, *, out=None) → Tensor
```

参数解释:
- `mean`（float）：所有分布的均值
- `std`（float）：所有分布的标准差
- `size`（int…）：定义输出张量形状的整数序列

3，`torch.randn()` 或 `torch.randn_like()`

功能：返回形状为 `size` 的一个张量，张量中的随机数来自均值为 0、方差为 1 的正态分布（也称为标准正态分布）。这里计算 `torch.mean(4,5).mean()` 的结果不为 0 的原因分析和上面一样，不再说明。

```python
torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
```

4，`torch.randint()` 和 `torch.randint_like()`

功能：在区间 `[low, high)` 上生成指定形状 `size` 且均匀分布随机整数张量。

```python
torch.randint（low=0， high， size， *， generator=None， out=None， dtype=None， layout=torch.strided， device=None， requires_grad =False）-> Tensor
```

5，`torch.tril` 和 `torch.tril` (triu 和 tril 分别是“triangle upper”和“triangle lower”的缩写。)

```python
torch.tril(input, diagonal=0, *, out=None) → Tensor
```

- `torch.tril` 函数返回输入矩阵（2D 张量）或一组矩阵的**下三角部分**，结果张量中的其余元素被设置为 0。下三角部分包含矩阵的主对角线及其以下的元素。
- `torch.triu` 函数返回一个矩阵的上三角部分（即矩阵上半部分，主对角线及其以上的元素），其余部分设为零。

参数 `diagonal` 控制要保留的对角线。如果 diagonal = 0，则保留主对角线及其以下的所有元素。正值的 diagonal 会包括主对角线上方的对角线，负值则排除主对角线下方的对角线，即表示向上或向下偏移的对角线。主对角线的索引为  ${(i, i)}$ ，其中 $i \in [0, \min \{d_1, d_2\} - 1]$， $d_1$ 和 $d_2$ 分别为矩阵的维度。

代码示例:
```python
>>> torch.arange(12).reshape(4,3)               # torch.arange() 用法
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
>>> torch.normal(mean=0, std=2, size=[1,4])    # torch.normal() 用法
tensor([[ 0.6018, -0.2399,  2.8425,  1.6153]]) # torch.randn() 用法
>>> torch.randn(4,6,2)
tensor([[[ 1.3282, -0.0920],
         [ 0.4889,  0.0805],
         [-0.5224, -0.5830],
         [-0.7645, -0.6670],
         [ 0.2376,  0.0135],
         [-0.3824,  0.1190]],

        [[ 1.0024, -1.6934],
         [ 0.2822, -0.1121],
         [ 0.1233,  0.4210],
         [ 1.5558,  1.1571],
         [-1.9819, -1.0007],
         [ 1.7181,  0.5641]],

        [[ 0.2924,  0.7369],
         [ 0.4954, -2.3034],
         [-1.1726,  0.7474],
         [-0.1254, -0.2139],
         [-0.3428,  1.2906],
         [ 1.2389, -0.5154]],

        [[ 0.8589,  2.7191],
         [-0.0905,  0.3279],
         [ 1.8878,  0.6622],
         [-0.1519,  0.4263],
         [-0.9688,  1.2181],
         [-2.0909, -0.3234]]])
>>> torch.randint(10, (2,5))           # torch.randint() 用法
tensor([[3, 8, 7, 3, 5],
        [9, 2, 2, 9, 6]])
>>> torch.randn(4,5).tril()            # 创建下三角矩阵
tensor([[-0.8062,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.4425,  0.8054,  0.0000,  0.0000,  0.0000],
        [ 0.9237,  0.8548,  0.2412,  0.0000,  0.0000],
        [-0.3921, -0.4880, -0.8847, -0.2170,  0.0000]])
>>> torch.randn(4,5).triu()           # 创建上三角矩阵
tensor([[-0.7354, -0.9383, -0.0798, -0.6155,  0.3702],
        [ 0.0000,  0.5686,  0.2166, -0.5724, -0.0596],
        [ 0.0000,  0.0000,  0.0751, -0.9832,  1.5582],
        [ 0.0000,  0.0000,  0.0000, -0.0491,  1.3136]])
```

检验 torch.normal 和 torch.randn 的均值情况：

```python
import torch

def generate_tensor_and_print_stats(size, func="normal", mean=0.0, std=1.0):
    """
    生成一个张量并打印其均值和标准差。

    参数:
    - size (tuple): 生成张量的形状。
    - func (str): 使用的生成函数。选项包括 "normal" 和 "randn"。
    - mean (float, 可选): 正态分布的均值，仅在 func="normal" 时使用。默认值为0.0。
    - std (float, 可选): 正态分布的标准差，仅在 func="normal" 时使用。默认值为1.0。

    返回:
    - tensor (Tensor): 生成的张量。
    """
    if func == "normal":
        tensor = torch.normal(mean=mean, std=std, size=size)
    elif func == "randn":
        tensor = torch.randn(size=size)
    else:
        raise ValueError(f"无效的函数类型: {func}. 请使用 'normal' 或 'randn'。")
    
    tensor_mean = tensor.mean().item()
    tensor_std = tensor.std().item()
    
    print(f"Tensor Size: {size}, by torch.{func}, Mean: {tensor_mean:.4f}, Std Dev: {tensor_std:.4f}")
    return tensor

def main():
    # 设置随机种子以确保结果可重复（可选）
    torch.manual_seed(40)

    # 定义不同的张量大小
    sizes = [(3, 3), (100, 100), (1000, 1000)]
    functions = ["normal", "randn"]

    # 生成并打印统计信息
    for func in functions:
        for size in sizes:
            if func == "normal":
                generate_tensor_and_print_stats(size=size, func=func, mean=0.0, std=1.0)
            elif func == "randn":
                generate_tensor_and_print_stats(size=size, func=func)

if __name__ == "__main__":
    main()
```

程序运行后输出结果如下所示:

```bash
Tensor Size: (3, 3), by torch.normal, Mean: 0.4151, Std Dev: 0.6608
Tensor Size: (100, 100), by torch.normal, Mean: -0.0135, Std Dev: 0.9947
Tensor Size: (1000, 1000), by torch.normal, Mean: 0.0002, Std Dev: 1.0006
Tensor Size: (3, 3), by torch.randn, Mean: -0.7619, Std Dev: 1.1099
Tensor Size: (100, 100), by torch.randn, Mean: -0.0133, Std Dev: 1.0029
Tensor Size: (1000, 1000), by torch.randn, Mean: -0.0001, Std Dev: 1.0001
```

解释：
- $3\times 3$ 张量: 样本均值和标准差有较大的偏差。
- $100\times 100$ 张量: 偏差减小，均值更接近0，标准差更接近1。
- $1000\times 1000$ 张量: 偏差进一步减小，均值非常接近0，标准差非常接近1。

### 创建张量方法总结

|方法名|方法功能|备注|
|-----|-------|---|
|`torch.rand(*sizes, out=None) → Tensor`|返回一个张量，包含了从区间 `[0, 1)` 的**均匀分布**中抽取的一组随机数。张量的形状由参数sizes定义。|推荐|
|`torch.randn(*sizes, out=None) → Tensor`|返回一个张量，包含了从**标准正态分布**（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。|不推荐|
|`torch.normal(means, std, out=None) → Tensor`|返回一个张量，包含了从指定均值 `means` 和标准差 `std` 的离散正态分布中抽取的一组随机数。标准差 `std` 是一个张量，包含每个输出元素相关的正态分布标准差。|多种形式，建议看源码|
|`torch.rand_like(a)`|根据数据 `a` 的 shape 来生成随机数据|不常用|
|`torch.randint(low=0, high, size)`|生成指定范围(`low, hight`)和 `size` 的随机整数数据|常用|
|`torch.full([2, 2], 4)`|生成给定维度，全部数据相等的数据|不常用|
|`torch.arange(start=0, end, step=1, *, out=None)`|生成指定间隔的数据|易用常用|
|`torch.ones(*size, *, out=None)`|生成给定 size 且值全为1 的矩阵数据|简单|
|`zeros()/zeros_like()/eye()`|全 `0` 的 tensor 和 对角矩阵|简单|

样例代码：

```python
>>> torch.rand([1,1,3,3])
tensor([[[[0.3005, 0.6891, 0.4628],
          [0.4808, 0.8968, 0.5237],
          [0.4417, 0.2479, 0.0175]]]])
>>> torch.normal(2, 3, size=(1, 4))
tensor([[3.6851, 3.2853, 1.8538, 3.5181]])
>>> torch.full([2, 2], 4)
tensor([[4, 4],
        [4, 4]])
>>> torch.arange(0,10,2)
tensor([0, 2, 4, 6, 8])
>>> torch.eye(3,3)
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
```

## 参考资料

+ [PyTorch：view() 与 reshape() 区别详解](https://blog.csdn.net/Flag_ing/article/details/109129752)
+ [torch.rand和torch.randn和torch.normal和linespace()](https://zhuanlan.zhihu.com/p/115997577)
