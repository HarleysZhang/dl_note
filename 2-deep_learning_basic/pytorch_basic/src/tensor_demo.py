import torch

def tensor_learn():
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
    tensor_learn()
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
