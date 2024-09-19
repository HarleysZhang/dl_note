import torch
import torch.nn as nn
import time

# 生成随机数据进行测试
def benchmark(model, input_tensor, batch_size, device, num_iterations=100):
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 预热 GPU
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)
    
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            output = model(input_tensor)
    
    elapsed_time = time.time() - start_time
    images_per_sec = batch_size * num_iterations / elapsed_time
    return images_per_sec

# 定义卷积块
class ConvBlock(nn.Module):
    def __init__(self, c1, c2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 第一层卷积
        x = self.relu(self.conv2(x))  # 第二层卷积
        return x

# 构建由 10 个卷积块组成的网络
class ConvNet(nn.Module):
    def __init__(self, c1, c2, num_blocks=10):
        super(ConvNet, self).__init__()
        self.blocks = nn.Sequential(*[ConvBlock(c1, c2) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

# 配置参数
input_size = (56, 56)  # 输入图片的大小
batch_sizes = [1, 2, 4]  # 不同的 batch 大小
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用 GPU 还是 CPU

# 四种通道配置 (c1:c2) 比例
channel_configs = {
    "1:1": (128, 128),
    "1:2": (90, 180),
    "1:6": (52, 312),
    "1:12": (36, 432)
}

# 逐一测试每种通道配置和不同 batch 大小的性能
for ratio, (c1, c2) in channel_configs.items():
    print(f"Testing 3x3 conv lsyers ratio {ratio} with channels ({c1}, {c2})")
    model = ConvNet(c1, c2).to(device)
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, c1, input_size[0], input_size[1])
        images_per_sec = benchmark(model, input_tensor, batch_size, device=device)
        print(f"Batch size {batch_size}, Images/sec: {images_per_sec:.2f}")