import torch
input = torch.randn([20, 10, 224, 224])
conv3x3 = torch.nn.Conv2d(10, 6, kernel_size=3, groups=5)
print(conv3x3.weight.shape)
output = conv3x3(input)
print(output.shape)