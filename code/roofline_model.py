import numpy as np
import matplotlib.pyplot as plt

def compute_flops_and_memory_access(h, b, s, o, n, V):
    """
    计算 LLaMA-13B 模型的计算量 FLOPs 和显存访问量
    """
    memory_access = 1.2 * 12 * n * h * h + 4 * n * b * h * (s + o)
    flops = n * (24 * b * s * h * h + 4 * b * s * s * h) + 2 * b * s * h * V

    return flops, memory_access

def compute_operational_intensity(flops, memory_access):
    """
    计算操作强度
    """
    return flops / memory_access

def plot_roofline_analysis(x, y, X, Y, BW_PERF):
    """
    绘制 A100 Roofline 图
    """
    # plt.figure(figsize=(10, 6))

    # # 绘制分段函数
    # plt.plot(x[x <= BW_PERF], y[x <= BW_PERF], color='red', label='BW Limit')
    # plt.plot(x[x > BW_PERF], y[x > BW_PERF], color='green', label='Compute Limit')
    
    # # 绘制 LLaMA-13B 性能点
    # plt.plot(X, Y, 'ro', label='LLaMA-13B Model Performance')

    # plt.title('A100 Roofline Analysis')
    # plt.xlabel('Memory Access (bytes)', color='#1C2833')
    # plt.ylabel('FLOPS', color='#1C2833')

    # plt.semilogx()
    # plt.semilogy()
    # plt.legend(loc='upper left')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    # plt.tight_layout()
    # plt.show()

def main():
    # A100-SXM GPU 硬件参数
    GPU_FP16_TFLOPS = 312 * 10**12
    GPU_HBM_BW = 2.039 * 10**12
    GPU_FP16_SLOPE = GPU_FP16_TFLOPS / GPU_HBM_BW  # A100 TFP16 操作强度
    BW_PERF = GPU_FP16_TFLOPS / GPU_FP16_SLOPE # bandwidth_threshold

    # LLaMA-13B 模型参数
    h = 5120    # 隐藏层维度
    b = 8       # 推理时的批大小
    s = 2048    # 输入序列长度
    o = 1024    # 输出序列长度
    n = 40      # Transformer 层数
    V = 32000   # 词表大小

    # 计算 FLOPs 和显存访问量
    total_flops, total_memory_access = compute_flops_and_memory_access(h, b, s, o, n, V)
    llama_oi = compute_operational_intensity(total_flops, total_memory_access)

    print(f"Total FLOPs for LLaMA-13B: {total_flops / 10**12:.2f} TFLOPs")
    print(f"Total memory access: {total_memory_access / 10**9:.2f} GB")
    print(f"Llama-13b Operational Intensity (OI): {llama_oi:.2f} FLOPs/Byte")
    print(f"A100 Operational Intensity (OI): {GPU_FP16_SLOPE:.2f} FLOPs/Byte")

    # x = np.linspace(10, 20e12, 1000)
    # y = np.piecewise(x, [x <= BW_PERF, x > BW_PERF], [lambda x: GPU_FP16_SLOPE * x, GPU_FP16_TFLOPS])
    # 绘制图形
    # plot_roofline_analysis(x, y, [total_memory_access, total_memory_access*200], [total_flops, total_flops], BW_PERF)

if __name__ == "__main__":
    main()

