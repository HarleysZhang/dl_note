## 一，依赖库知识速学

### aarch64

`aarch64`，也被称为 ARM64，是一种基于 `ARMv8-A` 架构的 `64` 位指令集体系结构。它是 ARM 体系结构的最新版本，旨在提供更好的性能和能效比。与先前的 `32` 位 `ARM` 架构相比，aarch64 具有更大的寻址空间、更多的寄存器和更好的浮点性能。

在 Linux 系统终端下输入以下命令，查看 `cpu` 架构。

```bash
uname -m # 我的英特尔服务器输出 x86_64，m1 pro 苹果电脑输出 arm64
```
### OpenMP

`OpenMP`（Open Multi-Processing）是一种基于共享内存的并行编程 API，用于编写多线程并行程序。使用 `OpenMP`，程序员可以通过在程序中**插入指令**来指示程序中的并行性。这些指令是以 `#pragma` 开头的编译指示符，告诉编译器如何并行化代码。

```cpp
#include <stdio.h>
#include <omp.h>

int main() {
    int i;
    #pragma omp parallel for
    for(i = 0; i < 10; i++) {
        printf("Thread %d executing iteration %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
```

### AVX512

`AVX` 全称是 Advanced Vector Extension，高级矢量扩展，用于处理 `N` 维数据的，例如 `8` 维及以下的 `64` 位双精度浮点矢量或 `16` 维及以下的单精度浮点矢量。

`AVX512` 是 `SIMD` 指令（单指令多数据），`x86` 架构上最早的 SIMD 指令是 128bit 的 `SSE`，然后是 256bit 的 AVX/AVX2，最后是现在 512bit 的 AVX512。

### submodule

github submodule（子模块）允许你将一个 Git 仓库作为另一个 Git 仓库的子目录。 它能让你将另一个仓库克隆到自己的项目中，同时还保持提交的独立。

### apt upgrade

- `apt update`：只检查，不更新（已安装的软件包是否有可用的更新，给出汇总报告）。
- `apt upgrade`：更新已安装的软件包。

## 二，硬件基础知识速学
### 2.1，内存硬件关键特性

`RAM`（随机访问存储）的一些关键特性是带宽(`bandwidth`)和延迟(`latency`)。

### 2.2，CPU

中央处理器(central processing unit，`CPU`)是任何计算机的核心，其由许多关键组件组成:
- **处理器核心** (processor cores): 用于执行机器代码的。
- **总线**（bus）: 用于连接不同组件(注意，总线会因为处理器型号、 各代产品和供应商之间的特定拓扑结构有明显不同)
- **缓存**(cache): 一般是三级缓（L1/L2/L3 cache），相比主内存实现更高的读取带宽和更低的延迟内存访问。

现代 CPU 都包含向量处理单元，都提供了 `SIMD` 指令，可以在单个指令中同时处理多个数据，从而支持高性能线性代数和卷积运算。这些 `SIMD` 指令有不同的名称: 在 ARM 上叫做 NEON，在 x86 上被称 为AVX2156。

一个典型的 Intel Skylake 消费级四核 CPU，其核心架构如下图所示。

![cpu 核心架构](../images/ncnn/cpu_architecture.png)

## 参考资料

1. [Git submodule使用指南（一）](https://juejin.cn/post/6844903812524670984)