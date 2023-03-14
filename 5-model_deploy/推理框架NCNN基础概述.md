## 依赖库速学

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

## 参考资料

1. [Git submodule使用指南（一）](https://juejin.cn/post/6844903812524670984)