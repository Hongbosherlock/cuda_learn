## 项目概述

本项目是一个 **CUDA 算子学习库**，旨在通过实现常见的深度学习算子来学习 CUDA 编程和 GPU 优化技术。项目包含多种核心算子的高性能 CUDA 实现，并提供了与 PyTorch 的无缝集成。

## 算子列表

### ✅ 已实现

- **Reduction 算子**
  - `reduce_sum`: 张量求和，支持多种规约策略

- **Quantization 算子**
  - `per_token_quant_fp8`: Per-token FP8 量化（支持 float16/bfloat16 → FP8 E4M3）

### 🚧 规划中

- **Matrix Multiplication (GEMM)**
  - 标准矩阵乘法
  - Tensor Core 优化版本
  - INT8/FP16 混合精度 GEMM

- **Normalization 算子**
  - LayerNorm
  - RMSNorm
  - GroupNorm

- **Activation 算子**
  - Softmax
  - GELU
  - SiLU/Swish

## 项目结构

```
cuda_learn/
├── kernel/               # CUDA kernel 实现
│   ├── reduce/          # Reduction 算子
│   │   └── reduce_sum.cu
│   ├── quant/           # 量化算子
│   │   └── per_token_quant_fp8.cu
│   ├── matmul/          # 矩阵乘法（规划中）
│   ├── norm/            # 归一化算子（规划中）
│   └── activation/      # 激活函数（规划中）
├── test/                # 测试脚本
│   ├── test_reduce_sum.py
│   └── test_per_token_quant_fp8.py
├── pyblind.cpp          # PyBind11 绑定代码
├── setup.py             # 构建配置
└── README.md            # 本文档
```

## 特性

- ✅ **PyTorch 原生集成**：直接支持 `torch.Tensor`，无需手动管理 GPU 内存
- ✅ **多精度支持**：支持 FP32、FP16、BF16、FP8 等多种数据类型
- ✅ **高性能优化**：
  - 共享内存优化
  - Warp-level primitives
  - 向量化内存访问
  - Occupancy 优化
- ✅ **完善的测试**：每个算子都配备精度测试和性能基准测试
- ✅ **易于扩展**：清晰的代码结构，方便添加新算子

## 环境要求

### 硬件要求
- NVIDIA GPU (Compute Capability >= 7.0)
- 推荐：Ampere (SM 80+) 或 Hopper (SM 90+) 架构

### 软件要求
- **CUDA Toolkit** >= 11.0
- **Python** >= 3.8
- **PyTorch** >= 2.0 (with CUDA support)
- **C++ 编译器** (gcc >= 7.0 or clang)

## 安装步骤

### 1. 克隆仓库

```bash
git clone <repository_url>
cd cuda_learn
```

### 2. 安装 Python 依赖

```bash
pip install torch numpy
```

### 3. 编译安装

#### 标准安装
```bash
python setup.py install
```

#### 开发模式（推荐用于学习和调试）
```bash
python setup.py develop
```

#### 清理构建文件
```bash
python setup.py clean --all
```

## 使用示例

### Reduce Sum

```python
import torch
import cuda_reduce

# 创建输入张量
x = torch.randn(1024, 2048, dtype=torch.float32, device='cuda')

# 调用 CUDA kernel
result = cuda_reduce.reduce_sum(x)

print(f"Sum: {result.item()}")
```

### Per-Token FP8 Quantization

```python
import torch
import cuda_reduce

# 输入: [num_tokens, hidden_dim]
x = torch.randn(512, 4096, dtype=torch.float16, device='cuda')

# 量化到 FP8
output, scale = cuda_reduce.per_token_quant_fp8(x)

print(f"Output dtype: {output.dtype}")  # torch.float8_e4m3fn
print(f"Scale shape: {scale.shape}")    # [512]
```

## 运行测试

### 测试单个算子

```bash
# 测试 reduce_sum
python test/test_reduce_sum.py

# 测试 per_token_quant_fp8
python test/test_per_token_quant_fp8.py
```

### 性能分析

```bash
# 使用 NSight Compute 进行性能分析
ncu --set full -o profile_output python test/test_per_token_quant_fp8.py
```

## 性能优化建议

当前实现可进一步优化：

1. **Warp-level优化**：使用 `__shfl_down_sync` 减少共享内存
2. **向量化加载**：使用 `float4` 提升带宽利用率
3. **流水线**：重叠计算与数据传输
4. **融合kernel**：减少kernel启动开销

示例：Warp shuffle优化
```cuda
template <typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

## 扩展方向

- [ ] 支持多种规约操作（max, min, prod）
- [ ] 支持多维tensor规约
- [ ] 实现Warp shuffle优化
- [ ] 添加FP8支持
- [ ] 支持稀疏tensor

## 参考资料

- [CUDA Programming Guide - Reduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#reduction)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Optimizing Parallel Reduction in CUDA (Mark Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

## License

MIT License