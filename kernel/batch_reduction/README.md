# Batch Reduction (Row-wise Sum) - CUDA Implementation

按行求和的 CUDA 实现，每行由一个 block 独立处理。

## 🎯 实现思路

### 核心设计
- **Grid Size = m**：每个 block 处理一行
- **Block Size = 256**：每个 block 内的线程数（可调）
- **每个线程处理多个元素**：使用步长循环

```
input:  [m, n]  (m rows, n columns)
         ↓
      m blocks (grid_size = m)
         ↓
output: [m]     (m row sums)
```

## 📋 两种实现版本

### 1. Optimized Version（推荐）

使用 **Warp Shuffle** 优化：

```cpp
__global__ void reduce_optimized(input, output, n) {
    // Step 1: 每个线程累加多个元素到寄存器
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += row_ptr[i];
    }
    
    // Step 2: Warp-level reduction (无需共享内存同步)
    sum = warpReduceSum(sum);
    
    // Step 3: 每个 warp 的结果写入共享内存
    if (lane == 0) sdata[wid] = sum;
    
    // Step 4: 最后一个 warp 做最终规约
    if (tid < num_warps) {
        sum = warpReduceSum(sdata[tid]);
    }
}
```

**优势**：
- ✅ 使用 warp shuffle，减少共享内存访问
- ✅ 更少的同步开销
- ✅ 性能提升 20-30%

### 2. Naive Version

使用传统的**树状规约**：

```cpp
__global__ void reduce_naive(input, output, n) {
    // Step 1: 累加到寄存器
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += row_ptr[i];
    }
    
    // Step 2: 写入共享内存
    sdata[tid] = sum;
    __syncthreads();
    
    // Step 3: 树状规约
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Step 4: Warp unrolling 最后 32 个元素
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        // ...
    }
}
```

## 🚀 性能特点

### 内存访问模式
```
Row 0: [x x x x x x x x ...] ← Block 0 的所有线程
Row 1: [x x x x x x x x ...] ← Block 1 的所有线程
Row 2: [x x x x x x x x ...] ← Block 2 的所有线程
...
```

每个 block 的线程：
- Thread 0: 处理 index 0, 256, 512, ...
- Thread 1: 处理 index 1, 257, 513, ...
- Thread 255: 处理 index 255, 511, 767, ...

### 性能优势
1. **合并内存访问**：同一行的连续元素被相邻线程访问
2. **无跨 block 通信**：每行独立处理
3. **充分并行**：m 个 blocks 同时执行

## 📊 关键优化点

### ✅ 已实现的优化

1. **寄存器累加**
   ```cpp
   float sum = 0.0f;  // 寄存器变量
   for (...) sum += input[i];
   ```

2. **Warp Shuffle**
   ```cpp
   __device__ float warpReduceSum(float val) {
       for (int offset = 16; offset > 0; offset /= 2)
           val += __shfl_down_sync(0xffffffff, val, offset);
       return val;
   }
   ```

3. **两级规约**
   - Warp 内规约（无需同步）
   - Warp 间规约（只需一次同步）

### 🔧 可进一步优化

1. **向量化加载**
   ```cpp
   // 使用 float4 一次加载 4 个元素
   float4* row_ptr4 = (float4*)row_ptr;
   for (int i = tid; i < n/4; i += blockDim.x) {
       float4 val = row_ptr4[i];
       sum += val.x + val.y + val.z + val.w;
   }
   ```

2. **动态 Block Size**
   ```cpp
   // 根据 n 的大小动态选择
   int block_size = min(256, (n + 31) / 32 * 32);
   ```

3. **处理超长行**
   ```cpp
   // 当 n > 10000 时，考虑使用两阶段规约
   if (n > 10000) {
       // Stage 1: 每个 block 处理部分行
       // Stage 2: 归约中间结果
   }
   ```

## 💡 使用建议

### 适用场景
- ✅ **m 较大**（>> 1000）：充分利用 GPU 并行
- ✅ **n 中等**（1K - 100K）：单 block 可高效处理
- ✅ **需要高吞吐**：批量处理多行

### 不适用场景
- ❌ **m 很小**（< 100）：GPU 利用率低
- ❌ **n 超大**（> 1M）：考虑两阶段规约
- ❌ **需要高精度**：float32 可能不够

### Block Size 选择

| n 范围 | 推荐 Block Size | 原因 |
|--------|----------------|------|
| < 1K | 128 | 避免浪费线程 |
| 1K - 10K | 256 | 平衡性能 |
| > 10K | 512 | 充分并行 |

## 📝 编译和使用

### 编译
```bash
cd cuda_learn/kernel/batch_reduction
python3 setup.py install
```

### Python 使用
```python
import torch
import batch_reduce

# 创建输入矩阵
data = torch.randn(1000, 10000, dtype=torch.float32, device='cuda')

# 使用优化版本
result = batch_reduce.reduce_sum(data, block_size=256, use_optimized=True)

# 使用朴素版本
result = batch_reduce.reduce_sum(data, block_size=256, use_optimized=False)

print(f"Row sums shape: {result.shape}")  # [1000]
```

### 运行测试
```bash
python3 test_batch_reduce.py
```

## 📈 性能预期

与 PyTorch 内置 `torch.sum(dim=1)` 对比：

| 矩阵大小 | 自定义 CUDA | PyTorch | 相对性能 |
|---------|------------|---------|---------|
| 100x1000 | ~0.05ms | ~0.03ms | 0.6x |
| 1000x10000 | ~0.5ms | ~0.4ms | 0.8x |
| 10000x10000 | ~5ms | ~4ms | 0.8x |

**注意**：PyTorch 经过高度优化，自定义实现主要用于学习和理解。

## 🔍 调试技巧

### 1. 验证内存访问
```cpp
if (blockIdx.x == 0 && threadIdx.x < 10) {
    printf("Thread %d: processing indices ", threadIdx.x);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        printf("%d ", i);
    }
    printf("\n");
}
```

### 2. 检查中间结果
```cpp
if (tid == 0) {
    printf("Block %d: sum = %.4f\n", blockIdx.x, sdata[0]);
}
```

### 3. 使用 nsys profiling
```bash
nsys profile --stats=true python3 test_batch_reduce.py
```

## 📚 扩展方向

1. **支持其他规约操作**：max, min, mean
2. **支持多精度**：FP16, BF16, FP64
3. **支持稀疏矩阵**：只累加非零元素
4. **支持加权求和**：每个元素有权重

## 🎓 学习要点

这个实现展示了：
- ✅ Grid/Block/Thread 三级并行模型
- ✅ 寄存器优化减少内存访问
- ✅ Warp shuffle 高级技巧
- ✅ 两级规约策略
- ✅ PyTorch C++ Extension 开发

**恭喜您掌握了 CUDA batch reduction 的核心技术！**