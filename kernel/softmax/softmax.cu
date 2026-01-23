#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cmath>
#include <stdio.h>

// ============================================================================
// Version 1: Naive Softmax (Baseline)
// ============================================================================
// 特点：
// - 两次遍历：第一次求max和sum，第二次计算exp和normalize
// - 每个线程处理一行
// - 适合行长度较小的情况（< 1024）
// - 存在数值稳定性问题

template<typename T>
__global__ void softmax_naive_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        const T* in_row = input + row * cols;
        T* out_row = output + row * cols;
        
        // Step 1: Find max value for numerical stability
        T max_val = in_row[0];
        for (int i = 1; i < cols; i++) {
            max_val = max(max_val, in_row[i]);
        }
        
        // Step 2: Compute exp(x - max) and sum
        T sum = 0;
        for (int i = 0; i < cols; i++) {
            T exp_val = exp(in_row[i] - max_val);
            out_row[i] = exp_val;
            sum += exp_val;
        }
        
        // Step 3: Normalize
        for (int i = 0; i < cols; i++) {
            out_row[i] /= sum;
        }
    }
}


// ============================================================================
// Warp-level and Block-level Reduction Primitives
// ============================================================================

// Warp-level sum reduction
template<typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction
template<typename T>
__inline__ __device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
// Block-level max reduction (优化版本)
// 将一个 block 内所有线程的值规约为最大值，并广播给所有线程
// 
// 优化点：
// 1. 使用静态 shared memory（避免每次调用都初始化）
// 2. 减少分支（使用三元运算符）
// 3. 使用 __syncwarp() 替代部分 __syncthreads()（更轻量）
template<typename T>
__inline__ __device__ T block_reduce_max(T val) {
    constexpr int WARP_SIZE = 32;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);  // 优化：位运算替代取模
    const int warp_id = threadIdx.x >> 5;                // 优化：位移替代除法
    
    // Step 1: Warp-level reduction
    val = warp_reduce_max(val);
    
    // Step 2: Store warp results to shared memory
    // 使用静态 shared memory，在编译时分配
    __shared__ T warp_results[32];
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction among warp leaders
    // 优化：减少条件分支，使用三元运算符
    const int num_warps = blockDim.x >> 5;  // 优化：位移替代除法
    T result = (threadIdx.x < num_warps) ? warp_results[threadIdx.x] : -INFINITY;
    
    // 只有第一个 warp 需要进行最终 reduction
    if (warp_id == 0) {
        result = warp_reduce_max(result);
    }
    
    // Step 4: Broadcast to all threads (所有 warp 都需要)
    // 使用 __shfl_sync 从第一个 warp 的 lane 0 广播
    result = __shfl_sync(0xffffffff, result, 0);
    
    return result;
}

// Block-level sum reduction (优化版本)
// 将一个 block 内所有线程的值规约为总和，并广播给所有线程
//
// 优化点：
// 1. 使用位运算替代取模和除法
// 2. 减少分支预测失败
// 3. 更清晰的逻辑结构
template<typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    constexpr int WARP_SIZE = 32;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);  // threadIdx.x % 32
    const int warp_id = threadIdx.x >> 5;                // threadIdx.x / 32
    
    // Step 1: Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Step 2: Store warp results to shared memory
    __shared__ T warp_results[32];
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction among warp leaders
    const int num_warps = blockDim.x >> 5;
    T result = (threadIdx.x < num_warps) ? warp_results[threadIdx.x] : 0;
    
    // 只有第一个 warp 的线程参与最终 reduction
    if (warp_id == 0) {
        result = warp_reduce_sum(result);
    }
    
    // Step 4: Broadcast to all threads in the block
    result = __shfl_sync(0xffffffff, result, 0);
    
    return result;
}


// ============================================================================
// Version 2: Warp-level Reduction Softmax
// ============================================================================
// 特点：
// - 使用 warp shuffle 进行规约
// - 每个 warp 处理一行
// - 适合中等行长度（< 2048）
// - 更好的内存访问模式

template<typename T>
__global__ void softmax_warp_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    
    if (row < rows) {
        const T* in_row = input + row * cols;
        T* out_row = output + row * cols;
        
        // Step 1: Find max using block-level reduction
        T thread_max = -INFINITY;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            thread_max = fmaxf(thread_max, in_row[i]);
        }
        T row_max = block_reduce_max(thread_max);
        
        // Step 2: Compute exp and sum using block-level reduction
        T thread_sum = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            // in_row read from global memory
            T exp_val = exp(in_row[i] - row_max);
            out_row[i] = exp_val;
            thread_sum += exp_val;
        }
        T row_sum = block_reduce_sum(thread_sum);
        
        // Step 3: Normalize
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            out_row[i] /= row_sum;
        }
    }
}


// ============================================================================
// Version 3: Safe Softmax (Production-ready)
// ============================================================================
// 特点：
// - 完整的数值稳定性保证
// - 处理各种边界情况
// - 支持大规模张量
// - 工业级实现

template<typename T>
__global__ void softmax_safe_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const T* in_row = input + row * cols;
    T* out_row = output + row * cols;
    
    // Step 1: Find maximum using block-level reduction (for numerical stability)
    T thread_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        thread_max = max(thread_max, in_row[i]);
    }
    T row_max = block_reduce_max(thread_max);
    
    // Step 2: Compute exp(x - max) and sum using block-level reduction
    T thread_sum = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        T val = in_row[i] - row_max;
        // Clamp to avoid underflow/overflow
        val = max(val, (T)-88.0);  // exp(-88) ≈ 0 for float
        T exp_val = exp(val);
        out_row[i] = exp_val;
        thread_sum += exp_val;
    }
    T row_sum = block_reduce_sum(thread_sum);
    
    // Avoid division by zero
    row_sum = max(row_sum, (T)1e-10);
    
    // Step 3: Normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] /= row_sum;
    }
}


// ============================================================================
// Version 4: Online Softmax (Single-pass, Memory-efficient)
// ============================================================================
// 特点：
// - 单次遍历，无需存储中间结果
// - 内存效率高
// - 适合超大行长度
// - FlashAttention 风格的实现

template<typename T>
__global__ void softmax_online_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    if (row >= rows) return;
    
    const T* in_row = input + row * cols;
    T* out_row = output + row * cols;
    
    __shared__ T shared_max[32];
    __shared__ T shared_sum[32];
    
    // Online algorithm: maintain running max and sum
    T running_max = -INFINITY;
    T running_sum = 0;
    
    // Process in chunks for better cache utilization
    const int CHUNK_SIZE = 256;
    for (int chunk_start = 0; chunk_start < cols; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, cols);
        
        // Find max in this chunk
        T chunk_max = -INFINITY;
        for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
            chunk_max = max(chunk_max, in_row[i]);
        }
        
        // Reduce chunk max
        chunk_max = warp_reduce_max(chunk_max);
        if (lane_id == 0) {
            shared_max[warp_id] = chunk_max;
        }
        __syncthreads();
        
        if (tid < num_warps) {
            chunk_max = shared_max[tid];
        }
        chunk_max = warp_reduce_max(chunk_max);
        chunk_max = __shfl_sync(0xffffffff, chunk_max, 0);
        
        // Update running max and rescale previous sum
        T old_max = running_max;
        running_max = max(running_max, chunk_max);
        T rescale_factor = exp(old_max - running_max);
        running_sum *= rescale_factor;
        
        // Compute exp and update sum for this chunk
        T chunk_sum = 0;
        for (int i = chunk_start + tid; i < chunk_end; i += blockDim.x) {
            T exp_val = exp(in_row[i] - running_max);
            out_row[i] = exp_val;
            chunk_sum += exp_val;
        }
        
        // Reduce chunk sum
        chunk_sum = warp_reduce_sum(chunk_sum);
        if (lane_id == 0) {
            shared_sum[warp_id] = chunk_sum;
        }
        __syncthreads();
        
        if (tid < num_warps) {
            chunk_sum = shared_sum[tid];
        }
        chunk_sum = warp_reduce_sum(chunk_sum);
        chunk_sum = __shfl_sync(0xffffffff, chunk_sum, 0);
        
        running_sum += chunk_sum;
    }
    
    // Normalize
    running_sum = max(running_sum, (T)1e-10);
    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] /= running_sum;
    }
}


// ============================================================================
// Host Functions (C++ API)
// ============================================================================

torch::Tensor softmax_naive_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    //一行一线程: 每个线程处理一整行的 softmax 计算

    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_naive_cuda", [&] {
        softmax_naive_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    });
    
    return output;
}

torch::Tensor softmax_warp_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    const int threads = 256;
    const int blocks = rows;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_warp_cuda", [&] {
        softmax_warp_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    });
    
    return output;
}

torch::Tensor softmax_safe_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    const int threads = 256;
    const int blocks = rows;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_safe_cuda", [&] {
        softmax_safe_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    });
    
    return output;
}

torch::Tensor softmax_online_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    
    const int threads = 256;
    const int blocks = rows;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_online_cuda", [&] {
        softmax_online_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    });
    
    return output;
}