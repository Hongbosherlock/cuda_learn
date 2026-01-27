#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cmath>
#include <stdio.h>


#define WARP_SIZE 32

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__device__ __forceinline__ float to_float(T v) { return (float)v; }

template<>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template<>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }


template<typename T>
__device__ __forceinline__ T from_float(float v) { return (T)v; }

template<>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half_rn(v); }

template<>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }


// input: __global__ float N
// output: __global__ float N (overwritten with exp(x)/sum(exp(x)))
__global__ void softmax_kernel_oneblock(const float* input, float* output, int N) {
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;

    int tid = threadIdx.x;

    // 1) local max
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    smax[tid] = local_max;
    __syncthreads();

    // 2) reduce max in shared memory
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smax[tid] = fmaxf(smax[tid], smax[tid + offset]);
        }
        __syncthreads();
    }

    // 3) broadcast max to all threads
    float xmax = smax[0];
    __syncthreads();

    // 3) local sum of exp(x - xmax), store exp to output to avoid recompute
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = expf(input[i] - xmax);
        output[i] = e;
        local_sum += e;
    }
    ssum[tid] = local_sum;
    __syncthreads();

    // 4) reduce sum
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            ssum[tid] += ssum[tid + offset];
        }
        __syncthreads();
    }
    // 5) broadcast inv_sum to all threads
    float inv_sum = 1.0f / ssum[0];

    // 6) normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] *= inv_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    int threadsPerBlock = 256;           
    int blocksPerGrid = 1;               
    size_t shmem = 2 * threadsPerBlock * sizeof(float);

    softmax_kernel_oneblock<<<blocksPerGrid, threadsPerBlock, shmem>>>(input, output, N);
    cudaDeviceSynchronize();
}




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
// __inline__ __device__ float warpReduceSum(float val) {
//     #pragma unroll
//     for (int offset = 16; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xffffffff, val, offset, 32);
//     }
//     return val;
// }
__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

// Warp-level max reduction
// __inline__ __device__ float warpReduceMax(float val) {
//     #pragma unroll
//     for (int offset = 16; offset > 0; offset /= 2) {
//         val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset, 32));
//     }
//     return val;
// }
__inline__ __device__ float warpReduceMax(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}


// Block-level max reduction  
/* Calculate the maximum of all elements in a block */
// 将一个 block 内所有线程的值规约为最大值，并广播给所有线程
// 
// 优化点：
// 1. 使用静态 shared memory（避免每次调用都初始化）
// 2. 减少分支（使用三元运算符）
__inline__ __device__ float block_reduce_max(float val) {
    
    // 使用静态 shared memory，在编译时分配
    static __shared__ float warp_results[32];
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);  // in-warp idx
    const int warp_id = threadIdx.x >> 5;                // warp idx
    
    // Step 1: Warp-level reduction
    val = warpReduceMax(val);
    
    // Step 2: Store warp results to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction among warp leaders
    // 优化：减少条件分支，使用三元运算符

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? warp_results[lane_id] : -INFINITY;;
    
    // 只有第一个 warp 需要进行最终 reduction
    if (warp_id == 0) {
        val = warpReduceMax(val);
    }
    
    // Step 4: Broadcast to all threads (跨 warp 广播需要用 shared memory)
    // __shfl_sync 只能在 warp 内广播，无法跨 warp
    if (threadIdx.x == 0) {
        warp_results[0] = val;
    }
    __syncthreads();
    
    return warp_results[0];
}

// Block-level sum reduction (优化版本)
// 将一个 block 内所有线程的值规约为总和，并广播给所有线程
//
// 优化点：
// 1. 使用位运算替代取模和除法
// 2. 减少分支预测失败
// 3. 更清晰的逻辑结构
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float warp_results[32];

    const int lane_id = threadIdx.x & (WARP_SIZE - 1);  // in-warp idx
    const int warp_id = threadIdx.x >> 5;                // warp idx
    
    // Step 1: Warp-level reduction
    val = warpReduceSum(val);
    
    // Step 2: Store warp results to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction among warp leaders
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? warp_results[lane_id] : 0;
    
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    
    // Step 4: Broadcast to all threads in the block
    // result = __shfl_sync(0xffffffff, result, 0);
    if(threadIdx.x==0)
        warp_results[0] = val;
    __syncthreads();
    
    return warp_results[0];
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
        float thread_max = -INFINITY;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            thread_max = fmaxf(thread_max, to_float(in_row[i]));
        }
        float row_max = block_reduce_max(thread_max);
        
        // Step 2: Compute exp and sum using block-level reduction
        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            // in_row read from global memory
            float exp_val = expf(to_float(in_row[i]) - row_max);
            // out_row[i] = from_float<T>(exp_val);
            thread_sum += exp_val;
        }
        float row_sum = block_reduce_sum(thread_sum);
        float inv_sum = 1.f / row_sum;

        // Step 3: Normalize
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float x = to_float(in_row[i]);
            float y = __expf(x - row_max) * inv_sum;
            out_row[i] = from_float<T>(y);
            // out_row[i] = to_float(from_float<T>(to_float(in_row[i]) - row_max) * inv_sum);
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
        chunk_max = warpReduceMax(chunk_max);
        if (lane_id == 0) {
            shared_max[warp_id] = chunk_max;
        }
        __syncthreads();
        
        if (tid < num_warps) {
            chunk_max = shared_max[tid];
        }
        chunk_max = warpReduceMax(chunk_max);
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
        chunk_sum = warpReduceSum(chunk_sum);
        if (lane_id == 0) {
            shared_sum[warp_id] = chunk_sum;
        }
        __syncthreads();
        
        if (tid < num_warps) {
            chunk_sum = shared_sum[tid];
        }
        chunk_sum = warpReduceSum(chunk_sum);
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
    
    // Dispatch based on input dtype
    if (input.scalar_type() == at::ScalarType::Float) {
        softmax_warp_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            rows,
            cols
        );
    } else if (input.scalar_type() == at::ScalarType::Half) {
        softmax_warp_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            rows,
            cols
        );
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        softmax_warp_kernel<nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<const nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
            reinterpret_cast<nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
            rows,
            cols
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type for softmax_warp_cuda. Supported types: float32, float16, bfloat16");
    }
    
    return output;
}

// torch::Tensor softmax_safe_cuda(torch::Tensor input) {
//     TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
//     TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
//     auto output = torch::empty_like(input);
//     int rows = input.size(0);
//     int cols = input.size(1);
    
//     const int threads = 256;
//     const int blocks = rows;
    
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_safe_cuda", [&] {
//         softmax_safe_kernel<scalar_t><<<blocks, threads>>>(
//             input.data_ptr<scalar_t>(),
//             output.data_ptr<scalar_t>(),
//             rows,
//             cols
//         );
//     });
    
//     return output;
// }

// torch::Tensor softmax_online_cuda(torch::Tensor input) {
//     TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
//     TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
//     auto output = torch::empty_like(input);
//     int rows = input.size(0);
//     int cols = input.size(1);
    
//     const int threads = 256;
//     const int blocks = rows;
    
//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_online_cuda", [&] {
//         softmax_online_kernel<scalar_t><<<blocks, threads>>>(
//             input.data_ptr<scalar_t>(),
//             output.data_ptr<scalar_t>(),
//             rows,
//             cols
//         );
//     });
    
//     return output;
// }