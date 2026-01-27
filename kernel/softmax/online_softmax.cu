#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cfloat>
#include <torch/extension.h>

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// -----------------------------------------------------------
// 辅助结构体：存储当前的 Max 和 Sum
// -----------------------------------------------------------
struct __align__(8) MaxSum {
    float max_val;
    float sum_val;
};

// -----------------------------------------------------------
// Device 函数：合并两个 MaxSum 状态 (Online Softmax 核心逻辑)
// -----------------------------------------------------------
__device__ __forceinline__ MaxSum combine_max_sum(MaxSum a, MaxSum b) {
    MaxSum res;
    res.max_val = fmaxf(a.max_val, b.max_val);
    
    // 如果一个是 -inf，处理这种边界情况
    if (res.max_val == -FLT_MAX) {
        res.sum_val = 0.0f;
    } else {
        // Online Softmax 更新公式
        // sum_new = sum_a * exp(m_a - m_new) + sum_b * exp(m_b - m_new)
        float factor_a = expf(a.max_val - res.max_val);
        float factor_b = expf(b.max_val - res.max_val);
        res.sum_val = a.sum_val * factor_a + b.sum_val * factor_b;
    }
    return res;
}

// -----------------------------------------------------------
// Warp 级规约：使用 Shuffle 指令
// -----------------------------------------------------------
__device__ __forceinline__ MaxSum warp_reduce_max_sum(MaxSum val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        MaxSum other;
        other.max_val = __shfl_down_sync(0xffffffff, val.max_val, offset);
        other.sum_val = __shfl_down_sync(0xffffffff, val.sum_val, offset);
        val = combine_max_sum(val, other);
    }
    return val;
}

// -----------------------------------------------------------
// Online Softmax Kernel
// 假设：gridDim.x = batch_size (行数), blockDim.x = 256 或 128
// 每个 Block 处理输入矩阵的一行
// -----------------------------------------------------------
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    // 共享内存，用于存储每个 Warp 的规约结果
    // 假设最大 block 大小为 1024，warp 数量最多 32
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int tid = threadIdx.x;
    int row_idx = blockIdx.x; // 当前处理的行号
    
    // 定位到当前行的起始位置
    const float* row_input = input + row_idx * N;
    float* row_output = output + row_idx * N;

    // 1. 线程局部规约 (Thread-local reduction)
    // 每个线程处理多个元素 (如果 N > blockDim.x)
    MaxSum local_state = {-FLT_MAX, 0.0f}; // 初始化为极小值和0

    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_input[i];
        MaxSum next_elem = {val, 1.0f}; // 单个元素本身 max=val, sum=exp(val-val)=1
        local_state = combine_max_sum(local_state, next_elem);
    }

    // 2. Warp 内部规约
    local_state = warp_reduce_max_sum(local_state);

    // 3. 将每个 Warp 的结果写入共享内存
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    
    if (lane == 0) {
        shared_max[warp_id] = local_state.max_val;
        shared_sum[warp_id] = local_state.sum_val;
    }
    __syncthreads();

    // 4. Block 级规约 (由第一个 Warp 处理共享内存中的数据)
    // 这里简单地让第一个 Warp 读取所有 Warp 的结果并再次规约
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        
        // 只有前 num_warps 个线程需要加载数据
        MaxSum warp_state = {-FLT_MAX, 0.0f};
        if (lane < num_warps) {
            warp_state.max_val = shared_max[lane];
            warp_state.sum_val = shared_sum[lane];
        }
        
        // 再次进行 Warp 规约
        warp_state = warp_reduce_max_sum(warp_state);

        // 此时 lane 0 拥有整个 Block (即这一行) 的全局 Max 和全局 Sum
        if (lane == 0) {
            shared_max[0] = warp_state.max_val;
            shared_sum[0] = warp_state.sum_val;
        }
    }
    __syncthreads();

    // 5. 最终计算输出
    // 读取全局 Max 和 Sum
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];

    // 再次遍历写入结果： exp(x - global_max) / global_sum
    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - global_max) / global_sum;
    }
}


// ============================================================================
// Host Interface (参考 online_softmax.cu.bk 风格)
// ============================================================================

torch::Tensor online_softmax_cuda(torch::Tensor input) {
    /*
    Online Softmax for 2D tensor
    
    Args:
        input: (num_rows, num_cols) tensor, float32
    
    Returns:
        output: softmax over last dimension
    */
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float, 
                "Only float32 is supported");
    
    auto output = torch::empty_like(input);
    
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    
    const int threads = 256;
    const int blocks = num_rows;
    
    // Launch kernel - each block handles one row
    online_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_cols
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
                "Kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}


// // Python binding
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("online_softmax", &online_softmax_cuda, 
//           "Online Softmax forward (CUDA)");
// }
