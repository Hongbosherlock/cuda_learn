
/**
 * RMSNorm (Root Mean Square Layer Normalization) Implementation
 * 
 * 主要特性：
 * 1. 对最后一个维度 (hidden_size) 进行归一化处理，计算输入张量在最后一个维度上的均方根(RMS)
    张量想象成一个表格：

    行数 = batch_size × seq_len = 所有的token位置
    列数 = hidden_size = 每个token的特征数
    LayerNorm就是：对每一行单独做归一化，行与行之间互不影响

 * 2. 保持输入输出shape完全一致，不改变张量的维度结构
 * 3. 稳定训练和推理过程，通过归一化减少内部协变量偏移
 * 4. 在残差连接之前进行归一化 (Pre-Norm架构)，更适合深层网络训练
 * 
 * 数学公式：
 *   E(input^2) = sum(x^2) / N  (N为最后一个维度的元素数量)
 *   output = gamma * input / sqrt(E(input^2) + eps) + beta
 *   其中gamma和beta是可学习的缩放和平移参数
 * 
 * 实现特点：
 * - 使用CUDA并行计算加速归一化过程
 * - 支持动态的epsilon值(默认1e-6)防止除零错误
 * - 适用于Transformer等需要稳定归一化的深度学习模型
 */
 
//E(input^2) = x^2 / N
// output = gamma * input / sqrt(sum(x^2) / N + eps) + beta
#include "rmsnorm.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__inline__ __device__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

// Inputs:
// input: input tensor, shape (batch_size, hidden_size)
// weight: weight vector, shape (hidden_size,)
// eps: small epsilon to prevent division by zero, default 1e-06
// Output:
// output: output tensor, shape (batch_size, hidden_size)
__global__ void RMSnormkernel(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output, const uint32_t batch_size, const uint32_t d,
                      float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float sum = 0.0f;
    float* input_row = input +row*d;
    float* out_row = output + row*d;
    for (int i=tid;i<d;i+=blockDim.x){
        sum += input_row[i] * input_row[i];
    }
    sum = warp_reduce_sum(sum);
    if (lane_id == 0){
        sdata[warp_id] = sum;
    }
    __syncthreads();

    const int num_warps = blockDim.x >> 5;
    float block_sum = (tid < num_warps) ? sdata[tid] : 0;
    block_sum = warp_reduce_sum(block_sum);

    __shared__ float rms;
    if (tid == 0){
        rms = sqrt(block_sum / d + eps);
    }
    __syncthreads();  // Critical: ensure all threads read the updated rms value
    
    for(int i=tid;i<d;i+=blockDim.x){
        out_row[i] = weight[i] * input_row[i] / rms;
    }
}


// input, output are device pointers
void RMSNorm(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, int batch_size, int d,
                      float eps) 
{
    int threadsPerBlock = 256;
    int blocksPerGrid = batch_size;
    int num_warps = threadsPerBlock / 32;
    int shared_mem_size = sizeof(float) * num_warps; // Each warp needs one float for partial sum
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    RMSnormkernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size, stream>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, d, eps);
}