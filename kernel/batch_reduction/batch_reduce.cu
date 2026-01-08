#include "batch_reduce.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Warp-level reduction using shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized version: each block processes one row with warp shuffle
__global__ void reduce_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)  // number of columns per row
{
    extern __shared__ float sdata[];
    
    const int row_idx = blockIdx.x;  // which row this block processes
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int wid = tid / 32;
    
    // Each thread accumulates multiple elements from its row
    float sum = 0.0f;
    const float* row_ptr = input + row_idx * n;
    
    for (int i = tid; i < n; i += blockDim.x) {
        sum += row_ptr[i];
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Write warp results to shared memory
    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < (blockDim.x / 32)) {
        sum = sdata[tid];
        sum = warpReduceSum(sum);
        
        if (tid == 0) {
            output[row_idx] = sum;
        }
    }
}

// Naive version: each block processes one row with tree reduction
__global__ void reduce_naive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    extern __shared__ float sdata[];
    
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Each thread accumulates multiple elements
    float sum = 0.0f;
    const float* row_ptr = input + row_idx * n;
    
    for (int i = tid; i < n; i += blockDim.x) {
        sum += row_ptr[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree reduction for s > 32
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp unrolling for last 32 elements
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[row_idx] = sdata[0];
    }
}

void batch_reduce_sum_cuda(
    torch::Tensor& input,   // [m, n]
    torch::Tensor& output,  // [m]
    int block_size,
    bool use_optimized)
{
    const int m = input.size(0);  // number of rows
    const int n = input.size(1);  // number of columns
    
    const int grid_size = m;  // one block per row
    const int shared_mem_size = block_size * sizeof(float);
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (use_optimized) {
        reduce_optimized<<<grid_size, block_size, shared_mem_size, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
    } else {
        reduce_naive<<<grid_size, block_size, shared_mem_size, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
    }
}