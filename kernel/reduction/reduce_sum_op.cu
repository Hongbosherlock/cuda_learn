#include "reduce_sum.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Warp-level reduction using shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized reduction kernel
__global__ void reduce_optimized(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int wid = tid / 32;
    
    // Each thread accumulates multiple elements
    float sum = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        sum += input[i];
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
            output[blockIdx.x] = sum;
        }
    }
}

// Original naive kernel for comparison
__global__ void reduce_naive(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Each thread accumulates multiple elements into a local variable
    float sum = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        sum += input[i];
    }
    
    // Write local sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction with optimization
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Warp unrolling - last 32 elements
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}

void reduce_sum_cuda(
    torch::Tensor& input,
    torch::Tensor& output,
    int block_size)
{
    const int n = input.numel();
    const int num_blocks = 1;
    
    const int shared_mem_size = block_size * sizeof(float);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Use optimized kernel
    reduce_optimized<<<num_blocks, block_size, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n);
}
