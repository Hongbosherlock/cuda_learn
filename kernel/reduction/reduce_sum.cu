#include "reduce_sum.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void reduce_naive(float *input, float *output, int n) {
    // only work for  block_size=1
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread accumulates multiple elements into a local variable
    float sum = 0.0f; //sum in resgister
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        sum += input[i];
    }
    
    // Write local sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s/=2){
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid ==0){
        output[blockIdx.x] = sdata[0];
    }
}


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
    
    // Vectorized load with alignment check
    // const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(input);
    // const bool is_aligned = (ptr_val % 16 == 0);
    // is_aligned && 
    if (n >= 4) {
        // Use float4 for vectorized load (4x bandwidth)
        const float4* in4 = reinterpret_cast<const float4*>(input);
        const int n4 = n / 4;
        
        for (int i = tid; i < n4; i += blockDim.x) {
            float4 v = __ldg(&in4[i]);  // Read-only cache
            sum += v.x + v.y + v.z + v.w;
        }
        
        // Handle tail elements
        const int remainder_start = n4 * 4;
        for (int i = remainder_start + tid; i < n; i += blockDim.x) {
            sum += input[i];
        }
    } else {
        // Fallback for unaligned or small n
        for (int i = tid; i < n; i += blockDim.x) {
            sum += input[i];
        }
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


void reduce_sum_cuda(
    torch::Tensor& input,   // [n]
    torch::Tensor& output,  // [1]
    int block_size)
{
    const int n = input.numel();
    const int num_blocks = 1;
    
    // Create temp tensor to store each block's result
    // auto options = torch::TensorOptions()
    //                    .dtype(input.dtype())
    //                    .device(input.device());
    // torch::Tensor temp = torch::empty({num_blocks}, options);
    
    const int shared_mem_size = block_size * sizeof(float);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // First round: each block processes part of input
    reduce_optimized<<<num_blocks, block_size, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n);
    
    // // Second round: if multiple blocks, reduce again
    // if (num_blocks > 1) {
    //     reduce_naive<<<1, block_size, shared_mem_size, stream>>>(
    //         temp.data_ptr<float>(),
    //         output.data_ptr<float>(),
    //         num_blocks);
    // } else {
    //     output.copy_(temp);
    // }
}