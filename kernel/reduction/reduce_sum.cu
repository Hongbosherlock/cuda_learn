#include "reduce_sum.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void reduce_naive(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
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
    reduce_naive<<<num_blocks, block_size, shared_mem_size, stream>>>(
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