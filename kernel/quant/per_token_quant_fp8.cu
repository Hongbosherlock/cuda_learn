#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cub/block/block_reduce.cuh>

#define WARP_SIZE 32
// #define FP8_E4M3_MAX 448.0f

// Using c10::Float8_e4m3fn as FP8_TYPE
using FP8_TYPE = c10::Float8_e4m3fn;
// constexpr float FP8_E4M3_MAX = 448.0f;  // Max value for FP8 E4M3 format
// C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();


// ========================== Util functions to convert types ==========================
template <typename T>
__device__ float convert_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else if constexpr (std::is_same_v<T, float>) {
    return x;
  } else {
    return static_cast<float>(x);
  }
}


__device__ __forceinline__ float warpReduceMax(float max_value) {
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
  max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));
  return max_value;
}


template<typename scalar_t>
__global__ void per_token_quant_fp8_kernel(
    const scalar_t* __restrict__ input,  //[M,K]
    FP8_TYPE* __restrict__ output,       //[M,K]
    float* __restrict__ scale,           //[M,]
    const int64_t hidden_dim,
    const int64_t num_tokens) {

    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = block_dim / WARP_SIZE;

    // address calculation
    const scalar_t* __restrict__ input_ptr = input + token_idx * hidden_dim;
    FP8_TYPE* __restrict__ output_ptr = output + token_idx * hidden_dim;
    
    // shared memory for warp-level max values
    extern __shared__ float warpLevelMaxs[];
    
    // Step 1: Find max absolute value for this token (row)
    float thread_max = 0.0f;
    // Each thread handles multiple elements with stride
    for(int i = tid; i < hidden_dim; i += block_dim){
        float value = convert_to_float(input_ptr[i]);  // Convert to float for computation
        thread_max = fmaxf(thread_max, fabsf(value));
    }
    
    // Step 2: Warp-level reduction
    float warp_max = warpReduceMax(thread_max);
    
    // Step 3: Store warp results to shared memory
    if (lane_id == 0) {
        warpLevelMaxs[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Step 4: Final reduction across warps (only first warp participates)
    float block_max = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warpLevelMaxs[lane_id] : 0.0f;
        block_max = warpReduceMax(val);
    }
    
    __shared__ float reciprocal_scale;  
    if (tid == 0) {
        reciprocal_scale = block_max / FP8_E4M3_MAX;
        scale[token_idx] = reciprocal_scale;
    }
    __syncthreads();

    const float scale_val = 1.0f / reciprocal_scale;

        
    // Step 6: Quantize input to FP8 with clamping
    for(int i = tid; i < hidden_dim; i += block_dim) {
        float val = convert_to_float(input_ptr[i]) * scale_val;
        // Clamp to FP8_E4M3 range [-448, 448]
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        output_ptr[i] = static_cast<FP8_TYPE>(val);
    }
}


void per_token_quant_fp8(
    torch::Tensor& input,    // [M,N]
    torch::Tensor& output,   // [M,N]
    torch::Tensor& scale,    // [M]
    int block_size = 256
    )
{
    const auto input_sizes = input.sizes();
    const int64_t num_tokens = input_sizes[0];
    const int64_t hidden_dim = input_sizes[1];

    const int num_blocks = num_tokens;
    
    dim3 grid(num_blocks);
    dim3 block(block_size);

    // Shared memory: one float per warp to store warp-level max
    const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    const int shared_mem_size = num_warps * sizeof(float);
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on input dtype
    if (input.scalar_type() == at::ScalarType::Half) {
        per_token_quant_fp8_kernel<__half><<<grid, block, shared_mem_size, stream>>>(
            reinterpret_cast<__half*>(input.data_ptr<at::Half>()),
            output.data_ptr<FP8_TYPE>(),
            scale.data_ptr<float>(),
            hidden_dim,
            num_tokens);
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        per_token_quant_fp8_kernel<nv_bfloat16><<<grid, block, shared_mem_size, stream>>>(
            reinterpret_cast<nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
            output.data_ptr<FP8_TYPE>(),
            scale.data_ptr<float>(),
            hidden_dim,
            num_tokens);
    } else {
        AT_ERROR("per_token_quant_fp8: Unsupported input dtype. Only float16 and bfloat16 are supported.");
    }
}