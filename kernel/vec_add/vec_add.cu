#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Version 1: Baseline - 基础实现
// 优化点：最简单的实现，每个线程处理一个元素
// ============================================================================
__global__ void vectorAddV1(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


// ============================================================================
// Version 7: Combined Optimizations (Production Version)
// 优化点：综合所有优化技术
// - float4 向量化
// - __restrict__ 和 __ldg
// - grid-stride 循环
// - 循环展开
// - 对齐内存访问
// ============================================================================
__global__ void vectorAddV7(const float* __restrict__ A, 
                            const float* __restrict__ B, 
                            float* __restrict__ C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 向量化部分
    int N_vec = N / 4;
    for (int i = tid; i < N_vec; i += stride) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        float4 b = reinterpret_cast<const float4*>(B)[i];
        float4 c;
        
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        
        reinterpret_cast<float4*>(C)[i] = c;
    }
    
    // 处理剩余元素
    int start = N_vec * 4 + tid;
    if (start < N) {
        C[start] = __ldg(&A[start]) + __ldg(&B[start]);
    }
}

// ============================================================================
// Main solve function (required signature)
// ============================================================================
void solve(const float* A, const float* B, float* C, int N) {
    // 使用 Version 7 (最优版本)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 限制 block 数量以避免过度分配
    blocksPerGrid = min(blocksPerGrid, 2048);
    
    vectorAddV7<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

