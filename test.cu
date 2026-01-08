#define BLOCKSIZE 32  // 最佳块大小通常为16x16或32x32
#define PADDING 1     // 避免共享内存bank冲突

__global__ void matrixMulSharedMem(float *A, float *B, float *C, int M, int N, int K) {
    // 声明共享内存，添加padding避免bank冲突
    __shared__ float As[BLOCKSIZE][BLOCKSIZE + PADDING];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE + PADDING];
    
    // 线程索引
    int threadRow = threadIdx.y;  // 线程块内的行索引
    int threadCol = threadIdx.x;  // 线程块内的列索引
    
    // 块索引
    int blockRow = blockIdx.y;  // 行块索引
    int blockCol = blockIdx.x;  // 列块索引
    
    // 计算当前线程处理的全局矩阵位置
    int globalRow = blockRow * BLOCKSIZE + threadRow;
    int globalCol = blockCol * BLOCKSIZE + threadCol;
    
    // 指针偏移到起始位置
    float *A_start = A + globalRow * K;      // A的行起始位置
    float *B_start = B + globalCol;          // B的列起始位置
    float *C_ptr = C + globalRow * N + globalCol; // C的目标位置
    
    float tmp = 0.0f;
    
    // 分块处理矩阵乘法
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        float *A_block = A_start + bkIdx;
        float *B_block = B_start + bkIdx * N;
        
        // 加载A的子块到共享内存
        if (bkIdx + threadCol < K && globalRow < M) {
            As[threadRow][threadCol] = A_block[threadCol];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }
        
        // 加载B的子块到共享内存，注意访问模式
        if (bkIdx + threadRow < K && globalCol < N) {
            Bs[threadRow][threadCol] = B_block[threadRow * N];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }
        
        __syncthreads();  // 确保所有数据加载完成
        
        // 计算当前块的矩阵乘法（展开循环提高性能）
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
        }
        
        __syncthreads();  // 确保所有计算完成再加载下一块
    }
    
    // 只将有效结果写入全局内存
    if (globalRow < M && globalCol < N) {
        *C_ptr = tmp;
    }
}