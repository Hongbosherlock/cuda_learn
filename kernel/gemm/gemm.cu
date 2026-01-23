/**
 * 最简单的GEMM kernel - Naive实现
 * 
 * 计算: C = A * B
 * 其中: A (M x K), B (K x N), C (M x N)
 * 
 * 思路: 每个线程计算输出矩阵C的一个元素
 *       C[row][col] = sum(A[row][k] * B[k][col]) for k in [0, K)
 */
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    // 计算当前线程负责的输出位置 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // 遍历K维度，计算点积
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}


/**
 * 共享内存分块优化的GEMM kernel - Tiled版本
 * 
 * 优化策略:
 * 1. 将A和B矩阵分块加载到共享内存（片上缓存）
 * 2. 一个block内的所有线程协同加载数据，实现数据复用
 * 3. 大幅减少全局内存访问次数
 * 
 * 性能提升原理:
 *   - Naive版本: 每个线程独立从全局内存读取K次
 *   - Tiled版本: 一个tile内所有线程共享数据，全局内存访问减少约TILE_SIZE倍
 * 
 * TILE_SIZE: 分块大小，通常设为16或32
 */
#define TILE_SIZE 16

__global__ void gemm_shared(float* A, float* B, float* C, int M, int N, int K) {
    // 共享内存：存储A和B的分块（片上高速缓存）
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 计算当前线程负责的输出位置
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; //沿着列 从上到下读
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  //沿着行 从左到右读
    
    float sum = 0.0f;
    
    // 沿K维度分块遍历
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // === 阶段1: 协同加载数据到共享内存 ===
        
        // 加载A矩阵的tile: A[row][t*TILE_SIZE : (t+1)*TILE_SIZE]
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;  // 边界padding
        }
        
        // 加载B矩阵的tile: B[t*TILE_SIZE : (t+1)*TILE_SIZE][col]
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;  // 边界padding
        }
        
        // 同步：确保block内所有线程都完成数据加载
        // 现在 As 和 Bs 已经填满了数据
        __syncthreads();
        
        // === 阶段2: 使用共享内存数据进行计算 ===
        // 此时数据在片上，访问速度比全局内存快~100倍
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // 同步：确保所有线程都用完当前tile数据，再加载下一块
        __syncthreads();
    }
    //这里沿着 K 的循环结束，得到sum = C[row][col] 的完整计算结果

    
    // 写回结果到全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


/**
 * 共享内存 + 线程级分块优化 - 每个线程计算多个元素
 * 
 * 优化策略:
 * 1. 保持共享内存分块的优势
 * 2. 每个线程计算一个小的输出块（THREAD_TILE_Y × THREAD_TILE_X）
 * 3. 介于naive共享内存版本和完整寄存器分块之间的折中方案
 * 
 * 性能提升原理:
 *   - 每个线程计算更多元素，提高数据复用
 *   - 减少线程数量，增加每个线程的工作量
 *   - 更好的指令级并行（ILP）
 * 
 * 参数说明:
 *   TILE_SIZE_M2: 共享内存tile的大小（行/列）
 *   THREAD_TILE_Y: 每个线程负责的行数
 *   THREAD_TILE_X: 每个线程负责的列数
 */
#define TILE_SIZE_M2 64           // 共享内存tile大小
#define THREAD_TILE_Y 8           // 每个线程计算8行
#define THREAD_TILE_X 1           // 每个线程计算1列（也可以改成更大值）
#define BLOCK_ROWS (TILE_SIZE_M2 / THREAD_TILE_Y)  // 8
#define BLOCK_COLS (TILE_SIZE_M2 / THREAD_TILE_X)  // 64

__global__ void gemm_shared_multi_elements(float* A, float* B, float* C, int M, int N, int K) {
    // 共享内存：存储A和B的分块
    __shared__ float As[TILE_SIZE_M2][TILE_SIZE_M2];
    __shared__ float Bs[TILE_SIZE_M2][TILE_SIZE_M2];
    
    // 当前线程在block内的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前线程负责的输出块的起始位置
    int rowStart = blockIdx.y * TILE_SIZE_M2 + ty * THREAD_TILE_Y;
    int col = blockIdx.x * TILE_SIZE_M2 + tx * THREAD_TILE_X;
    
    // 每个线程维护THREAD_TILE_Y个累加器（寄存器）
    float sum[THREAD_TILE_Y];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++) {
        sum[i] = 0.0f;
    }
    
    // 沿K维度分块遍历
    int numTiles = (K + TILE_SIZE_M2 - 1) / TILE_SIZE_M2;
    
    for (int t = 0; t < numTiles; t++) {
        // === 阶段1: 协同加载数据到共享内存 ===
        
        // 每个线程加载A矩阵的THREAD_TILE_Y个元素（一列中的多行）
        #pragma unrollm
        for (int i = 0; i < THREAD_TILE_Y; i++) {
            int aRow = blockIdx.y * TILE_SIZE_M2 + ty * THREAD_TILE_Y + i;
            int aCol = t * TILE_SIZE_M2 + tx;
            
            if (aRow < M && aCol < K) {
                As[ty * THREAD_TILE_Y + i][tx] = A[aRow * K + aCol];
            } else {
                As[ty * THREAD_TILE_Y + i][tx] = 0.0f;
            }
        }
        
        // 每个线程加载B矩阵的THREAD_TILE_Y个元素（一行中的多列）
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_Y; i++) {
            int bRow = t * TILE_SIZE_M2 + ty * THREAD_TILE_Y + i;
            int bCol = blockIdx.x * TILE_SIZE_M2 + tx;
            
            if (bRow < K && bCol < N) {
                Bs[ty * THREAD_TILE_Y + i][tx] = B[bRow * N + bCol];
            } else {
                Bs[ty * THREAD_TILE_Y + i][tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // === 阶段2: 计算 - 每个线程处理THREAD_TILE_Y行 ===
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_M2; k++) {
            float b_val = Bs[k][tx];  // 预先加载B的值到寄存器
            
            // 每个线程计算THREAD_TILE_Y个输出元素
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; i++) {
                sum[i] += As[ty * THREAD_TILE_Y + i][k] * b_val;
            }
        }
        
        __syncthreads();
    }
    
    // === 阶段3: 写回结果 ===
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++) {
        int row = rowStart + i;
        if (row < M && col < N) {
            C[row * N + col] = sum[i];
        }
    }
}


/**
 * 共享内存 + 线程级分块优化 - 每个线程计算8×8个元素
 * 
 * 优化策略:
 * 1. 保持共享内存分块的优势
 * 2. 每个线程计算8×8=64个输出元素
 * 3. 使用二维寄存器数组存储中间结果
 * 
 * 性能提升原理:
 *   - 数据复用度提高（A的8个元素×B的8个元素 = 64次计算）
 *   - 更高的指令级并行
 *   - 接近完整寄存器分块的性能
 * 
 * 参数说明:
 *   TILE_SIZE_M3: 共享内存tile的大小
 *   THREAD_TILE_Y3, THREAD_TILE_X3: 每个线程负责的行数和列数
 */
#define TILE_SIZE_M3 64           // 共享内存tile大小
#define THREAD_TILE_Y3 8          // 每个线程计算8行
#define THREAD_TILE_X3 8          // 每个线程计算8列
#define BLOCK_ROWS3 (TILE_SIZE_M3 / THREAD_TILE_Y3)  // 8
#define BLOCK_COLS3 (TILE_SIZE_M3 / THREAD_TILE_X3)  // 8

__global__ void gemm_shared_8x8(float* A, float* B, float* C, int M, int N, int K) {
    // 共享内存：存储A和B的分块
    __shared__ float As[TILE_SIZE_M3][TILE_SIZE_M3];
    __shared__ float Bs[TILE_SIZE_M3][TILE_SIZE_M3];
    
    // 当前线程在block内的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前线程负责的输出块的起始位置
    int rowStart = blockIdx.y * TILE_SIZE_M3 + ty * THREAD_TILE_Y3;
    int colStart = blockIdx.x * TILE_SIZE_M3 + tx * THREAD_TILE_X3;
    
    // 每个线程维护8×8个累加器（寄存器）
    float sum[THREAD_TILE_Y3][THREAD_TILE_X3];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y3; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_X3; j++) {
            sum[i][j] = 0.0f;
        }
    }
    
    // 沿K维度分块遍历
    int numTiles = (K + TILE_SIZE_M3 - 1) / TILE_SIZE_M3;
    
    for (int t = 0; t < numTiles; t++) {
        // === 阶段1: 协同加载数据到共享内存 ===
        
        // 每个线程加载A矩阵的8×8=64个元素
        // 内存访问不够合并,相邻线程访问的地址跨度大，合并效率低
        // 64个线程，每个加载64个元素 = 4096次访问
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_Y3; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X3; j++) {
                int aRow = blockIdx.y * TILE_SIZE_M3 + ty * THREAD_TILE_Y3 + i;
                int aCol = t * TILE_SIZE_M3 + tx * THREAD_TILE_X3 + j;
                
                if (aRow < M && aCol < K) {
                    As[ty * THREAD_TILE_Y3 + i][tx * THREAD_TILE_X3 + j] = A[aRow * K + aCol];
                } else {
                    As[ty * THREAD_TILE_Y3 + i][tx * THREAD_TILE_X3 + j] = 0.0f;
                }
            }
        }
        
        // 每个线程加载B矩阵的8×8=64个元素
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_Y3; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X3; j++) {
                int bRow = t * TILE_SIZE_M3 + ty * THREAD_TILE_Y3 + i;
                int bCol = blockIdx.x * TILE_SIZE_M3 + tx * THREAD_TILE_X3 + j;
                
                if (bRow < K && bCol < N) {
                    Bs[ty * THREAD_TILE_Y3 + i][tx * THREAD_TILE_X3 + j] = B[bRow * N + bCol];
                } else {
                    Bs[ty * THREAD_TILE_Y3 + i][tx * THREAD_TILE_X3 + j] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // === 阶段2: 计算 - 每个线程处理8×8个元素 ===
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_M3; k++) {
            // 外积计算：8×8次乘加运算
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y3; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_X3; j++) {
                    sum[i][j] += As[ty * THREAD_TILE_Y3 + i][k] * 
                                  Bs[k][tx * THREAD_TILE_X3 + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // === 阶段3: 写回结果 ===
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y3; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_X3; j++) {
            int row = rowStart + i;
            int col = colStart + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}


/**
 * 寄存器分块优化的GEMM kernel - Register Tiling版本
 * 
 * 优化策略:
 * 1. 每个线程计算C矩阵的一个小块（TM×TN），而不是单个元素
 * 2. 使用寄存器存储中间结果，减少共享内存bank conflict
 * 3. 提高计算密度和指令级并行（ILP）
 * 
 * 性能提升原理:
 *   - 共享内存版本: 每个线程计算1个输出元素
 *   - 寄存器分块: 每个线程计算TM×TN个输出元素
 *   - 数据复用度提高，寄存器访问更快
 * 
 * 总结：
 * 每个 block 计算 C 的一个 BM×BN tile；沿 K 维按 BK 分块；
 * 把 A 的 BM×BK 和 B 的 BK×BN 放进 shared；每个线程用寄存器累加 TM×TN 的输出子块。

 * 参数说明:
 *   BM, BN: Block tile大小（共享内存分块）
 *   BK: K维度的分块大小
 *   TM, TN: Thread tile大小（每个线程负责的输出块）
 */
#define BM 128        // Block tile M维度
#define BN 128        // Block tile N维度  
#define BK 8          // Block tile K维度
#define TM 8          // Thread tile M维度（每个线程计算8行）
#define TN 8          // Thread tile N维度（每个线程计算8列）

__global__ void gemm_register_tiling(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K,
                                     float alpha, float beta)
{
    // Shared memory tiles
    __shared__ float As[BM][BK]; // 128x8
    __shared__ float Bs[BK][BN]; // 8x128

    // Thread coordinates in block
    const int tx = threadIdx.x; // [0..15]
    const int ty = threadIdx.y; // [0..15]
    const int tid = ty * blockDim.x + tx; // [0..255]

    // Block coordinates in grid
    const int bx = blockIdx.x; // N dimension (columns)
    const int by = blockIdx.y; // M dimension (rows)

    
    // 计算当前线程负责的输出块的起始位置
    int threadRowStart = by * BM + ty * TM; //第by个block里的第ty行thread
    int threadColStart = bx * BN + tx * TN; //第bx列block里的第tx列thread
    
    // 寄存器：存储每个线程计算的TM×TN个输出元素
    // float regC[TM][TN] = {0.0f};
      float regC[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      regC[i][j] = 0.0f;
    }
  }
    
    // 寄存器：临时存储从共享内存加载的数据
    float regA[TM];
    float regB[TN];
    
    // 沿K维度分块遍历
    //- register_tiling使用更小的K tile，可以有更大的M/N tile
    //- 更大的M/N tile → 更好的数据复用
    int numTiles = (K + BK - 1) / BK;
    
    for (int t = 0; t < numTiles; t++) {
        // === 阶段1: 协同加载数据到共享内存 ===
        
        // 每个线程加载多个元素
        // 加载A矩阵: 每个线程加载 (BM*BK)/(blockDim.x*blockDim.y) 个元素
        // 一个block(16,16) 加载(128,8)这个矩阵大小的元素，每个线程加载 4 个元素
        int numLoadsA = (BM * BK) / (blockDim.x * blockDim.y);
        for (int i = 0; i < numLoadsA; i++) {
            // idx的范围在0-1023，先求idx 再映射到二维
            int idx = tid + i * blockDim.x * blockDim.y; // tid + i*256
            int aRow = idx / BK;                 // [0..127]
            int aCol = idx % BK;                 // [0..7]
            
            //映射到A矩阵的坐标
            int globalRow = by * BM + aRow;
            int globalCol = t * BK + aCol;
            
            if (globalRow < M && globalCol < K) {
                As[aRow][aCol] = A[globalRow * K + globalCol];
            } else {
                As[aRow][aCol] = 0.0f;
            }
            // i=0 会填满 As 的 row 0..31，全 8 列
            // i=1 会填满 As 的 row 32..63，全 8 列
            // i=2 会填满 As 的 row 64..95，全 8 列
            // i=3 会填满 As 的 row 96..127，全 8 列
        }
        
        // -------------------------
        // Load B tile into shared: Bs[BK][BN]
        // Total elements: 8*128 = 1024
        // Use 256 threads => 4 elements per thread
        // -------------------------        
        int numLoadsB = (BK * BN) / (blockDim.x * blockDim.y);
        for (int i = 0; i < numLoadsB; i++) {
            int idx = tid + i * blockDim.x * blockDim.y;
            int bRow = idx / BN;
            int bCol = idx % BN;
            int globalRow = t * BK + bRow;
            int globalCol = bx * BN + bCol;
            
            if (globalRow < K && globalCol < N) {
                Bs[bRow][bCol] = B[globalRow * N + globalCol];
            } else {
                Bs[bRow][bCol] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // === 阶段2: 寄存器级计算 ===
        // 遍历当前tile的K维度
        // -------------------------
        // Compute: regC += As(row0..row0+TM-1, kk) * Bs(kk, col0..col0+TN-1)
        // kk in [0..BK-1]
        // Use registers regA[TM], regB[TN]
        // -------------------------
        for (int k = 0; k < BK; k++) {
            // 从共享内存加载到寄存器
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                //加载As某一列的8个元素
                regA[i] = As[ty * TM + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                //加载Bs某一行的8个元素
                regB[j] = Bs[k][tx * TN + j];
            }
            
            // 寄存器级外积：TM×TN次乘加运算
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    regC[i][j] += regA[i] * regB[j];
                    // 这是FMA (Fused Multiply-Add) 指令
                    // 一条指令完成：乘法 + 加法
                    // 吞吐量：2 FLOPs/cycle
                }
            }
        }
        
        __syncthreads();
    }
    
    // === 阶段3: 写回结果到全局内存 ===
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalRow = threadRowStart + i;
            int globalCol = threadColStart + j;
            if (globalRow < M && globalCol < N) {
                int idx = globalRow * N + globalCol;
                float old = C[idx];                 // 读原 C
                C[idx] = alpha * regC[i][j] + beta * old;
            }
        }
    }
}


/**
 * ============================================================================
 * 调用示例和性能对比
 * ============================================================================
 */

// 示例1: Naive版本调用
void launch_gemm_naive(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(16, 16);  // 256个线程/block
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// 示例2: 共享内存版本调用
void launch_gemm_shared(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);  // 16×16=256个线程/block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// 示例3: 共享内存+线程级分块版本调用（每个线程处理8×1个元素）
void launch_gemm_shared_multi_elements(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    // 每个block的线程布局：
    // - threadIdx.x ∈ [0, BLOCK_COLS) = [0, 64)  (处理列方向)
    // - threadIdx.y ∈ [0, BLOCK_ROWS) = [0, 8)   (处理行方向)
    // - 总共 64×8 = 512 个线程/block
    // - 每个线程计算 8×1 = 8 个输出元素（同一列的8行）
    dim3 block(BLOCK_COLS, BLOCK_ROWS);  // (64, 8)
    
    // Grid布局：将整个矩阵C分成64×64的块
    dim3 grid((N + TILE_SIZE_M2 - 1) / TILE_SIZE_M2,   // 列方向的block数
              (M + TILE_SIZE_M2 - 1) / TILE_SIZE_M2);   // 行方向的block数
    
    gemm_shared_multi_elements<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// 示例4: 共享内存+线程级分块版本调用（每个线程处理8×8个元素）
void launch_gemm_shared_8x8(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    // 每个block的线程布局：
    // - threadIdx.x ∈ [0, BLOCK_COLS3) = [0, 8)  (处理列方向)
    // - threadIdx.y ∈ [0, BLOCK_ROWS3) = [0, 8)  (处理行方向)
    // - 总共 8×8 = 64 个线程/block
    // - 每个线程计算 8×8 = 64 个输出元素
    dim3 block(BLOCK_COLS3, BLOCK_ROWS3);  // (8, 8)
    
    // Grid布局：将整个矩阵C分成64×64的块
    dim3 grid((N + TILE_SIZE_M3 - 1) / TILE_SIZE_M3,   // 列方向的block数
              (M + TILE_SIZE_M3 - 1) / TILE_SIZE_M3);   // 行方向的block数
    
    gemm_shared_8x8<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// 示例5: 寄存器分块版本调用（推荐）
void launch_gemm_register_tiling(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    
    // 每个block的线程布局：
    // - threadIdx.x ∈ [0, BN/TN) = [0, 16)  (处理列方向)
    // - threadIdx.y ∈ [0, BM/TM) = [0, 16)  (处理行方向)
    // - 总共 16×16 = 256 个线程/block
    dim3 block(BN / TN, BM / TM);  // (16, 16)
    
    // Grid布局：将整个矩阵C分块
    dim3 grid((N + BN - 1) / BN,   // 列方向的block数
              (M + BM - 1) / BM);   // 行方向的block数
    
    gemm_register_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}

// 完整使用示例
void example_usage() {
    int M = 4096, N = 4096, K = 4096;
    
    // 1. 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // 2. 初始化数据（此处省略host端数据准备和拷贝）
    // cudaMemcpy(...);
    
    // 3. 调用kernel
    launch_gemm_register_tiling(d_A, d_B, d_C, M, N, K);
    
    // 4. 获取结果
    // cudaMemcpy(...);
    
    // 5. 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * 性能对比参考（A100 GPU, M=N=K=4096, FP32）:
 * 
 * ┌──────────────────────────┬──────────────┬───────────┬──────────────┬─────────────────┐
 * │ 版本                      │ 性能(GFLOPS) │ Block配置 │ 每线程输出   │ 理论峰值占比    │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ gemm_naive               │ ~50          │ 16×16     │ 1 (1×1)      │ 0.26%           │
 * │ (基准版本)                │              │ =256线程  │              │                 │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ gemm_shared              │ ~700         │ 16×16     │ 1 (1×1)      │ 3.6%            │
 * │ (共享内存分块)            │              │ =256线程  │              │ (14x ↑)         │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ gemm_shared_multi_elements│ ~1200       │ 64×8      │ 8 (8×1)      │ 6.2%            │
 * │ (线程级分块 8×1)          │              │ =512线程  │              │ (24x ↑ / 1.7x)  │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ gemm_shared_8x8          │ ~1800        │ 8×8       │ 64 (8×8)     │ 9.2%            │
 * │ (线程级分块 8×8)          │              │ =64线程   │              │ (36x ↑ / 2.6x)  │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ gemm_register_tiling     │ ~2500        │ 16×16     │ 64 (8×8)     │ 12.8%           │
 * │ (寄存器分块优化)          │              │ root=256线程  │              │ (50x ↑ / 3.6x)  │
 * ├──────────────────────────┼──────────────┼───────────┼──────────────┼─────────────────┤
 * │ cuBLAS (参考基准)         │ ~15000       │ 高度优化  │ -            │ 77%             │
 * │                          │              │           │              │ (300x ↑ / 6x)   │
 * └──────────────────────────┴──────────────┴───────────┴──────────────┴─────────────────┘
 * 

 每个线程计算更多结果会提高算术强度


 还记得一维 Thread Tile 中的例子吗？如果输入的 A 和 B 都是 7x7 的矩阵：
 1. 如果我们一次读取 1 行 A 和 1 列 B，当每一个线程只计算一个结果的时候，我们需要从 A 中读取 7 个数据，从 B 中读取 7 个数据，从 C 中读取 1 个数据，然后写 1 次 C。这样的话，每个线程需要读取 15 个数据，写 1 次数据。计算 16 个结果需要 16 个线程，共 16x16 = 256 次 IO。
 2. 如果我们一次读取 4 行 A 和 1 列 B，那么每一个线程计算 4 个结果，此时需要从 A 中读取 4x7 个数据，从 B 中读取 7 个数据，从 C 中读取 4 个数据，然后写 4 次 C。计算 16 个结果需要 4 个线程，共 4x43 = 172 次 IO。
 3. 如果我们一次读取 4 行 A 和 4 列 B，那么每一个线程计算 16 个结果，此时需要从 A 中读取 4x7 个数据，从 B 中读取 4x7 个数据，从 C 中读取 16 个数据，然后写 16 次 C。计算 16 个结果一共需要 1 个线程，共 1x88 = 88 次 IO。
 上述的 2 就是一维 Thread Tile 优化，上述的 3 就是 二维 Thread Tile 优化，计算结果不变的同时，减少 IO 次数，提升算法的执行时间。所以想要继续优化这个 Kernel 的性能，我们可以使用二维线程块来计算二维的矩阵块。

每个线程计算更多结果会提高算术强度

 * 注：
 * 1. A100 FP32理论峰值 ≈ 19.5 TFLOPS
 * 2. 相对提升第一个数字是相对naive，第二个是相对上一版本
 * 3. 以上性能为预估值，实际性能受矩阵大小、数据对齐、GPU架构等因素影响
 * 
 * 关键技术对比：
 * ┌──────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
 * │ 优化技术                  │ Shared Mem  │ Multi 8×1   │ Shared 8×8  │ Register    │
 * ├──────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
 * │ 共享内存分块              │ ✅          │ ✅          │ ✅          │ ✅          │
 * │ 线程级数据复用            │ ❌          │ ✅ (8×)     │ ✅ (64×)    │ ✅ (64×)    │
 * │ 寄存器级临时缓存          │ ❌          │ ❌          │ ❌          │ ✅          │
 * │ 协同加载（内存合并）      │ ✅          │ ⚠️ 中等     │ ⚠️ 中等     │ ✅ 完美     │
 * │ 指令级并行（ILP）         │ 低          │ 中          │ 高          │ 很高        │
 * │ 实现复杂度                │ 简单        │ 中等        │ 中等        │ 较高        │
 * └──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
 * 
 * 进一步优化方向：
 * - 🔄 双缓冲（Double Buffering）：隐藏数据加载延迟
 - pipline 优化。load和 compute overlap        cp.async
 * - 🔗 Warp级优化：利用warp shuffle指令
 * - 🚀 向量化加载：使用float4减少内存事务
 * - 💎 Tensor Core：使用wmma API实现10x+性能提升
 * - 🏆 CUTLASS库：达到cuBLAS级别性能（90%+ 峰值）
 */