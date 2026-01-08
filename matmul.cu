
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    itn col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if(row < M && col < N){
        for(int i = 0; i < K; i++){
            sum += A[row * K+i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_shared(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;  
    int col = bx * TILE_SIZE + tx;   
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for(int t=0; t<numTiles; ++t){
        if(row < M && t*TILE_SIZE+tx < K){
            As[ty][tx] = A[row*K + t*TILE_SIZE + tx];
        }
        else{
            As[ty][tx] = 0.0f;
        }
        if(col < N && t*TILE_SIZE + ty < K){
            Bs[ty][tx] = B[(t*TILE_SIZE + ty)*N + col];
        }
        else{
            Bs[ty][tx] = 0.0f;
        }
    }
    __syncthreads();

    for(int t=0;t<TILE_SIZE;++t){
        sum+=As[ty][t] * Bs[t][tx];
    }
    __syncthreads();
    if(row < M && col < N){
        C[row*N + col] = sum;
    }

}