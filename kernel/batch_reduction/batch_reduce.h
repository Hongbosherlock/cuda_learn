#ifndef BATCH_REDUCE_H
#define BATCH_REDUCE_H

#include <torch/extension.h>

// C++ interface function
void batch_reduce_sum_cuda(
    torch::Tensor& input,   // [m, n]
    torch::Tensor& output,  // [m]
    int block_size = 256,
    bool use_optimized = true);

#endif // BATCH_REDUCE_H