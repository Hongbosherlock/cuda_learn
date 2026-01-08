#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include <torch/extension.h>

void reduce_sum_cuda(
    torch::Tensor& input,   // [n]
    torch::Tensor& output,  // [1]
    int block_size = 256);

#endif // REDUCE_SUM_H