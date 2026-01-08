#ifndef REDUCE_SUM_H
#define REDUCE_SUM_H

#include <torch/extension.h>

// C++接口函数声明
void reduce_sum_cuda(
    torch::Tensor& input,   // [n]
    torch::Tensor& output,  // [1]
    int block_size = 256);

#endif // REDUCE_SUM_H