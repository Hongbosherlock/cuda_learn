#pragma once

#include <torch/extension.h>

// RMSNorm CUDA kernel wrapper
void RMSNorm(torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output, 
             int batch_size, int d, float eps=1e-06);