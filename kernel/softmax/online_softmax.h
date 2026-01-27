#pragma once

#include <torch/extension.h>

// 2D online softmax, shape: [rows, cols]
// Returns a tensor with same shape/dtype/device as input.
torch::Tensor online_softmax_cuda(torch::Tensor input);
