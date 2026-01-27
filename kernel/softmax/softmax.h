#pragma once

#include <torch/extension.h>

// Online softmax (single-pass) entry point
torch::Tensor softmax_online_cuda(torch::Tensor input);

