#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/all.h>
#include "kernel/reduction/reduce_sum.h"
#include "kernel/batch_reduction/batch_reduce.h"
#include "kernel/quant/per_token_quant_fp8.h"

// Python接口包装函数 - 1D reduction
torch::Tensor reduce_sum_wrapper(torch::Tensor input, int block_size = 256) {
    // 检查输入
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 1, "Input must be 1-dimensional");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
                "Input must be float32");
    
    // 确保输入连续
    input = input.contiguous();
    
    // 创建输出tensor
    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .device(input.device());
    torch::Tensor output = torch::empty({1}, options);
    
    // 调用CUDA函数
    reduce_sum_cuda(input, output, block_size);
    
    return output;
}

// Python接口包装函数 - 2D batch reduction
torch::Tensor batch_reduce_sum_wrapper(
    torch::Tensor input,
    int block_size = 256,
    bool use_optimized = true)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2-dimensional [m, n]");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
                "Input must be float32");
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int m = input.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .device(input.device());
    torch::Tensor output = torch::empty({m}, options);
    
    // Call CUDA function
    batch_reduce_sum_cuda(input, output, block_size, use_optimized);
    
    return output;
}

// Python接口包装函数 - FP8 Quantization
std::tuple<torch::Tensor, torch::Tensor> per_token_quant_fp8_wrapper(
    torch::Tensor input,
    int block_size = 256)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2-dimensional [num_tokens, hidden_dim]");
    // TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
    //             "Input must be float32");
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    const int64_t num_tokens = input.size(0);
    const int64_t hidden_dim = input.size(1);
    
    // Create output tensors
    auto options_fp8 = torch::TensorOptions()
                           .dtype(torch::kFloat8_e4m3fn)
                           .device(input.device());
    auto options_scale = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(input.device());
    
    torch::Tensor output = torch::empty({num_tokens, hidden_dim}, options_fp8);
    torch::Tensor scale = torch::empty({num_tokens}, options_scale);
    
    // Call CUDA function
    per_token_quant_fp8(input, output, scale, block_size);
    
    return std::make_tuple(output, scale);
}

PYBIND11_MODULE(cuda_reduce, m) {
    m.doc() = "CUDA operations: reduce sum and FP8 quantization";
    
    // 1D reduction: reduce entire array to single value
    m.def("reduce_sum", &reduce_sum_wrapper,
          py::arg("input"),
          py::arg("block_size") = 256,
          "Compute sum of all elements in 1D array\n\n"
          "Args:\n"
          "    input (torch.Tensor): Input 1D CUDA tensor (float32)\n"
          "    block_size (int): CUDA block size, default 256\n\n"
          "Returns:\n"
          "    torch.Tensor: Scalar tensor containing the sum");
    
    // 2D batch reduction: reduce each row independently
    m.def("batch_reduce_sum", &batch_reduce_sum_wrapper,
          py::arg("input"),
          py::arg("block_size") = 256,
          py::arg("use_optimized") = true,
          "Compute row-wise sum of 2D tensor (batch reduction)\n\n"
          "Args:\n"
          "    input (torch.Tensor): Input 2D CUDA tensor [m, n] (float32)\n"
          "    block_size (int): CUDA block size, default 256\n"
          "    use_optimized (bool): Use optimized warp shuffle version, default True\n\n"
          "Returns:\n"
          "    torch.Tensor: Output 1D tensor [m] containing row-wise sums");
    
    // FP8 quantization: per-token quantization
    m.def("per_token_quant_fp8", &per_token_quant_fp8_wrapper,
          py::arg("input"),
          py::arg("block_size") = 256,
          "Per-token (per-row) quantization to FP8 E4M3 format\n\n"
          "For each token (row), this function:\n"
          "1. Finds the maximum absolute value across the hidden dimension\n"
          "2. Computes a per-token scale factor: scale = 448.0 / max_abs_value\n"
          "3. Quantizes all elements in that row: output = clamp(input * scale, -448, 448)\n\n"
          "Args:\n"
          "    input (torch.Tensor): Input 2D CUDA tensor [num_tokens, hidden_dim] (float32 or bfloat16)\n"
          "    block_size (int): CUDA block size, default 256\n\n"
          "Returns:\n"
          "    tuple: (output, scale) where\n"
          "        output (torch.Tensor): Quantized tensor [num_tokens, hidden_dim] (float8_e4m3fn)\n"
          "        scale (torch.Tensor): Scale factors [num_tokens] (float32)");
}