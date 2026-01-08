#include <torch/extension.h>
#include "kernel/reduce_sum.h"
#include <pybind11/pybind11.h>
#include <torch/all.h>
// Python接口包装函数
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

PYBIND11_MODULE(cuda_reduce, m) {
    m.doc() = "CUDA reduce sum operations with PyTorch integration";
    
    m.def("reduce_sum", &reduce_sum_wrapper,
          py::arg("input"),
          py::arg("block_size") = 256,
          "Compute sum of array elements using CUDA reduction\n\n"
          "Args:\n"
          "    input (torch.Tensor): Input 1D CUDA tensor (float32 only)\n"
          "    block_size (int): CUDA block size, default 256\n\n"
          "Returns:\n"
          "    torch.Tensor: Scalar tensor containing the sum");
}
