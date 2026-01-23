"""
RMSNorm CUDA实现测试
比较CUDA算子和PyTorch RMSNorm的精度和性能
"""
import torch
import torch.nn as nn
import time
import sys
import os

# 添加父目录到路径以导入CUDA扩展
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import cuda_reduce  # 假设CUDA扩展模块名为cuda_reduce
    CUDA_AVAILABLE = True
except ImportError:
    print("警告: 无法导入CUDA扩展模块，将只运行PyTorch版本")
    CUDA_AVAILABLE = False


class Qwen3RMSNorm(nn.Module):
    """PyTorch参考实现"""
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def test_correctness(batch_size=32, hidden_size=4096, eps=1e-6, device='cuda'):
    """测试正确性"""
    print(f"\n{'='*60}")
    print(f"正确性测试: batch_size={batch_size}, hidden_size={hidden_size}")
    print(f"{'='*60}")
    
    # 创建测试数据
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    
    # PyTorch参考实现
    torch_norm = Qwen3RMSNorm(hidden_size, eps=eps).to(device)
    torch_norm.weight.data = weight.clone()
    
    with torch.no_grad():
        torch_output = torch_norm(input_tensor)
    
    if not CUDA_AVAILABLE:
        print("跳过CUDA测试（未找到CUDA扩展）")
        return
    
    # CUDA实现
    cuda_output = torch.empty_like(input_tensor)
    cuda_reduce.RMSNorm(input_tensor, weight, cuda_output, eps)
    
    # 计算误差
    abs_diff = torch.abs(torch_output - cuda_output)
    rel_diff = abs_diff / (torch.abs(torch_output) + 1e-8)
    
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    
    print(f"\n误差统计:")
    print(f"  最大绝对误差: {max_abs_error:.2e}")
    print(f"  平均绝对误差: {mean_abs_error:.2e}")
    print(f"  最大相对误差: {max_rel_error:.2e}")
    print(f"  平均相对误差: {mean_rel_error:.2e}")
    
    # 判断是否通过
    atol = 1e-5  # 绝对容差
    rtol = 1e-4  # 相对容差
    passed = (max_abs_error < atol) or (max_rel_error < rtol)
    
    if passed:
        print(f"\n✅ 测试通过！")
    else:
        print(f"\n❌ 测试失败！误差超出容差范围")
        print(f"   期望: max_abs < {atol} 或 max_rel < {rtol}")
    
    return passed


def benchmark_performance(batch_size=32, hidden_size=4096, eps=1e-6, 
                         warmup=10, iterations=100, device='cuda'):
    """性能基准测试"""
    print(f"\n{'='*60}")
    print(f"性能测试: batch_size={batch_size}, hidden_size={hidden_size}")
    print(f"warmup={warmup}, iterations={iterations}")
    print(f"{'='*60}")
    
    # 创建测试数据
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    
    # PyTorch实现
    torch_norm = Qwen3RMSNorm(hidden_size, eps=eps).to(device)
    torch_norm.weight.data = weight.clone()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = torch_norm(input_tensor)
    torch.cuda.synchronize()
    
    # 测试PyTorch性能
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = torch_norm(input_tensor)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"\nPyTorch RMSNorm: {torch_time:.4f} ms/iter")
    
    if not CUDA_AVAILABLE:
        print("跳过CUDA性能测试（未找到CUDA扩展）")
        return
    
    # CUDA实现
    cuda_output = torch.empty_like(input_tensor)
    
    # Warmup
    for _ in range(warmup):
        cuda_reduce.RMSNorm(input_tensor, weight, cuda_output, eps)
    torch.cuda.synchronize()
    
    # 测试CUDA性能
    start = time.time()
    for _ in range(iterations):
        cuda_reduce.RMSNorm(input_tensor, weight, cuda_output, eps)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start) / iterations * 1000  # ms
    
    print(f"CUDA RMSNorm:    {cuda_time:.4f} ms/iter")
    
    speedup = torch_time / cuda_time
    print(f"\n加速比: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✅ CUDA实现比PyTorch快 {speedup:.2f}x")
    else:
        print(f"⚠️  CUDA实现比PyTorch慢 {1/speedup:.2f}x")
    
    return torch_time, cuda_time, speedup


def test_edge_cases():
    """测试边界情况"""
    print(f"\n{'='*60}")
    print("边界情况测试")
    print(f"{'='*60}")
    
    if not CUDA_AVAILABLE:
        print("跳过CUDA边界测试（未找到CUDA扩展）")
        return
    
    device = 'cuda'
    test_cases = [
        # (batch_size, hidden_size, description)
        (1, 128, "小batch + 小hidden"),
        (1, 4096, "小batch + 大hidden"),
        (128, 128, "大batch + 小hidden"),
        (64, 5120, "中等batch + 中等hidden"),
    ]
    
    all_passed = True
    for batch_size, hidden_size, desc in test_cases:
        print(f"\n测试: {desc}")
        try:
            passed = test_correctness(batch_size, hidden_size, device=device)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            all_passed = False
    
    if all_passed:
        print(f"\n✅ 所有边界测试通过！")
    else:
        print(f"\n❌ 部分边界测试失败！")


def main():
    """主测试函数"""
    print("="*60)
    print("RMSNorm CUDA实现测试套件")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    device = 'cuda'
    print(f"\n使用设备: {torch.cuda.get_device_name(0)}")
    
    # 1. 基础正确性测试
    test_correctness(batch_size=32, hidden_size=4096, device=device)
    
    # 2. 边界情况测试
    test_edge_cases()
    
    # 3. 性能测试 - 不同配置
    configs = [
        (32, 4096, "标准配置"),
        (64, 5120, "Qwen配置"),
        (128, 2048, "大batch"),
        (16, 8192, "大hidden"),
    ]
    
    print(f"\n{'='*60}")
    print("性能基准测试")
    print(f"{'='*60}")
    
    for batch_size, hidden_size, desc in configs:
        print(f"\n配置: {desc}")
        benchmark_performance(batch_size, hidden_size, device=device)
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()