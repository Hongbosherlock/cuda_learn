import torch
import time

# 导入编译好的CUDA扩展
try:
    import cuda_reduce
except ImportError:
    print("Error: cuda_reduce module not found. Please build the extension first:")
    print("  cd cuda_learn && python3 setup.py install")
    exit(1)

def test_accuracy():
    """测试CUDA reduce sum的精度"""
    print("=" * 60)
    print("精度测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False
    
    test_cases = [
        ("小数组", 100),
        ("中等数组", 10000),
        ("大数组", 1000000),
        ("超大数组", 10000000),
    ]
    
    all_passed = True
    
    for name, size in test_cases:
        # 生成随机测试数据
        torch.manual_seed(42)
        data = torch.randn(size, dtype=torch.float32, device='cuda')
        
        # PyTorch参考结果
        expected = torch.sum(data)
        
        # CUDA结果
        result = cuda_reduce.reduce_sum(data).squeeze()  # Convert shape [1] to []
        
        # Use torch.testing.assert_close for validation
        try:
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-5)
            passed = True
            status = "✓ PASS"
        except AssertionError as e:
            passed = False
            status = "✗ FAIL"
        
        all_passed = all_passed and passed
        
        # Calculate relative error for display
        result_val = result.cpu().item()
        expected_val = expected.cpu().item()
        abs_error = abs(result_val - expected_val)
        rel_error = abs(result_val - expected_val) / (abs(expected_val) + 1e-10)
        
        print(f"{name:12s} (n={size:>8d}): {status}")
        print(f"  Expected: {expected_val:.6f}")
        print(f"  Got:      {result_val:.6f}")
        print(f"  Abs Err:  {abs_error:.6f}")
        print(f"  Rel Err:  {rel_error:.2e}")
        print()
    
    if all_passed:
        print("✓ 所有精度测试通过！")
    else:
        print("✗ 部分测试失败")
    
    return all_passed

def test_performance():
    """测试CUDA reduce sum的性能"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    block_sizes = [128, 256, 512,1024]
    num_runs = 100  # 每个配置运行次数
    
    print(f"{'数组大小':>12s} | {'块大小':>8s} | {'自定义CUDA':>12s} | {'PyTorch':>12s} | {'加速比':>8s}")
    print("-" * 70)
    
    for size in sizes:
        # 生成测试数据
        torch.manual_seed(42)
        data = torch.randn(size, dtype=torch.float32, device='cuda')
        
        # 预热
        for _ in range(10):
            _ = cuda_reduce.reduce_sum(data)
            _ = torch.sum(data)
        torch.cuda.synchronize()
        
        # 测试不同的block size
        best_time = float('inf')
        best_block_size = 256
        
        for block_size in block_sizes:
            # 自定义CUDA性能测试
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = cuda_reduce.reduce_sum(data, block_size)
            torch.cuda.synchronize()
            cuda_time = (time.perf_counter() - start) / num_runs
            
            if cuda_time < best_time:
                best_time = cuda_time
                best_block_size = block_size
        
        # PyTorch性能测试
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = torch.sum(data)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / num_runs
        
        speedup = pytorch_time / best_time
        
        print(f"{size:>12d} | {best_block_size:>8d} | {best_time*1000:>10.4f}ms | {pytorch_time*1000:>10.4f}ms | {speedup:>7.2f}x")
    
    print("\n说明:")
    print("- PyTorch的实现经过高度优化，包含了多种优化技术")
    print("- 自定义kernel是基础实现，主要用于学习和理解")
    print("- 实际性能受GPU型号、数据传输开销等因素影响")

def main():
    print("\nCUDA Reduce Sum 测试")
    print("=" * 60)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA设备")
        print("请确保:")
        print("  1. 安装了支持CUDA的PyTorch")
        print("  2. 系统有可用的NVIDIA GPU")
        return
    
    print(f"✓ 检测到CUDA设备: {torch.cuda.get_device_name(0)}")
    print()
    
    # 精度测试
    accuracy_passed = test_accuracy()
    
    if not accuracy_passed:
        print("\n⚠️  警告: 精度测试未通过，请检查CUDA实现")
        return
    
    # 性能测试
    test_performance()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()