import torch
import time

# Import compiled CUDA extension
try:
    import cuda_reduce
except ImportError:
    print("Error: batch_reduce module not found. Please build first:")
    print("  cd cuda_learn/kernel/batch_reduction && python3 setup.py install")
    exit(1)

def test_accuracy():
    """Test accuracy of batch reduction"""
    print("=" * 60)
    print("Accuracy Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    test_cases = [
        ("Small matrix", 10, 100),
        ("Medium matrix", 100, 1000),
        ("Large matrix", 1000, 10000),
        ("Wide matrix", 10, 100000),
    ]
    
    all_passed = True
    
    for name, m, n in test_cases:
        torch.manual_seed(42)
        data = torch.randn(m, n, dtype=torch.float32, device='cuda')
        
        # PyTorch reference
        expected = torch.sum(data, dim=1)
        
        # CUDA optimized version
        result_opt = cuda_reduce.batch_reduce_sum(data, use_optimized=True)
        
        # CUDA naive version
        result_naive = cuda_reduce.batch_reduce_sum(data, use_optimized=False)
        
        # Check optimized version
        try:
            torch.testing.assert_close(result_opt, expected, atol=1e-2, rtol=1e-5)
            passed_opt = True
        except AssertionError:
            passed_opt = False
        
        # Check naive version
        try:
            torch.testing.assert_close(result_naive, expected, atol=1e-2, rtol=1e-5)
            passed_naive = True
        except AssertionError:
            passed_naive = False
        
        all_passed = all_passed and passed_opt and passed_naive
        
        # Calculate errors
        max_error_opt = torch.max(torch.abs(result_opt - expected)).item()
        max_error_naive = torch.max(torch.abs(result_naive - expected)).item()
        
        status_opt = "✓" if passed_opt else "✗"
        status_naive = "✓" if passed_naive else "✗"
        
        print(f"\n{name} (m={m}, n={n}):")
        print(f"  Optimized: {status_opt} (max_error={max_error_opt:.2e})")
        print(f"  Naive:     {status_naive} (max_error={max_error_naive:.2e})")
    
    if all_passed:
        print("\n✓ All accuracy tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    return all_passed

def test_performance():
    """Test performance of batch reduction"""
    print("\n" + "=" * 60)
    print("Performance Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    test_configs = [
        # (m, n, description)
        (100, 1000, "Small batch, short rows"),
        (1000, 1000, "Medium batch, medium rows"),
        (100, 10000, "Small batch, long rows"),
        (1000, 10000, "Medium batch, long rows"),
        (10000, 1000, "Large batch, medium rows"),
    ]
    
    num_runs = 100
    block_sizes = [128, 256, 512]
    
    print(f"\n{'Config':<30s} | {'Block':>5s} | {'Optimized':>11s} | {'Naive':>11s} | {'PyTorch':>11s} | {'Speedup':>8s}")
    print("-" * 95)
    
    for m, n, desc in test_configs:
        torch.manual_seed(42)
        data = torch.randn(m, n, dtype=torch.float32, device='cuda')
        
        # Find best block size for optimized version
        best_time_opt = float('inf')
        best_block_opt = 256
        
        for block_size in block_sizes:
            # Warmup
            for _ in range(10):
                _ = cuda_reduce.batch_reduce_sum(data, block_size=block_size, use_optimized=True)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = cuda_reduce.batch_reduce_sum(data, block_size=block_size, use_optimized=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / num_runs
            
            if elapsed < best_time_opt:
                best_time_opt = elapsed
                best_block_opt = block_size
        
        # Benchmark naive version with best block size
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = cuda_reduce.batch_reduce_sum(data, block_size=best_block_opt, use_optimized=False)
        torch.cuda.synchronize()
        time_naive = (time.perf_counter() - start) / num_runs
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = torch.sum(data, dim=1)
        torch.cuda.synchronize()
        time_pytorch = (time.perf_counter() - start) / num_runs
        
        speedup_vs_naive = time_naive / best_time_opt
        speedup_vs_pytorch = time_pytorch / best_time_opt
        
        config_str = f"{desc} ({m}x{n})"
        print(f"{config_str:<30s} | {best_block_opt:>5d} | {best_time_opt*1000:>9.4f}ms | "
              f"{time_naive*1000:>9.4f}ms | {time_pytorch*1000:>9.4f}ms | {speedup_vs_pytorch:>7.2f}x")
    
    print("\nNotes:")
    print("- Optimized version uses warp shuffle for better performance")
    print("- PyTorch's implementation is highly optimized")
    print("- Performance depends on GPU model and data size")

def main():
    print("\nBatch Reduction (Row-wise Sum) Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA device not detected")
        return
    
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Accuracy test
    accuracy_passed = test_accuracy()
    
    if not accuracy_passed:
        print("\n⚠️  Warning: Accuracy test failed")
        return
    
    # Performance test
    test_performance()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()