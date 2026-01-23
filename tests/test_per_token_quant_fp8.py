#!/usr/bin/env python3
"""
Test script for per_token_quant_fp8 kernel
Tests both accuracy and performance
"""

import torch
import time
import numpy as np
from typing import Tuple
from sgl_kernel import sgl_per_token_quant_fp8

# Import the custom kernel (adjust the import path based on your build setup)
try:
    import cuda_reduce
    KERNEL_AVAILABLE = True
except ImportError:
    print("Warning: Could not import cuda_reduce module. Will only test reference implementation.")
    KERNEL_AVAILABLE = False

def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    # Use standard FP8 E4M3 format (not FNUZ), matching CUDA kernel
    output = torch.empty_like(input, device=input.device, dtype=torch.float8_e4m3fn)

    sgl_per_token_quant_fp8(input, output, scale)
    scale = scale.reshape(-1, 1)
    # print(output)
    # exit(0)
    return output, scale

def per_token_quant_fp8_reference(input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of per-token FP8 quantization in PyTorch
    
    Args:
        input_tensor: [num_tokens, hidden_dim] float32 tensor
    
    Returns:
        output: [num_tokens, hidden_dim] float8_e4m3fn tensor
        scale: [num_tokens] float32 tensor
    """
    # FP8_E4M3_MAX = 448.0
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max

    fp8_min = -fp8_max

    # Find max absolute value per token (per row)
    max_abs_val = torch.amax(torch.abs(input_tensor), dim=1, keepdim=True)  # [num_tokens, 1]
    
    # Compute scale factor per token
    scale = max_abs_val / fp8_max  # [num_tokens, 1]
    scale = scale.to(torch.float32)
    # scale = scale.squeeze(1)  # [num_tokens]
    inv_scale = scale.reciprocal()
    # Quantize: multiply by scale and clamp
    # quantized = input_tensor * scale.unsqueeze(1)
    # quantized = torch.clamp(quantized, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    qweight = (input_tensor.to(torch.float32) * inv_scale).clamp(min=fp8_min, max=fp8_max)
    # Convert to FP8
    output = qweight.to(torch.float8_e4m3fn)
    
    return output, scale.squeeze(1)

def torch_per_token_quant_fp8(tensor, inv_scale):
    # The reference implementation that fully aligns to
    # the kernel being tested.
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = inv_scale.view(-1, 1)
    scale = inv_scale.reciprocal()
    qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight

def test_accuracy(num_tokens: int = 128, hidden_dim: int = 4096, device: str = "cuda", dtype=torch.float16):
    """Test accuracy by comparing kernel output with reference implementation"""
    print(f"\n{'='*60}")
    print(f"Testing Accuracy: [{num_tokens}, {hidden_dim}] dtype={dtype}")
    print(f"{'='*60}")
    
    # Generate random input
    torch.manual_seed(42)
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    
    # Reference implementation
    # ref_output, ref_scale = per_token_quant_fp8_reference(input_tensor)
    # print("torch version1 output",ref_output)
    # print("torch version1 scale",ref_scale)
    ref_output, ref_scale = sglang_per_token_quant_fp8(input_tensor)
    # print("sglang version output",ref_output)
    # print("sglang version scale",ref_scale)
    # ref_output = torch_per_token_quant_fp8(input_tensor, ref_scale)
    # print("ref_output",torch_output)
    # torch_output = torch_per_token_quant_fp8(input_tensor, sgl_scale)
    # print("sglang version output",torch_output)

    # exit(0)
    if not KERNEL_AVAILABLE:
        print("Kernel not available, skipping kernel test")
        return
    
    # Kernel implementation
    output, scale = cuda_reduce.per_token_quant_fp8(input_tensor)
    # print("cuda version output",output)
    # print("cuda version scale",scale)
    # output, scale = sglang_per_token_quant_fp8(input_tensor)
    # output, scale = per_token_quant_fp8_reference(input_tensor)
    # print("reference version output",output)
    # print("reference version scale",scale)

    # exit(0)
    # Convert both to float32 for comparison
    ref_output_float = ref_output.to(torch.float32)
    output_float = output.to(torch.float32)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(ref_output_float - output_float)).item()
    mean_diff = torch.mean(torch.abs(ref_output_float - output_float)).item()
    
    # Compare scales
    scale_max_diff = torch.max(torch.abs(ref_scale - scale)).item()
    scale_mean_diff = torch.mean(torch.abs(ref_scale - scale)).item()
    
    print(f"\nOutput Comparison:")
    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    
    print(f"\nScale Comparison:")
    print(f"  Max absolute difference:  {scale_max_diff:.6e}")
    print(f"  Mean absolute difference: {scale_mean_diff:.6e}")
    
    # Check if results are close enough
    # FP8 has limited precision, so we use a relaxed tolerance
    output_close = torch.allclose(ref_output_float, output_float, rtol=1e-2, atol=1e-1)
    scale_close = torch.allclose(ref_scale, scale, rtol=1e-2, atol=1e-2)
    
    if output_close and scale_close:
        print(f"\n✓ PASSED: Kernel output matches reference implementation")
    else:
        print(f"\n✗ FAILED: Kernel output differs from reference")
        if not output_close:
            print(f"  Output mismatch detected")
        if not scale_close:
            print(f"  Scale mismatch detected")
    
    return output_close and scale_close


def benchmark_performance(
    num_tokens: int = 128,
    hidden_dim: int = 4096,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda"
):
    """Benchmark performance of kernel vs reference implementation"""
    print(f"\n{'='*60}")
    print(f"Benchmarking Performance: [{num_tokens}, {hidden_dim}]")
    print(f"{'='*60}")
    
    # Generate random input
    torch.manual_seed(42)
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=device)
    
    # Benchmark reference implementation
    print(f"\nReference Implementation (PyTorch):")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(num_warmup):
        ref_output, ref_scale = per_token_quant_fp8_reference(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        ref_output, ref_scale = per_token_quant_fp8_reference(input_tensor)
    torch.cuda.synchronize()
    ref_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    print(f"  Average time: {ref_time:.4f} ms")
    
    if not KERNEL_AVAILABLE:
        print("\nKernel not available, skipping kernel benchmark")
        return
    
    # Benchmark kernel implementation
    print(f"\nCustom CUDA Kernel:")
    
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(num_warmup):
        output, scale = cuda_reduce.per_token_quant_fp8(input_tensor)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        output, scale = cuda_reduce.per_token_quant_fp8(input_tensor)
    torch.cuda.synchronize()
    kernel_time = (time.time() - start_time) / num_iterations * 1000  # ms
    
    print(f"  Average time: {kernel_time:.4f} ms")
    
    # Calculate speedup
    speedup = ref_time / kernel_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Calculate throughput
    num_elements = num_tokens * hidden_dim
    throughput_ref = num_elements / (ref_time / 1000) / 1e9  # GB/s (assuming 4 bytes per element)
    throughput_kernel = num_elements / (kernel_time / 1000) / 1e9
    
    print(f"\nThroughput:")
    print(f"  Reference: {throughput_ref * 4:.2f} GB/s")
    print(f"  Kernel:    {throughput_kernel * 4:.2f} GB/s")


def test_edge_cases(device: str = "cuda"):
    """Test edge cases like zero values, extreme values, etc."""
    print(f"\n{'='*60}")
    print(f"Testing Edge Cases")
    print(f"{'='*60}")
    
    if not KERNEL_AVAILABLE:
        print("Kernel not available, skipping edge case tests")
        return
    
    test_cases = [
        ("All zeros", torch.zeros(16, 128, dtype=torch.float16, device=device)),
        ("All ones", torch.ones(16, 128, dtype=torch.float16, device=device)),
        ("Large values", torch.randn(16, 128, dtype=torch.float16, device=device) * 1000),
        ("Small values", torch.randn(16, 128, dtype=torch.float16, device=device) * 0.001),
        ("Mixed signs", torch.randn(16, 128, dtype=torch.float16, device=device)),
    ]
    
    all_passed = True
    for name, input_tensor in test_cases:
        try:
            output, scale = cuda_reduce.per_token_quant_fp8(input_tensor)
            
            # Basic sanity checks
            assert not torch.isnan(output.to(torch.float32)).any(), f"NaN detected in output"
            assert not torch.isnan(scale).any(), f"NaN detected in scale"
            assert not torch.isinf(scale).any(), f"Inf detected in scale"
            
            print(f"✓ {name}: PASSED")
        except Exception as e:
            print(f"✗ {name}: FAILED - {str(e)}")
            all_passed = False
    
    return all_passed


def main():
    """Main test function"""
    print(f"\n{'#'*60}")
    print(f"# Per-Token FP8 Quantization Kernel Test Suite")
    print(f"{'#'*60}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    device = "cuda"
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Test different sizes
    test_configs = [
        (128, 4096),    # Small
        (512, 4096),    # Medium
        (1024, 4096),   # Large
        (2048, 4096),   # Very Large
    ]
    
    print(f"\n{'='*60}")
    print("ACCURACY TESTS - FLOAT16")
    print(f"{'='*60}")
    
    accuracy_results = []
    for num_tokens, hidden_dim in test_configs:
        result = test_accuracy(num_tokens, hidden_dim, device, dtype=torch.float16)
        accuracy_results.append(result)
    
    # exit(0)
    print(f"\n{'='*60}")
    print("ACCURACY TESTS - BFLOAT16")
    print(f"{'='*60}")
    
    accuracy_results_bf16 = []
    for num_tokens, hidden_dim in test_configs:
        result = test_accuracy(num_tokens, hidden_dim, device, dtype=torch.bfloat16)
        accuracy_results_bf16.append(result)
    
    print(f"\n{'='*60}")
    print("EDGE CASE TESTS")
    print(f"{'='*60}")
    edge_case_result = test_edge_cases(device)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARKS")
    print(f"{'='*60}")
    
    for num_tokens, hidden_dim in test_configs:
        benchmark_performance(num_tokens, hidden_dim, device=device)
    
    # Summary
    print(f"\n{'#'*60}")
    print("# TEST SUMMARY")
    print(f"{'#'*60}")
    
    if KERNEL_AVAILABLE:
        all_passed = all(accuracy_results) and all(accuracy_results_bf16) and edge_case_result
        if all_passed:
            print("\n✓ ALL TESTS PASSED")
        else:
            print("\n✗ SOME TESTS FAILED")
            if not all(accuracy_results):
                print("  - Float16 accuracy tests failed")
            if not all(accuracy_results_bf16):
                print("  - BFloat16 accuracy tests failed")
            if not edge_case_result:
                print("  - Edge case tests failed")
    else:
        print("\nKernel not available - only reference implementation tested")


if __name__ == "__main__":
    main()