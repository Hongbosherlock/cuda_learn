import torch

# 导入编译好的CUDA扩展
try:
    import cuda_reduce
except ImportError:
    print("Error: cuda_reduce module not found. Please build the extension first:")
    print("  cd cuda_learn && python3 setup.py install")
    raise SystemExit(1)


def _supported_dtypes():
    dtypes = [torch.float32]
    if torch.cuda.is_available():
        dtypes.append(torch.float16)
        # bfloat16 需要较新架构支持
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
    return dtypes


def test_online_softmax_accuracy():
    """最小数值对齐测试：cuda_reduce.softmax_online vs torch.softmax"""
    print("=" * 60)
    print("Online Softmax 数值对齐测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return False

    test_shapes = [
        (1, 128),
        (4, 256),
        (8, 1024),
        (16, 4096),
    ]

    all_passed = True

    for dtype in _supported_dtypes():
        print(f"\n测试 dtype={dtype}")
        for rows, cols in test_shapes:
            torch.manual_seed(42)
            x = torch.randn(rows, cols, device="cuda", dtype=dtype)

            # CUDA kernel
            y = cuda_reduce.softmax_online(x)

            # PyTorch reference (use float32 accumulator)
            ref = torch.softmax(x.float(), dim=-1).to(dtype)

            # 容忍度：根据 dtype 调整
            if dtype == torch.float32:
                atol, rtol = 1e-4, 1e-4
            elif dtype == torch.float16:
                atol, rtol = 5e-3, 5e-3
            else:  # bfloat16
                atol, rtol = 1e-2, 1e-2

            try:
                torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
                status = "✓ PASS"
                passed = True
            except AssertionError:
                status = "✗ FAIL"
                passed = False

            all_passed = all_passed and passed
            print(f"  shape=({rows}, {cols}) {status}")

    if all_passed:
        print("\n✓ 所有数值对齐测试通过！")
    else:
        print("\n✗ 部分测试失败")

    return all_passed


if __name__ == "__main__":
    test_online_softmax_accuracy()
