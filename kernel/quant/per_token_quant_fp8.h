#pragma once

#include <torch/extension.h>

/**
 * Per-token (per-row) quantization to FP8 E4M3 format
 * 
 * For each token (row), this function:
 * 1. Finds the maximum absolute value across the hidden dimension
 * 2. Computes a per-token scale factor: scale = FP8_E4M3_MAX / max_abs_value
 * 3. Quantizes all elements in that row: output = clamp(input * scale, -448, 448)
 * 
 * @param input:  Input tensor [num_tokens, hidden_dim], dtype=float32 or bfloat16
 * @param output: Output tensor [num_tokens, hidden_dim], dtype=float8_e4m3fn
 * @param scale:  Scale tensor [num_tokens], dtype=float32
 * @param block_size: CUDA block size (default=256)
 */
void per_token_quant_fp8(
    torch::Tensor& input,    // [M, N] float32 or bfloat16
    torch::Tensor& output,   // [M, N] float8_e4m3fn
    torch::Tensor& scale,    // [M] float32
    int block_size = 256
);