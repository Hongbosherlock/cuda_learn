"""
Flash Attention V2 三种实现对比分析

本文件比较以下三种 Flash Attention V2 的 PyTorch 实现：
1. flash_attention_v2_impl.py  ->  online_softmax_attention
2. gpt_flash2.py               ->  flash_attn2_forward_pseudo
3. gemini_flash2.py            ->  flash_attention_v2_simulation

============================================================================
实现对比总结
============================================================================

| 特性                | online_softmax_attention | flash_attn2_forward_pseudo | flash_attention_v2_simulation |
|---------------------|--------------------------|----------------------------|-------------------------------|
| 循环结构            | 外Q内KV (向量化batch/heads) | 外batch/heads/Q内KV        | 外Q内KV (向量化batch/heads)   |
| 因果掩码            | ✓ 支持                   | ✓ 支持                     | ✗ 不支持                      |
| 数据类型            | 保持输入dtype            | 内部用fp32累加             | 保持输入dtype                 |
| 统计量存储          | 块级 (m_i, l_i)          | 行级 (m, l, acc)           | 全局 (m, l, o)                |
| O累积方式           | A = α*A + β@V            | acc = acc*α + exp@V        | o = o*corr + exp@V            |
| 内存效率            | 高 (块级统计量)          | 中 (行级统计量)            | 低 (全局统计量)               |

============================================================================
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple, Dict, List

# ============================================================================
# 导入三种实现
# ============================================================================

# 实现1: online_softmax_attention (from flash_attention_v2_impl.py)
def online_softmax_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal: bool = False,
    block_size_q: int = 64,
    block_size_kv: int = 64,
) -> torch.Tensor:
    """实现1: 外层Q内层KV，向量化处理batch和heads"""
    batch, heads, seq_q, head_dim = Q.shape
    seq_kv = K.shape[2]
    
    device = Q.device
    dtype = Q.dtype
    
    num_blocks_q = math.ceil(seq_q / block_size_q)
    num_blocks_kv = math.ceil(seq_kv / block_size_kv)
    
    O = torch.zeros_like(Q)
    
    for q_block_idx in range(num_blocks_q):
        q_start = q_block_idx * block_size_q
        q_end = min(q_start + block_size_q, seq_q)
        actual_block_q = q_end - q_start
        
        Q_block = Q[:, :, q_start:q_end, :]
        
        # 块级统计量初始化
        m_i = torch.full((batch, heads, actual_block_q, 1), float('-inf'), device=device, dtype=dtype)
        l_i = torch.zeros((batch, heads, actual_block_q, 1), device=device, dtype=dtype)
        O_block = torch.zeros((batch, heads, actual_block_q, head_dim), device=device, dtype=dtype)
        
        for kv_block_idx in range(num_blocks_kv):
            kv_start = kv_block_idx * block_size_kv
            kv_end = min(kv_start + block_size_kv, seq_kv)
            
            if causal and kv_start > q_end - 1:
                continue
            
            K_block = K[:, :, kv_start:kv_end, :]
            V_block = V[:, :, kv_start:kv_end, :]
            
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
            
            if causal:
                q_positions = torch.arange(q_start, q_end, device=device).view(-1, 1)
                kv_positions = torch.arange(kv_start, kv_end, device=device).view(1, -1)
                causal_mask = q_positions >= kv_positions
                S_block = S_block.masked_fill(~causal_mask, float('-inf'))
            
            m_block = S_block.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_block)
            
            alpha = torch.exp(m_i - m_new)
            beta = torch.exp(S_block - m_new)
            
            l_new = alpha * l_i + beta.sum(dim=-1, keepdim=True)
            O_block = alpha * O_block + torch.matmul(beta, V_block)
            
            m_i = m_new
            l_i = l_new
        
        l_i = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
        O_block = O_block / l_i
        O[:, :, q_start:q_end, :] = O_block
    
    return O


# 实现2: flash_attn2_forward_pseudo (from gpt_flash2.py)
def flash_attn2_forward_pseudo(q, k, v, causal: bool = False, block_q: int = 128, block_k: int = 128):
    """实现2: 四层循环 (batch, heads, Q块, KV块)，内部fp32累加"""
    B, H, T, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)
    
    o = torch.empty((B, H, T, Dh), device=q.device, dtype=q.dtype)
    
    for b in range(B):
        for h in range(H):
            for i0 in range(0, T, block_q):
                i1 = min(i0 + block_q, T)
                q_blk = q[b, h, i0:i1, :]
                
                # 行级统计量 (fp32)
                m = torch.full((i1 - i0,), -float("inf"), device=q.device, dtype=torch.float32)
                l = torch.zeros((i1 - i0,), device=q.device, dtype=torch.float32)
                acc = torch.zeros((i1 - i0, Dh), device=q.device, dtype=torch.float32)
                
                for j0 in range(0, T, block_k):
                    j1 = min(j0 + block_k, T)
                    k_blk = k[b, h, j0:j1, :]
                    v_blk = v[b, h, j0:j1, :]
                    
                    scores = (q_blk.float() @ k_blk.float().T) * scale
                    
                    if causal:
                        i_idx = torch.arange(i0, i1, device=q.device).view(-1, 1)
                        j_idx = torch.arange(j0, j1, device=q.device).view(1, -1)
                        scores = scores.masked_fill(j_idx > i_idx, -float("inf"))
                    
                    block_m = scores.max(dim=-1).values
                    m_new = torch.maximum(m, block_m)
                    
                    exp_scores = torch.exp(scores - m_new.view(-1, 1))
                    alpha = torch.exp(m - m_new)
                    
                    l = l * alpha + exp_scores.sum(dim=-1)
                    acc = acc * alpha.view(-1, 1) + exp_scores @ v_blk.float()
                    m = m_new
                
                o_blk = acc / l.view(-1, 1)
                o[b, h, i0:i1, :] = o_blk.to(q.dtype)
    
    return o


# 实现3: flash_attention_v2_simulation (from gemini_flash2.py)
def flash_attention_v2_simulation(q, k, v, block_size_r=64, block_size_c=64):
    """实现3: 外Q内KV，使用全局统计量张量"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    o = torch.zeros_like(q)
    
    # 全局统计量 (整个序列)
    l = torch.zeros((batch_size, num_heads, seq_len, 1), device=q.device)
    m = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=q.device)
    
    for i in range(0, seq_len, block_size_r):
        q_block = q[:, :, i:i+block_size_r, :]
        
        m_block = m[:, :, i:i+block_size_r, :]
        l_block = l[:, :, i:i+block_size_r, :]
        o_block = o[:, :, i:i+block_size_r, :]
        
        for j in range(0, seq_len, block_size_c):
            k_block = k[:, :, j:j+block_size_c, :]
            v_block = v[:, :, j:j+block_size_c, :]
            
            attn_score = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            block_max = torch.max(attn_score, dim=-1, keepdim=True)[0]
            m_new = torch.maximum(m_block, block_max)
            
            exp_score = torch.exp(attn_score - m_new)
            correction = torch.exp(m_block - m_new)
            
            o_inter = o_block * correction
            o_current = torch.matmul(exp_score, v_block)
            o_block = o_inter + o_current
            
            l_inter = l_block * correction
            l_current = torch.sum(exp_score, dim=-1, keepdim=True)
            l_block = l_inter + l_current
            
            m_block = m_new
        
        o_block = o_block / l_block
        o[:, :, i:i+block_size_r, :] = o_block
    
    return o


# ============================================================================
# 参考实现：标准注意力
# ============================================================================

def standard_attention(Q, K, V, causal=False):
    """标准注意力实现（用于验证正确性）"""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = Q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


# ============================================================================
# 测试函数
# ============================================================================

def test_correctness(verbose=True):
    """测试三种实现的数值正确性"""
    print("=" * 80)
    print("数值正确性测试")
    print("=" * 80)
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # 测试配置
    configs = [
        {"batch": 2, "heads": 4, "seq_len": 64, "head_dim": 32, "causal": False, "name": "小规模非因果"},
        {"batch": 2, "heads": 4, "seq_len": 128, "head_dim": 64, "causal": False, "name": "中等规模非因果"},
        {"batch": 1, "heads": 8, "seq_len": 256, "head_dim": 64, "causal": True, "name": "因果注意力"},
    ]
    
    results = []
    
    for cfg in configs:
        batch = cfg["batch"]
        heads = cfg["heads"]
        seq_len = cfg["seq_len"]
        head_dim = cfg["head_dim"]
        causal = cfg["causal"]
        name = cfg["name"]
        
        print(f"测试: {name}")
        print(f"  配置: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}, causal={causal}")
        
        # 生成输入
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        # 参考输出
        ref_output = standard_attention(Q, K, V, causal=causal)
        
        # 测试三种实现
        implementations = [
            ("online_softmax_attention", 
             lambda: online_softmax_attention(Q, K, V, scale, causal=causal, block_size_q=32, block_size_kv=32)),
            ("flash_attn2_forward_pseudo", 
             lambda: flash_attn2_forward_pseudo(Q, K, V, causal=causal, block_q=32, block_k=32)),
        ]
        
        # gemini实现不支持causal，单独处理
        if not causal:
            implementations.append(
                ("flash_attention_v2_simulation", 
                 lambda: flash_attention_v2_simulation(Q, K, V, block_size_r=32, block_size_c=32))
            )
        
        for impl_name, impl_fn in implementations:
            output = impl_fn()
            max_diff = (ref_output - output).abs().max().item()
            mean_diff = (ref_output - output).abs().mean().item()
            
            status = "✓ 通过" if max_diff < 1e-4 else "✗ 失败"
            print(f"  {impl_name:35s}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {status}")
            
            results.append({
                "config": name,
                "implementation": impl_name,
                "max_diff": max_diff,
                "passed": max_diff < 1e-4
            })
        
        if causal:
            print(f"  {'flash_attention_v2_simulation':35s}: (不支持因果掩码，跳过)")
        
        print()
    
    return results


def test_performance():
    """性能对比测试"""
    print("=" * 80)
    print("性能对比测试")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("需要 CUDA 进行性能测试，跳过...\n")
        return
    
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    configs = [
        {"seq_len": 128, "desc": "短序列"},
        {"seq_len": 256, "desc": "中等序列"},
        {"seq_len": 512, "desc": "长序列"},
    ]
    
    batch = 2
    heads = 8
    head_dim = 64
    n_warmup = 5
    n_runs = 20
    
    print(f"\n配置: batch={batch}, heads={heads}, head_dim={head_dim}")
    print(f"预热: {n_warmup} 次, 测试: {n_runs} 次\n")
    
    for cfg in configs:
        seq_len = cfg["seq_len"]
        desc = cfg["desc"]
        
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        implementations = [
            ("standard_attention", 
             lambda: standard_attention(Q, K, V, causal=False)),
            ("online_softmax_attention", 
             lambda: online_softmax_attention(Q, K, V, scale, causal=False)),
            ("flash_attn2_forward_pseudo", 
             lambda: flash_attn2_forward_pseudo(Q, K, V, causal=False)),
            ("flash_attention_v2_simulation", 
             lambda: flash_attention_v2_simulation(Q, K, V)),
        ]
        
        print(f"{desc} (seq_len={seq_len}):")
        
        for impl_name, impl_fn in implementations:
            # 预热
            for _ in range(n_warmup):
                _ = impl_fn()
            torch.cuda.synchronize()
            
            # 计时
            start = time.perf_counter()
            for _ in range(n_runs):
                _ = impl_fn()
            torch.cuda.synchronize()
            avg_time = (time.perf_counter() - start) / n_runs * 1000
            
            print(f"  {impl_name:35s}: {avg_time:8.3f} ms")
        
        print()


def analyze_implementations():
    """详细分析三种实现的差异"""
    print("=" * 80)
    print("实现差异详细分析")
    print("=" * 80)
    
    analysis = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        三种实现的核心差异分析                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. online_softmax_attention (flash_attention_v2_impl.py)                   │
│     ─────────────────────────────────────────────────────                   │
│     循环结构: for q_block in Q_blocks:                                       │
│                  for kv_block in KV_blocks:                                 │
│     特点:                                                                    │
│     - 外层Q内层KV，向量化处理 batch 和 heads 维度                            │
│     - 块级统计量 (m_i, l_i)，每个Q块独立维护                                 │
│     - 支持因果掩码，使用位置索引生成 mask                                    │
│     - O累积公式: O = α * O + β @ V                                          │
│                                                                             │
│  2. flash_attn2_forward_pseudo (gpt_flash2.py)                              │
│     ─────────────────────────────────────────────────────                   │
│     循环结构: for b in batch:                                                │
│                  for h in heads:                                            │
│                      for q_block in Q_blocks:                               │
│                          for kv_block in KV_blocks:                         │
│     特点:                                                                    │
│     - 四层显式循环，更接近硬件实现的逻辑                                     │
│     - 内部使用 fp32 累加，提高数值稳定性                                     │
│     - 行级统计量 (m, l, acc)，形状为 [block_q,]                             │
│     - 支持因果掩码                                                           │
│     - acc累积公式: acc = acc * α + exp_scores @ V                           │
│                                                                             │
│  3. flash_attention_v2_simulation (gemini_flash2.py)                        │
│     ─────────────────────────────────────────────────────                   │
│     循环结构: for q_block in Q_blocks:                                       │
│                  for kv_block in KV_blocks:                                 │
│     特点:                                                                    │
│     - 外层Q内层KV，向量化处理 batch 和 heads                                 │
│     - 全局统计量张量 (m, l)，形状为 [batch, heads, seq_len, 1]              │
│     - 不支持因果掩码                                                         │
│     - O累积公式: o = o * correction + exp @ V                               │
│     - 通过切片访问全局张量的对应块                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              数学公式对比                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  在线 Softmax 更新公式 (三种实现共同遵循):                                   │
│                                                                             │
│    m_new = max(m_old, max(S_block))           # 更新最大值                  │
│    α = exp(m_old - m_new)                     # 修正因子                    │
│    β = exp(S_block - m_new)                   # 当前块指数                  │
│    l_new = α * l_old + sum(β)                 # 更新归一化因子              │
│    O_new = α * O_old + β @ V                  # 更新输出累积                │
│                                                                             │
│  最终: O = O_new / l_new                                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              内存效率对比                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  实现                          统计量内存占用                                │
│  ────────────────────────────────────────────────────                       │
│  online_softmax_attention      O(block_q)      - 块级，最优                 │
│  flash_attn2_forward_pseudo    O(block_q)      - 行级，最优                 │
│  flash_attention_v2_simulation O(seq_len)      - 全局，较差                 │
│                                                                             │
│  注: 真正的 CUDA 实现应使用块级/行级统计量，避免全局张量                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(analysis)


def main():
    """运行所有测试和分析"""
    print("\n" + "=" * 80)
    print("Flash Attention V2 实现对比分析")
    print("=" * 80 + "\n")
    
    # 1. 分析实现差异
    analyze_implementations()
    
    # 2. 数值正确性测试
    test_correctness()
    
    # 3. 性能对比
    test_performance()
    
    print("=" * 80)
    print("所有测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()