import torch

import triton
import triton.language as tl
        

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_N": 64}, num_warps=1),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_warps=16),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def rms_norm_triton_impl(x_ptr,
                  w_ptr,
                  out_ptr,
                  stride_row: tl.constexpr,
                  eps: tl.constexpr,
                  M: tl.constexpr,
                  N: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  x_dtype: tl.constexpr):
    
    row = tl.program_id(axis=0)
    x_ptr += row * stride_row
    out_ptr += row * stride_row
    
    # 计算平方和
    pow_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for block_n_idx in range(tl.cdiv(N, BLOCK_N)):
        offset_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N
        
        x = tl.load(x_ptr + offset_n, mask=mask_n, other=0.0).to(tl.float32)
        pow_sum += x * x  # 平方累加
    
    # 计算 RMS
    variance = tl.sum(pow_sum) / N + eps
    rstd = tl.rsqrt(variance)
    
    # 归一化并应用权重
    for block_n_idx in range(tl.cdiv(N, BLOCK_N)):
        offset_n = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offset_n < N
        
        x = tl.load(x_ptr + offset_n, mask=mask_n, other=0.0).to(tl.float32)
        weight = tl.load(w_ptr + offset_n, mask=mask_n, other=1.0).to(tl.float32)
        
        # 计算归一化结果
        normalized = (x * rstd * weight).to(x_dtype)
        tl.store(out_ptr + offset_n, normalized, mask=mask_n)

def rms_norm_triton_kernel(x, weight, eps):
    in_features, out_features = x.shape
    assert weight.shape[0] == out_features
    
    output = torch.empty_like(x)
    x_dtype = tl.float16 if x.dtype in [torch.float16, torch.bfloat16] else tl.float32

    grid = lambda meta: (triton.cdiv(in_features, meta['BLOCK_N']), )
    rms_norm_triton_impl[grid](x,
                               weight,
                               output,
                               x.stride(0),
                               eps,
                               in_features,
                               out_features,
                               x_dtype=x_dtype)
    

if __name__ == "__main__":
    test_shapes = [
            (1, 4096),      # 方阵
            (4096, 4096),      # 大方阵
            # (4096, 14336),     # 宽矩阵
            # (14336, 4096),     # 高矩阵
        ]
        
    num_iterations = 100
    rtol, atol = 1e-3, 1e-3
    eps = 1e-5
    device = 'cuda'
    print(f"{'Shape':<15} {'Ref Time (ms)':<15} {'Custom Time (ms)':<15} {'Speedup':<10} {'Accuracy':<10}")
    print("-" * 75)


    for in_features, out_features in test_shapes:
        # 准备数据 - 注意权重应该是向量
        x = torch.randn(in_features, out_features, device=device, dtype=torch.float16)
        weight = torch.randn(out_features, device=device, dtype=torch.float16)  # 向量
        output = torch.empty_like(x)
        # 调用kernel
        grid = (in_features,)
        rms_norm_triton_impl[grid](
            x, weight, output, 
            x.stride(0), eps, 
            in_features, out_features, 
            128, tl.float16
        )

        print(output)
