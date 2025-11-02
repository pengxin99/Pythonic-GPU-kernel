import torch
import triton
import triton.language as tl


# 优化的版本，针对不同尺寸自动调整块大小
'''
num_warps – the number of warps to use for the kernel when compiled for GPUs. 
            For example, if num_warps=8, then each kernel instance will be automatically 
            parallelized to cooperatively execute using 8 * 32 = 256 threads.
'''
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 256}, num_warps=8),
    ],
    key=['M', 'K'],
    warmup=3, 
    rep=10,
)
@triton.jit
def gemv_kernel(
    x_ptr, w_ptr, y_ptr,
    M, K,
    stride_wm, stride_wk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取程序 id
    pid_m = tl.program_id(axis=0)
    
    # 创建掩码
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = rm < M
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # K维度进行循环处理
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + rk
        mask_k = k_offsets < K
        
        # 加载权重矩阵的块 [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # 地址[i,j] = w_ptr + rm[i] * stride_wm + k_offsets[j] * stride_wk
        w_block_ptrs = w_ptr + rm[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_block = tl.load(w_block_ptrs, mask = mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # 加载 tensor x
        x_ptrs = x_ptr + k_offsets
        x_block = tl.load(x_ptrs, mask=mask_k, other=0.0, cache_modifier='.ca')
        
        # 矩阵向量乘法累加
        acc += tl.sum(x_block * w_block, axis=1)
        
    y_ptrs = y_ptr + rm
    tl.store(y_ptrs, acc, mask=mask_m)

def triton_gemv(
    x: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    """
    Triton GEMV 实现
    参数:
        x: [K] 输入向量
        weight: [M, K] 权重矩阵
    返回:
        y: [M] 输出向量
    """
    assert x.dim() == 1, "输入x必须是1D向量"
    assert weight.dim() == 2, "权重weight必须是2D矩阵"
    K = x.shape[0]
    M = weight.shape[0]
    assert weight.shape[1] == K, "权重矩阵的列数必须等于输入向量的长度"

    # 分配输出
    y = torch.empty((1, M), device=x.device, dtype=x.dtype)
    
    # 选择块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 128
    
    # for no autotune
    # 计算网格大小， 这里需要是二维tuple
    # grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    #     gemv_kernel[grid](
    #     x, weight, y,
    #     M, K,
    #     weight.stride(0), weight.stride(1),
    #     BLOCK_SIZE_M=BLOCK_SIZE_M,
    #     BLOCK_SIZE_K=BLOCK_SIZE_K
    # )
    
    # for autotune
    # 计算网格大小， 这里需要是二维tuple
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )
    # 启动kernel
    gemv_kernel[grid](
        x, weight, y,
        M, K,
        weight.stride(0), weight.stride(1),
    )
    
    return y
    