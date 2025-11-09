import torch
import time

# 导入你的自定义 GEMV kernel
from gemv_triton import triton_gemv
triton_gemv_custom = triton_gemv

from gemv_tilelang import naive_gemv, naive_splitk_gemv, splitk_gemv_vectorized, get_autotuned_kernel
tilelang_gemv_custom = get_autotuned_kernel

torch.manual_seed(0)

def ref_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现的 GEMV"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # output = torch.matmul(x, weight.t())
    output = x @ weight.T
    if output.size(0) == 1:
        output = output.squeeze(0)
    return output

def test_gemv():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on: {device}")
    
    # 测试配置
    test_shapes = [
        (1024, 1024),      # 方阵
        (4096, 4096),      # 大方阵
        (4096, 14336),     # 宽矩阵
        (14336, 4096),     # 高矩阵
    ]
    
    num_iterations = 100
    rtol, atol = 1e-3, 1e-3
    
    print(f"{'Shape':<15} {'Ref Time (ms)':<15} {'Custom Time (ms)':<15} {'Speedup':<10} {'Accuracy':<10}")
    print("-" * 75)
    
    for in_features, out_features in test_shapes:
        # 准备数据
        x = torch.randn(in_features, device=device)
        weight = torch.randn(out_features, in_features, device=device)
        
        # 参考实现
        ref_output = ref_gemv(x, weight)
        # print(ref_output)
        

        # 性能测试 - 参考实现
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            ref_gemv(x, weight)
        torch.cuda.synchronize() if device == 'cuda' else None
        ref_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
        
        
        ###################  Triton kernel  ##########################
        # 自定义 kernel (临时用参考实现代替)
        custom_output = triton_gemv_custom(x, weight)  # 替换为: custom_gemv(x, weight)
        # print(custom_output)

        # 准确性检查
        accuracy = torch.allclose(ref_output, custom_output, rtol=rtol, atol=atol)
        # 性能测试 - 自定义 kernel
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            triton_gemv_custom(x, weight)  # 替换为: custom_gemv(x, weight)
        torch.cuda.synchronize() if device == 'cuda' else None
        custom_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
        
        speedup = ref_time / custom_time
        
        print(f"Triton {in_features}x{out_features:<8} {ref_time:<15.3f} {custom_time:<15.3f} {speedup:<10.2f} {str(accuracy):<10}")
        ###################  Triton kernel  ##########################



        ###################  TileLang kernel  ##########################
        # tilelang_gemv_custom_kernel = tilelang_gemv_custom(N=out_features, K=in_features, BLOCK_N=128, BLOCK_L=128)       # for naive_gemv 
        # tilelang_gemv_custom_kernel = tilelang_gemv_custom(N=out_features, K=in_features, BLOCK_N=4, reduce_threads=128)  # for splitk_gemv_vectorized
        tilelang_gemv_custom_kernel = tilelang_gemv_custom(N=out_features, K=in_features)                                   # for get_autotuned_kernel    
        # 自定义 kernel (临时用参考实现代替)
        custom_output = tilelang_gemv_custom_kernel(x, weight)  # 替换为: custom_gemv(x, weight)
        # print(custom_output)

        # 准确性检查
        accuracy = torch.allclose(ref_output, custom_output, rtol=rtol, atol=atol)
        # 性能测试 - 自定义 kernel
        torch.cuda.synchronize() if device == 'cuda' else None
        
        start = time.time()
        for _ in range(num_iterations):
            tilelang_gemv_custom_kernel(x, weight)  # 替换为: custom_gemv(x, weight)
        torch.cuda.synchronize() if device == 'cuda' else None
        custom_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
        ###################  TileLang kernel  ##########################
        
        speedup = ref_time / custom_time
        
        print(f"TileLang {in_features}x{out_features:<8} {ref_time:<15.3f} {custom_time:<15.3f} {speedup:<10.2f} {str(accuracy):<10}")


if __name__ == "__main__":
    test_gemv()