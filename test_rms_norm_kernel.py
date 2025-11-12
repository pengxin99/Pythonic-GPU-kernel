import time
import torch

from rms_norm_triton import rms_norm_triton_kernel as rms_tt
from rms_norm_tilelang import rms_norm_splitk_impl_tilelang as rms_tl
'''
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
'''


def ref_rms_norm(input_tensor, weight, eps):
        input_dtype = input_tensor.dtype
        input_tensor = input_tensor.to(torch.float32)
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        input_tensor = input_tensor * torch.rsqrt(variance + eps)
        
        return weight * input_tensor.to(input_dtype)



def test_RMSNorm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on: {device}")
    
    # 测试配置
    test_shapes = [
        (1, 4096), 
        (4096, 4096), 
        (4096, 14336),
        (14336, 4096),
    ]
    
    num_iterations = 100
    rtol, atol = 1e-3, 1e-3
    eps = 1e-6
    eps_tensor = torch.ones(1, device=device) * eps     # for tilelang, input need an tensor

    print(f"{'Shape':<15} {'Ref Time (ms)':<15} {'Custom Time (ms)':<15} {'Speedup':<10} {'Accuracy':<10}")
    print("-" * 75)
    
    for in_features, out_features in test_shapes:
        # 准备数据
        x = torch.randn(in_features, out_features, device=device, dtype=torch.float16)
        weight = torch.ones(out_features, device=device, dtype=torch.float16)  # 向量
        output = torch.empty_like(x)

        ref_output = ref_rms_norm(x, weight, eps)

        # 性能测试 - 参考实现
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            ref_rms_norm(x, weight, eps)
        torch.cuda.synchronize() if device == 'cuda' else None
        ref_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
        
        
        if True:
            ###################  Triton kernel  ##########################
            # 自定义 kernel (临时用参考实现代替)
            output_tt = torch.empty_like(x)
            rms_tt(x, weight, eps)
            # print(custom_output)

            # 准确性检查
            accuracy = torch.allclose(ref_output, output_tt, rtol=rtol, atol=atol)
            # 性能测试 - 自定义 kernel
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(num_iterations):
                rms_tt(x, weight, eps)

            torch.cuda.synchronize() if device == 'cuda' else None
            custom_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
            
            speedup = ref_time / custom_time
            
            print(f"Triton {in_features}x{out_features:<8} {ref_time:<15.3f} {custom_time:<15.3f} {speedup:<10.2f} {str(accuracy):<10}")
            ###################  Triton kernel  ##########################


        if True:
            ###################  TileLang kernel  ##########################
            tilelang_RMSNorm_custom_kernel = rms_tl(M=in_features, K=out_features)      # for get_autotuned_kernel    
            # 自定义 kernel (临时用参考实现代替)
            custom_output = tilelang_RMSNorm_custom_kernel(x, weight, eps_tensor)
            # print(custom_output)

            # 准确性检查
            accuracy = torch.allclose(ref_output, custom_output, rtol=rtol, atol=atol)
            # 性能测试 - 自定义 kernel
            torch.cuda.synchronize() if device == 'cuda' else None
            
            start = time.time()
            for _ in range(num_iterations):
                tilelang_RMSNorm_custom_kernel(x, weight, eps_tensor)
            torch.cuda.synchronize() if device == 'cuda' else None
            custom_time = (time.time() - start) * 1000 / num_iterations  # ms per iteration
            ###################  TileLang kernel  ##########################
            
            speedup = ref_time / custom_time
            
            print(f"TileLang {in_features}x{out_features:<8} {ref_time:<15.3f} {custom_time:<15.3f} {speedup:<10.2f} {str(accuracy):<10}")


        
if __name__ == "__main__":
    test_RMSNorm()