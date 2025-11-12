import torch
import itertools
import tilelang as T
import tilelang.language as tl
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit
import tilelang.language.tir.op as ttop


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

def get_thread_template_configs():
    iter_params = dict(blk_m=[1, 32, 64, 128, 256], blk_k=[64, 128, 256, 512])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@T.autotune(
    configs=get_thread_template_configs(),
    warmup=3,
    rep=20,
)
@T.jit(out_idx=[-1])
def rms_norm_splitk_impl_tilelang(
    M: int,
    K: int,
    blk_m: int = None,      # need assign None when during autotune
    blk_k: int = None,      # need assign None when during autotune
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tl.prim_func
    def main(input_tensor: tl.Tensor((M, K), dtype),
             weight: tl.Tensor((K,), dtype),
             eps: tl.Tensor((1,), accum_dtype),
             output_tensor: tl.Tensor((M, K), dtype),
             ):
        with tl.Kernel(tl.ceildiv(M, blk_m), threads=128) as bx:
            input_tensor_shared = tl.alloc_shared((blk_m, blk_k), dtype)
            input_tensor_fp32_shared = tl.alloc_shared((blk_m, blk_k), accum_dtype)
            input_tensor_local = tl.alloc_fragment((blk_m, blk_k), accum_dtype)
            input_tensor_powsum = tl.alloc_fragment((blk_m,), accum_dtype)
            weight_local = tl.alloc_fragment((blk_k, ), dtype)

            num_k_step = tl.ceildiv(K, blk_k)
            tl.clear(input_tensor_local)
            for k in range(num_k_step):
                tl.copy(input_tensor[bx * blk_m, k * blk_k], input_tensor_shared)
                tl.copy(weight[k * blk_k], weight_local)
                for i, j in tl.Parallel(blk_m, blk_k):
                    input_tensor_fp32_shared[i, j] = input_tensor_shared[i, j]
                    input_tensor_local[i, j] += input_tensor_fp32_shared[i, j] * input_tensor_fp32_shared[i, j]
            tl.reduce_sum(input_tensor_local, input_tensor_powsum, dim=1)
            for i in tl.Parallel(blk_m):
                input_tensor_powsum[i] = tl.rsqrt(input_tensor_powsum[i] / K) + eps[0]

            for k in range(num_k_step):
                # reverse, better cache hit rate
                tl.copy(input_tensor[bx * blk_m, (num_k_step - 1 - k) * blk_k], input_tensor_shared)
                for i, j in tl.Parallel(blk_m, blk_k):
                    input_tensor_shared[i, j] *= input_tensor_powsum[i]
                    input_tensor_shared[i, j] *= weight_local[j]
                tl.copy(input_tensor_shared, output_tensor[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


@T.jit(out_idx=[-1])
def rms_norm_impl_tilelang(
    M: int,
    K: int,
    blk_m: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tl.prim_func
    def main(input_tensor: tl.Tensor((M, K), dtype),
             weight: tl.Tensor((K,), dtype),
             eps: tl.Tensor((1,), accum_dtype),
             output_tensor: tl.Tensor((M, K), dtype),
             ):
        with tl.Kernel(tl.ceildiv(M, blk_m), threads=128) as bx:
            input_tensor_shared = tl.alloc_shared((blk_m, K), dtype)
            input_tensor_pow_local = tl.alloc_fragment((blk_m, K), accum_dtype)
            input_tensor_local = tl.alloc_fragment((blk_m, K), accum_dtype)
            input_tensor_powsum = tl.alloc_fragment((blk_m,), dtype)
            weight_local = tl.alloc_fragment((K, ), dtype)

            tl.copy(input_tensor[bx * blk_m:(bx + 1) * blk_m, :], input_tensor_shared)
            tl.copy(input_tensor_shared, input_tensor_local)
            tl.copy(weight[:], weight_local)
            for i, j in tl.Parallel(blk_m, K):
                input_tensor_pow_local[i, j] = input_tensor_local[i, j] * input_tensor_local[i, j]
            tl.reduce_sum(input_tensor_pow_local, input_tensor_powsum, dim=1)
            for i in tl.Parallel(blk_m):
                input_tensor_powsum[i] = tl.rsqrt(input_tensor_powsum[i] / K) + eps[0]
            for i, j in tl.Parallel(blk_m, K):
                input_tensor_local[i, j] *= input_tensor_powsum[i]
                input_tensor_local[i, j] *= weight_local[j]
            tl.copy(input_tensor_local, output_tensor[bx * blk_m:(bx + 1) * blk_m, :])

    return main

def ref_rms_norm(input_tensor, weight, eps):
    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    input_tensor = input_tensor * torch.rsqrt(variance + eps)
    
    return weight * input_tensor.to(input_dtype)
    
if __name__ == "__main__":
    test_shapes = [
            (1, 4096),      # 方阵
            (4096, 4096),      # 大方阵
            (4096, 14336),     # 宽矩阵
            (14336, 4096),     # 高矩阵
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
        weight = torch.ones(out_features, device=device, dtype=torch.float16)  # 向量
        eps = torch.ones(1, device=device, dtype=torch.float) * 1e-5 # 向量
        output = torch.empty_like(x)
        # 调用kernel
        # rms_norm_func = rms_norm_impl_tilelang(M=in_features, K=out_features, blk_m=1)
        # output = rms_norm_func(x, weight, eps)

        rms_norm_func = rms_norm_splitk_impl_tilelang(M=in_features, K=out_features)
        output = rms_norm_func(x, weight, eps)
    
        print(output)
        
        ref_out = ref_rms_norm(x, weight, eps)
        print(ref_out)