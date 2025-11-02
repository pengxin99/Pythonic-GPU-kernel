import torch
import itertools
import tilelang as T
import tilelang.language as tl
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit

@T.jit(out_idx=[-1])
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float",
    accum_dtype: str = "float",
):

    @tl.prim_func
    def main(
            A: tl.Tensor((K,), dtype),
            B: tl.Tensor((N, K), dtype),
            C: tl.Tensor((N,), dtype),
    ):
        with tl.Kernel(tl.ceildiv(N, BLOCK_N)) as bn:
            tn = tl.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = tl.alloc_shared((BLOCK_K,), dtype)
            B_shared = tl.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = tl.alloc_local((1,), accum_dtype)
            tl.clear(C_reg)
            for bk in tl.serial(tl.ceildiv(K, BLOCK_K)):
                for tk in tl.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in tl.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn,
                                                                            tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main



@T.jit(out_idx=[-1])
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float",
    accum_dtype: str = "float",
):
    @tl.prim_func
    def main(
            A: tl.Tensor((K,), dtype),
            B: tl.Tensor((N, K), dtype),
            C: tl.Tensor((N,), dtype),
    ):
        with tl.Kernel(tl.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = tl.get_thread_binding(0)
            tk = tl.get_thread_binding(1)
            A_local = tl.alloc_local((1,), dtype)
            B_local = tl.alloc_local((1,), dtype)
            C_accum = tl.alloc_local((1,), accum_dtype)
            C_shared = tl.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            tl.clear(C_accum)
            for bk in tl.serial(tl.ceildiv(K, BLOCK_K)):
                A_local[0] = A[bk * BLOCK_K + tk]
                B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
            tl.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


@T.jit(out_idx=[-1])
def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @tl.prim_func
    def main(
            A: tl.Tensor((K,), dtype),
            B: tl.Tensor((N, K), dtype),
            C: tl.Tensor((N,), dtype),
    ):
        with tl.Kernel(tl.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = tl.get_thread_binding(0)
            tk = tl.get_thread_binding(1)
            A_local = tl.alloc_local((TILE_K,), dtype)
            B_local = tl.alloc_local((TILE_K,), dtype)
            C_shared = tl.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = tl.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            tl.clear(C_accum)
            for bk in tl.serial(tl.ceildiv(K, BLOCK_K)):
                for k in tl.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in tl.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            tl.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


def get_thread_template_configs():
    iter_params = dict(BLOCK_N=[2, 4, 8, 32, 64, 128], reduce_threads=[4, 8, 32])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@T.autotune(
    configs=get_thread_template_configs(),
    warmup=3,
    rep=20,
)
@T.jit(
    out_idx=[-1],
    target="auto",
)
def get_autotuned_kernel(
    N,
    K,
    BLOCK_N=None,
    reduce_threads=None,
):
    dtype = "float"
    accum_dtype = "float"
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @tl.prim_func
    def main(
            A: tl.Tensor((K,), dtype),
            B: tl.Tensor((N, K), dtype),
            C: tl.Tensor((N,), dtype),
    ):
        with tl.Kernel(tl.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = tl.get_thread_binding(0)
            tk = tl.get_thread_binding(1)
            A_local = tl.alloc_local((TILE_K,), dtype)
            B_local = tl.alloc_local((TILE_K,), dtype)
            C_accum = tl.alloc_local((1,), accum_dtype)

            tl.clear(C_accum)
            for bk in tl.serial(tl.ceildiv(K, BLOCK_K)):
                for k in tl.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in tl.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = tl.alloc_local((1,), accum_dtype)
            with tl.attr(
                    tl.comm_reducer(lambda x, y: x + y, [tl.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    tl.reinterpret(tl.uint64(0), dtype="handle"),
            ):
                tl.evaluate(
                    tl.tvm_thread_allreduce(
                        tl.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main


if __name__ == "__main__":
    # 假设的输入数据
    N, K = 1024, 1024  # 矩阵维度
    W = torch.randn(N, K, device='cuda', dtype=torch.float32)
    x = torch.randn(K, device='cuda', dtype=torch.float32)
    y = torch.zeros(N, device='cuda', dtype=torch.float32)

    # 设置分块大小
    block_N, block_K = 4, 8
    # 创建kernel函数
    # gemv_func = naive_gemv(N=N, K=K, BLOCK_N=block_N, BLOCK_K=block_K)
    # gemv_func = naive_splitk_gemv(N=N, K=K, BLOCK_N=block_N, BLOCK_K=block_K)
    gemv_func = splitk_gemv_vectorized(N=N, K=K, BLOCK_N=block_N, reduce_threads=256)
    gemv_func = get_autotuned_kernel(N=N, K=K)

    print(x.shape, W.shape)
    # 执行kernel (实际使用时需要编译和参数绑定)
    print(gemv_func(x, W))
    print(x @ W.T)