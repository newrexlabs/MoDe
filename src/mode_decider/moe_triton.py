"""A grouped GEMM for MoE inference in Triton, for GPUs where torch's version syncs with the host.

``torch._grouped_mm`` has fast, graph-capturable kernels on sm_90 (H100/H200) and sm_100 only. Elsewhere, including the
RTX PRO 6000 (sm_120), it copies the group offsets to the host: about 1 ms per MoE layer, and it can't be captured in a
CUDA graph. ``grouped_mm(x, w, offs)`` computes the same thing with all bookkeeping on the device: ``x`` [T, K] with rows
sorted by expert, ``w`` [E, K, N], ``offs`` [E] cumulative row ends. The launch grid is a static upper bound from the
shapes, so it is capturable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_mm_kernel(x_ptr, w_ptr, y_ptr, offs_ptr, block_end_ptr, K, N, E,
                       stride_xt, stride_xk, stride_we, stride_wk, stride_wn, stride_yt, stride_yn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, E_POW2: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ids = tl.arange(0, E_POW2)
    block_end = tl.load(block_end_ptr + ids, mask=ids < E, other=2147483647)  # cumulative row blocks per expert
    total_blocks = tl.max(tl.where(ids < E, block_end, 0), axis=0)
    if pid_m >= total_blocks:
        return
    e = tl.sum((block_end <= pid_m).to(tl.int32), axis=0)  # expert owning this row block
    first_block = tl.where(e > 0, tl.load(block_end_ptr + e - 1, mask=e > 0, other=0), 0)
    row_start = tl.where(e > 0, tl.load(offs_ptr + e - 1, mask=e > 0, other=0), 0)
    row_end = tl.load(offs_ptr + e)
    rows = row_start + (pid_m - first_block) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < row_end
    col_ok = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * stride_we
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_ok = ks < K
        a = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xt + ks[None, :] * stride_xk,
                    mask=row_ok[:, None] & k_ok[None, :], other=0.0)
        b = tl.load(w_base + ks[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=k_ok[:, None] & col_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(y_ptr + rows[:, None].to(tl.int64) * stride_yt + cols[None, :] * stride_yn, acc.to(y_ptr.dtype.element_ty),
             mask=row_ok[:, None] & col_ok[None, :])


def _config(rows: int, experts: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps) by rows per expert: small blocks for interactive requests, larger for bulk."""
    per_expert = rows / max(experts, 1)
    if per_expert <= 16:
        return 16, 64, 64, 4
    if per_expert <= 64:
        return 32, 64, 64, 4
    if per_expert <= 256:
        return 64, 128, 64, 4
    return 128, 128, 64, 8


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, config: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    T, K = x.shape
    E, K2, N = w.shape
    assert K == K2 and offs.shape == (E,), (x.shape, w.shape, offs.shape)
    BM, BN, BK, warps = config or _config(T, E)
    y = torch.empty((T, N), device=x.device, dtype=x.dtype)
    if T == 0:
        return y
    offs = offs.to(torch.int32)
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    block_end = torch.cumsum((counts + BM - 1) // BM, 0, dtype=torch.int32)
    grid = (triton.cdiv(T, BM) + E, triton.cdiv(N, BN))  # static upper bound on row blocks: capturable
    _grouped_mm_kernel[grid](x, w, y, offs, block_end, K, N, E,
                             x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2), y.stride(0), y.stride(1),
                             BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, E_POW2=triton.next_power_of_2(E), num_warps=warps)
    return y
