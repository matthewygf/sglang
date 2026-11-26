"""Native gfx950 DSA prefill for BF16 queries and packed MXFP4 KV rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ROW_BYTES = 400
_MAX_OFFSET = 0x7FFFFFFF


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "sparse_mla_prefill_mxfp4_gfx950",
        cuda_files=["attention/sparse_mla_prefill_mxfp4_gfx950.cuh"],
        cuda_wrappers=[
            ("run", "SparseMlaPrefillMxfp4Gfx950Kernel::run"),
            ("run_split", "SparseMlaPrefillMxfp4Gfx950Kernel::run_split"),
        ],
        extra_dependencies=["aiter"],
    )


def supports_sparse_mla_prefill_mxfp4_gfx950(
    q_nope: torch.Tensor, q_rope: torch.Tensor, kv: torch.Tensor, d_v: int
) -> bool:
    """Check the layout contract; the caller gates on gfx950 separately."""
    return (
        d_v == 512
        and q_nope.dtype == q_rope.dtype == torch.bfloat16
        and q_nope.device == q_rope.device == kv.device
        and q_nope.ndim == 3
        and q_nope.shape[2] == d_v
        and 1 <= q_nope.shape[1] <= 16
        and q_rope.shape == (*q_nope.shape[:2], 64)
        and q_nope.stride(-1) == q_rope.stride(-1) == 1
        and all(s % 8 == 0 for s in (*q_nope.stride()[:2], *q_rope.stride()[:2]))
        and q_nope.data_ptr() % 16 == q_rope.data_ptr() % 16 == 0
        and kv.dtype == torch.uint8
        and kv.shape[1:] == (1, _ROW_BYTES)
        and kv.stride(-1) == 1
        and _ROW_BYTES <= kv.stride(0) <= _MAX_OFFSET
        and kv.stride(0) % 16 == 0
        and kv.data_ptr() % 16 == 0
        and kv.shape[0] * kv.stride(0) <= _MAX_OFFSET
    )


def _prepare(q_nope, q_rope, kv, indices, d_v):
    if not q_nope.is_cuda or not supports_sparse_mla_prefill_mxfp4_gfx950(
        q_nope, q_rope, kv, d_v
    ):
        raise ValueError("Unsupported native MXFP4 prefill query or KV layout")
    seq, heads, _ = q_nope.shape
    idx = indices.squeeze(1) if indices.ndim == 3 else indices
    if (
        idx.ndim != 2
        or idx.shape[0] != seq
        or idx.shape[1] == 0
        or idx.device != q_nope.device
    ):
        raise ValueError("Expected indices [T, 1, topk] or [T, topk] on query device")
    idx = idx.to(torch.int32).contiguous()
    out = torch.empty((seq, heads, d_v), device=q_nope.device, dtype=torch.bfloat16)
    lse = torch.empty((seq, heads), device=q_nope.device, dtype=torch.float32)
    return idx, out, lse


def sparse_mla_prefill_mxfp4_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return [1,T,H,512] BF16 and optionally [T,H] FP32 natural-log LSE.

    The uint8 KV view is [N,1,400], possibly with padded physical rows.
    Indices [T,1,topk] or [T,topk] use -1 for masked keys.
    """
    idx, out, lse = _prepare(q_nope, q_rope, kv, indices, d_v)
    _jit_module().run(q_nope, q_rope, kv, idx, out, lse, float(sm_scale), int(return_lse))
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)


def sparse_mla_split_mxfp4_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    splits: int,
    d_v: int = 512,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Native MXFP4 attention with split-key CTAs and FP32 partial combine."""
    if not 1 <= splits <= 65535:
        raise ValueError("splits must fit grid.y")
    idx, out, lse = _prepare(q_nope, q_rope, kv, indices, d_v)
    seq, heads, _ = q_nope.shape
    part_acc = torch.empty(
        (seq, splits, heads, d_v), device=q_nope.device, dtype=torch.float32
    )
    part_ml = torch.empty(
        (2, seq, splits, heads), device=q_nope.device, dtype=torch.float32
    )
    _jit_module().run_split(
        q_nope, q_rope, kv, idx, out, lse, part_acc, part_ml,
        float(sm_scale), int(return_lse), splits,
    )
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)
