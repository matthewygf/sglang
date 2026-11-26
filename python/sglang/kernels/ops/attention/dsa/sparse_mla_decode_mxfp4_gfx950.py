"""Sparse MLA decode on gfx950 over native MXFP4 KV, one query per row (a DCP rank's partial).

Same contract as `triton_sparse_mla_decode_splitk` with `return_lse=True` and `topk_length`;
Q's NoPE is quantized to MXFP8 for the kernel's scaled-MFMA QK dot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 import (
    _merge_kernel,
    _mxfp8_quant_kernel,
    num_splits_for,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

D_V, D_ROPE, ROW_BYTES = 512, 64, 400


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "sparse_mla_decode_mxfp4_gfx950",
        cuda_files=["attention/sparse_mla_decode_mxfp4_gfx950.cuh"],
        cuda_wrappers=[("run", "SparseMlaDecodeMxfp4Gfx950Kernel::run")],
        # aiter's opus headers
        extra_dependencies=["aiter"],
    )


def supports_sparse_mla_decode_mxfp4_gfx950(
    q_nope: torch.Tensor, kv: torch.Tensor, d_v: int
) -> bool:
    if (
        d_v != D_V
        or q_nope.ndim != 3
        or q_nope.shape[-1] != D_V
        or q_nope.stride(-1) != 1
        or q_nope.dtype != torch.bfloat16
        or not 0 < q_nope.shape[1] <= 128
        or q_nope.shape[1] % 16 != 0
        or kv.ndim != 3
        or kv.shape[1:] != (1, ROW_BYTES)
        or kv.dtype != torch.uint8
        or kv.device != q_nope.device
        or kv.stride(-1) != 1
        or not ROW_BYTES <= kv.stride(0) <= 0x7FFFFFFF
        or kv.stride(0) % 16 != 0
        or kv.data_ptr() % 16 != 0
        or (kv.size(0) - 1) * kv.stride(0) + ROW_BYTES > 0x7FFFFFFF
        or not torch.version.hip
        or q_nope.device.type != "cuda"
    ):
        return False
    return torch.cuda.get_device_properties(q_nope.device).gcnArchName.split(":")[0] == "gfx950"


def sparse_mla_decode_mxfp4_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor,
    expected_topk: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode BF16 Q [R,H,512] / RoPE [R,H,64] over uint8 KV [N,1,400].

    KV base and row stride are 16-byte aligned, with stride >=400. The writer
    guarantees finite RoPE and E8M0 exponents in [-30,30]. `indices` [R,1,topk] must have a prefix of
    valid local slots, counted by contiguous int32 `topk_length` [R]. The
    suffix is ignored. Returns ([1,R,H,512] BF16, [R,H] natural-log FP32 LSE).
    Empty rows return zero output and -inf LSE.
    """
    if not supports_sparse_mla_decode_mxfp4_gfx950(q_nope, kv, D_V):
        raise ValueError("Native MXFP4 decode requires gfx950 and supported Q/KV layouts")
    R, H, _ = q_nope.shape
    device = q_nope.device
    if (
        q_rope.shape != (R, H, D_ROPE)
        or q_rope.stride(-1) != 1
        or q_rope.dtype != torch.bfloat16
        or q_rope.device != device
    ):
        raise ValueError("Expected BF16 q_rope [R,H,64] with unit last stride")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (R, 1)
        or indices.dtype != torch.int32
        or indices.device != device
        or not indices.is_contiguous()
        or topk_length.ndim != 1
        or topk_length.numel() < R
        or topk_length.dtype != torch.int32
        or topk_length.device != device
        or not topk_length.is_contiguous()
    ):
        raise ValueError("Expected contiguous int32 indices [R,1,topk] and lengths [>=R]")
    idx = indices.squeeze(1)
    topk = idx.shape[1]
    if num_splits is None:
        num_splits = num_splits_for(R, min(expected_topk or topk, topk))
    S = num_splits
    if S < 1 or S & (S - 1):
        raise ValueError("num_splits must be a positive power of two")

    out = torch.empty(R, H, D_V, device=device, dtype=torch.bfloat16)
    lse = torch.empty(R, H, device=device, dtype=torch.float32)
    if R == 0:
        return out.unsqueeze(0), lse
    if topk == 0 or kv.shape[0] == 0:
        return out.zero_().unsqueeze(0), lse.fill_(-float("inf"))

    q8 = torch.empty(R, H, D_V, device=device, dtype=torch.float8_e4m3fn)
    qs = torch.empty(R, H, D_V // 32, device=device, dtype=torch.uint8)
    qr = torch.empty(R, H, D_ROPE, device=device, dtype=torch.bfloat16)
    _mxfp8_quant_kernel[(R, triton.cdiv(H, 16))](
        q_nope,
        q8,
        qs,
        q_rope,
        qr,
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        H,
        HB=16,
    )
    slots = R * S if S > 1 else 1
    o_acc = torch.empty(slots, H, D_V, device=device, dtype=torch.float32)
    lse_acc = torch.empty(slots, H, device=device, dtype=torch.float32)
    _jit_module().run(
        q8,
        qs,
        qr,
        kv.squeeze(1),
        idx,
        topk_length,
        o_acc,
        lse_acc,
        out,
        lse,
        float(sm_scale),
        S,
    )
    if S > 1:
        _merge_kernel[(R * H,)](o_acc, lse_acc, out, lse, H, S=S)
    return out.unsqueeze(0), lse
