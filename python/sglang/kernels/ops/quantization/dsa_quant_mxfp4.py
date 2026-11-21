# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MXFP4 recipe and E2M1 thresholds adapted from:
# https://github.com/vllm-project/vllm/blob/0599e4ba72fd29265a0206b29a0d54b308dddcb8/vllm/v1/attention/ops/mxfp4_mla/store.py
"""Original, bounded native DSA MXFP4 scatter; no alternate codec recipes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_mxfp4_rows(
    X,
    R,
    Slots,
    Cache,
    XS: tl.constexpr,
    XC: tl.constexpr,
    RS: tl.constexpr,
    RC: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    CAP: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(Slots + row * SS).to(tl.int64)
    in_bounds = (loc >= 0) & (loc < CAP * DCP_WORLD_SIZE)
    tl.device_assert(in_bounds, "native MXFP4 write location is out of bounds")
    # Local row zero is reserved for padding, including DCP graph capture.
    slot = loc // DCP_WORLD_SIZE
    valid = in_bounds & (slot > 0) & (loc % DCP_WORLD_SIZE == DCP_RANK)
    slot = tl.where(valid, slot, 0)
    cols = tl.arange(0, 512)
    x = tl.load(X + row * XS + cols * XC, valid, 0).to(tl.float32)
    grouped = tl.reshape(x, (16, 32))
    amax = tl.max(tl.abs(grouped), 1)
    ratio = tl.where(amax > 0, amax / 6.0, 1.0)
    bits = ratio.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    # Groups needing scale <= 2^-31 flush as zero; FP8 also loses most of them.
    flush = (amax > 0) & (exponent < -30)
    finite = tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 0) == 512
    bounded = tl.sum((exponent <= 30).to(tl.int32), 0) == 16
    rc = tl.arange(0, 64)
    rope_value = tl.load(R + row * RS + rc * RC, valid, 0)
    rope_finite = tl.sum(
        (tl.abs(rope_value.to(tl.float32)) < float("inf")).to(tl.int32), 0
    ) == 64
    row_ok = finite & bounded & rope_finite
    # debug=True at the only launcher makes these runtime checks unconditional,
    # including graph replay. Mask stores too: bad rows never become cache data.
    tl.device_assert(
        (~valid) | row_ok,
        "native MXFP4 requires finite BF16 and scale exponents <= 30",
    )
    valid = valid & row_ok
    encoded = tl.where((amax == 0) | flush, 127, exponent + 127)
    grouped = tl.where(flush[:, None], 0.0, grouped)
    scale = (encoded << 23).to(tl.float32, bitcast=True)
    normalized = tl.reshape(grouped / scale[:, None], (512,))
    a = tl.abs(normalized)
    # Midpoints choose the code with an even least significant bit.
    q = tl.where(a > 0.25, 1, 0)
    q = tl.where(a >= 0.75, 2, q)
    q = tl.where(a > 1.25, 3, q)
    q = tl.where(a >= 1.75, 4, q)
    q = tl.where(a > 2.5, 5, q)
    q = tl.where(a >= 3.5, 6, q)
    q = tl.where(a > 5.0, 7, q)
    q = q | ((normalized < 0).to(tl.int32) << 3)
    pairs = tl.reshape(q, (256, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1)
    tl.store(Cache + slot * CS + tl.arange(0, 256), packed.to(tl.uint8), valid)
    tl.store(Cache + slot * CS + 256 + tl.arange(0, 16), encoded.to(tl.uint8), valid)
    rope = rope_value.to(tl.uint16, bitcast=True)
    tl.store(Cache + slot * CS + 272 + 2 * rc, (rope & 255).to(tl.uint8), valid)
    tl.store(Cache + slot * CS + 273 + 2 * rc, (rope >> 8).to(tl.uint8), valid)


def set_mla_kv_buffer_mxfp4(
    cache: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
) -> None:
    """Scatter BF16 [N,1,512]/[N,1,64] rows into contiguous uint8 [CAP,1,400].

    Real locations must be unique per (owner rank, local row) within a write.
    Device checks do not host-sync or inspect padding or foreign inputs,
    and reject nonfinite/overflowing real rows;
    underflowing 32-value groups are stored as zero.
    """
    if not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("native MXFP4 requires 0 <= dcp_rank < dcp_world_size")
    if cache.dtype != torch.uint8 or cache.ndim != 3 or cache.shape[1:] != (1, 400):
        raise ValueError("native MXFP4 cache requires uint8 [capacity,1,400]")
    if not cache.is_contiguous():
        raise ValueError("native MXFP4 cache must be contiguous")
    if loc.ndim != 1 or loc.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "native MXFP4 locations require a one-dimensional int32/int64 tensor"
        )
    n = loc.numel()
    if (
        k_rope is None
        or k_nope.dtype != torch.bfloat16
        or k_rope.dtype != torch.bfloat16
    ):
        raise ValueError("native MXFP4 writer requires BF16 NoPE and RoPE")
    if k_nope.shape != (n, 1, 512) or k_rope.shape != (n, 1, 64):
        raise ValueError("native MXFP4 writer requires [N,1,512] NoPE and [N,1,64] RoPE")
    if any(t.device != cache.device for t in (loc, k_nope, k_rope)):
        raise ValueError("native MXFP4 writer tensors must share a device")
    if n == 0:
        return
    _write_mxfp4_rows[(n,)](
        k_nope,
        k_rope,
        loc,
        cache,
        k_nope.stride(0),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(2),
        loc.stride(0),
        cache.stride(0),
        cache.shape[0],
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        num_warps=4,
        debug=True,
    )
