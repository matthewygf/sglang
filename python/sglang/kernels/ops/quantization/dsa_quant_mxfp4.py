# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MXFP4 recipe and E2M1 thresholds adapted from:
# https://github.com/vllm-project/vllm/blob/0599e4ba72fd29265a0206b29a0d54b308dddcb8/vllm/v1/attention/ops/mxfp4_mla/store.py
# Port of the Kimi investigation's prototype/codec_smoke.py write_rows.
# The port retains group32 E2M1/E8M0 and the raw BF16 RoPE tail.
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
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row * SS).to(tl.int64)
    in_bounds = (slot >= 0) & (slot < CAP)
    tl.device_assert(in_bounds, "native MXFP4 write location is out of bounds")
    # Slot zero is reserved for padded tokens, which may repeat and contain NaNs.
    valid = (slot > 0) & in_bounds
    cols = tl.arange(0, 512)
    x = tl.load(X + row * XS + cols * XC, valid, 0).to(tl.float32)
    grouped = tl.reshape(x, (16, 32))
    amax = tl.max(tl.abs(grouped), 1)
    ratio = tl.where(amax > 0, amax / 6.0, 1.0)
    bits = ratio.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    finite = tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 0) == 512
    bounded = tl.sum(((exponent >= -30) & (exponent <= 30)).to(tl.int32), 0) == 16
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
        "native MXFP4 requires finite BF16 and scale exponents [-30,30]",
    )
    valid = valid & row_ok
    encoded = tl.where(amax == 0, 127, exponent + 127)
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
) -> None:
    """Scatter BF16 [N,1,512]/[N,1,64] rows into contiguous uint8 [CAP,1,400].

    Real physical locations must be unique within a scheduler write; only the
    reserved padding location zero may repeat. Device checks do not host-sync
    or inspect padded inputs, and reject nonfinite/out-of-domain real rows.
    """
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
        num_warps=4,
        debug=True,
    )
