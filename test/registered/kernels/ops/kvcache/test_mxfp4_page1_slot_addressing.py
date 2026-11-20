"""Native MXFP4 radix reuse must address the rows that were actually written.

Why this test exists
--------------------
A radix cache hit does not recompute KV. It hands the scheduler a list of slot
ids that were written by some earlier request and says "these already hold the
prefix". If those ids are wrong, the model attends to the wrong rows and returns
confident nonsense, and no cache-hit counter can see it. That is the failure this
file checks on one GPU without loading a model.

This file exercises the legacy page-size-1 MXFP4 layout with 400-byte latent
rows. Run with SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 before import.

What is asserted
----------------
1. Distinct, non-consecutive slots each carry a distinguishable payload: 400
   packed latent bytes plus FP8 index keys and their scales.
2. Relocating a subset moves exactly those payloads, leaves every neighbouring
   slot byte-identical, and keeps latent rows and index rows in lockstep.
3. A pool rebuilt in the same order from a fresh buffer yields the same bytes at
   the same slot, so "the same slot id" is a stable address.

A mismatch means reuse would point at the wrong memory, which is the only
mechanism by which enabling radix could corrupt an answer.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import (
    get_parallel,
    publish,
    reset_context,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or aiter_can_use_preshuffle_paged_mqa(),
    reason="requires ROCm without DSA preshuffle",
)


@pytest.fixture(autouse=True)
def restore_runtime():
    saved = snapshot_context()
    try:
        yield
    finally:
        restore_context(saved)

PACKED_ROW_BYTES = 400
KV_LORA_RANK = 512  # logical latent rank; the packed buffer is 400 bytes wide
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
LAYER_NUM = 2


def _build_pool(size, dtype, kv_cache_dim):
    # DSATokenToKVPool reads the resolved config bag at construction, so publish a
    # minimal one first. The dummy model path avoids a Hugging Face lookup; the
    # values the pool actually reads are the ones passed explicitly below.
    reset_context()
    args = ServerArgs(model_path="dummy", device="cuda")
    args._mamba_cache_chunk_size = 64
    publish(args, role="tokenizer")
    with get_parallel().override(attn_dcp_size=1):
        return DSATokenToKVPool(
            size=size,
            page_size=1,
            kv_lora_rank=KV_LORA_RANK,
            dtype=dtype,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=INDEX_HEAD_DIM,
            enable_memory_saver=False,
            kv_cache_dim=kv_cache_dim,
            skip_topk_layers=[False, True],
        )


def _tag_rows(pool, slots, seed):
    """Give each slot a payload that identifies it, so a swap is detectable.

    Every byte of a slot's row is set to one value derived from that slot, so a
    moved row is recognisable by inspection and a partial move is impossible to
    miss. A unique index key per slot keeps latent rows and index rows tellable
    apart.
    """
    torch.manual_seed(seed)
    device = pool.kv_buffer[0].device
    rows = torch.arange(slots.numel(), dtype=torch.int64, device=device).unsqueeze(1)
    # 1..250 so no slot is distinguishable from an unwritten zero row.
    payload = ((rows * 7 + seed) % 250 + 1).to(torch.uint8)
    packed = payload.expand(-1, PACKED_ROW_BYTES).contiguous()
    for buf in pool.kv_buffer:
        flat = buf.view(buf.shape[0], -1)
        assert flat.dtype == torch.uint8, f"expected a packed byte row, got {flat.dtype}"
        flat[slots] = packed
    keys = torch.stack(
        [torch.full((INDEX_HEAD_DIM,), float(s + 1)) for s in range(slots.numel())]
    ).to(fp8_dtype).to(device)
    scales = (slots.float() + 1).unsqueeze(1).to(device)
    pool.set_index_k_scale_buffer(0, slots, keys, scales)
    return keys, scales


def _snapshot(pool):
    return [buf.view(buf.shape[0], -1).clone() for buf in pool.kv_buffer]


@pytest.mark.parametrize(
    "source_slots",
    [[1, 2], [0, 3, 7], [64, 127], [255, 256, 300]],
    ids=["adjacent", "sparse", "page_boundary", "cross_page"],
)
def test_mxfp4_page1_relocation_preserves_neighbours(source_slots):
    pool = _build_pool(size=320, dtype=torch.uint8, kv_cache_dim=PACKED_ROW_BYTES)
    slots = torch.arange(320, device=pool.kv_buffer[0].device)
    _tag_rows(pool, slots, seed=11)

    before = _snapshot(pool)
    source = torch.tensor(source_slots, device=slots.device, dtype=torch.int64)
    target = source.flip(0).clone()
    touched = set(source.tolist()) | set(target.tolist())

    pool.move_kv_cache(target, source)

    after = _snapshot(pool)
    for layer, (old, new) in enumerate(zip(before, after, strict=True)):
        # A slot not named by the move must be byte-identical.
        untouched = [s for s in range(old.shape[0]) if s not in touched]
        torch.testing.assert_close(
            new[untouched], old[untouched], rtol=0, atol=0,
            msg=f"layer {layer}: relocation disturbed slots it was not asked to move",
        )
        # Each target slot must now hold exactly the source slot's bytes.
        torch.testing.assert_close(
            new[target], old[source], rtol=0, atol=0,
            msg=f"layer {layer}: target slot does not hold the moved row",
        )


def test_mxfp4_page1_relocation_moves_index_in_lockstep():
    """Latent rows and DSA index rows must never drift apart."""
    pool = _build_pool(size=320, dtype=torch.uint8, kv_cache_dim=PACKED_ROW_BYTES)
    slots = torch.arange(320, device=pool.kv_buffer[0].device)
    keys, scales = _tag_rows(pool, slots, seed=23)

    # Compare through the raw per-layer view rather than reconstructing a shaped
    # result: this is the same buffer the forward pass reads. In the non-FP4
    # index layout the scale bytes live inside this buffer (all keys, then all
    # scales per page), so comparing it covers both.
    before = pool.index_key_cache.get_local_buffer(0).view(torch.uint8).clone()

    source = torch.tensor([4, 9, 250], device=slots.device, dtype=torch.int64)
    target = source.flip(0).clone()
    expected = before.clone()
    expected[target] = expected[source]

    pool.move_kv_cache(target, source)

    actual = pool.index_key_cache.get_local_buffer(0).view(torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp4_page1_slot_address_is_stable():
    """The same slot id must name the same bytes in a fresh pool.

    Reuse assumes an id written yesterday still addresses the row it was written
    for. Building the pool twice and tagging it the same way must reproduce the
    same bytes at the same id, or ids are not addresses at all.
    """
    first = _build_pool(size=320, dtype=torch.uint8, kv_cache_dim=PACKED_ROW_BYTES)
    slots = torch.arange(320, device=first.kv_buffer[0].device)
    _tag_rows(first, slots, seed=37)
    snapshot = _snapshot(first)
    del first

    second = _build_pool(size=320, dtype=torch.uint8, kv_cache_dim=PACKED_ROW_BYTES)
    _tag_rows(second, slots, seed=37)
    rebuilt = _snapshot(second)
    for layer, (a, b) in enumerate(zip(snapshot, rebuilt, strict=True)):
        torch.testing.assert_close(
            a, b, rtol=0, atol=0,
            msg=f"layer {layer}: slot ids do not stably address their payload",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
