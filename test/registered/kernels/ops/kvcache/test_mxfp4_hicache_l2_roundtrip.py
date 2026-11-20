"""Test MXFP4 HiCache L2 descriptors and device-host-device copies.

Set SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 before import for the page-1 run.
Unset it and enable AITER for the page-64 run.
"""

from threading import Event

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.device_pool_info import IndexKeyLayout, MLABufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import build_hybrid_mamba_stack
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MambaPool
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

LATENT_BYTES = 400
INDEX_KEY_BYTES = 128
SIZE = 64
KV_LAYERS = {0: 0, 2: 1}


@pytest.fixture
def runtime():
    saved = snapshot_context()
    reset_context()

    def configure(device, **kwargs):
        args = ServerArgs(model_path="dummy", device=device, **kwargs)
        args._mamba_cache_chunk_size = 64
        publish(args, role="tokenizer")
        return get_parallel().override(attn_dcp_size=1)

    try:
        yield configure
    finally:
        restore_context(saved)


def _pool(device, page_size=1):
    pool = DSATokenToKVPool(
        size=SIZE * page_size,
        page_size=page_size,
        dtype=torch.uint8,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=2,
        device=device,
        index_head_dim=INDEX_KEY_BYTES,
        enable_memory_saver=False,
        kv_cache_dim=LATENT_BYTES,
        skip_topk_layers=[False, True],
    )
    pool.model_layer_ids = tuple(KV_LAYERS)
    return pool


def _rocm_page_geometry_on_cpu(monkeypatch, page_size):
    # The pool checks platform page eligibility before allocating CPU tensors.
    monkeypatch.setattr(memory_pool, "_is_hip", True)
    monkeypatch.setattr(
        memory_pool, "aiter_can_use_preshuffle_paged_mqa", lambda: page_size == 64
    )


def test_platform_uint8_pool_does_not_select_native_codec(runtime, monkeypatch):
    _rocm_page_geometry_on_cpu(monkeypatch, 1)
    monkeypatch.setattr(memory_pool.current_platform, "is_out_of_tree", lambda: True)
    with runtime("cpu"):
        pool = _pool("cpu")
        assert not pool.dsa_kv_cache_store_mxfp4
        assert pool.kv_cache_dim == 512 + 64


def test_mxfp4_non_descriptor_guards(runtime, monkeypatch):
    _rocm_page_geometry_on_cpu(monkeypatch, 1)
    with runtime("cpu"):
        pool = _pool("cpu")
        with pytest.raises(NotImplementedError, match="disaggregation"):
            pool.get_contiguous_buf_infos()
        with pytest.raises(ValueError, match="page_size=64 with AITER preshuffle"):
            _pool("cpu", page_size=64)


@pytest.mark.parametrize("page_size,index_page_bytes", [(1, 132), (64, 8448)])
def test_hicache_descriptor_geometry(runtime, monkeypatch, page_size, index_page_bytes):
    _rocm_page_geometry_on_cpu(monkeypatch, page_size)
    with runtime("cpu"):
        pool = _pool("cpu", page_size)
        config = prepare_host_pool_config(
            root=pool,
            full_layer_mapping=KV_LAYERS,
            transfer_layer_id_max=3,
            transfer_page_size=page_size,
        )

    assert config.pool_names == (PoolName.KV, PoolName.INDEXER)
    kv = config.pool_infos[PoolName.KV]
    info = kv.buffer_info
    assert kv.layer_ids == (0, 2)
    assert isinstance(info, MLABufferInfo)
    info.validate(layer_ids=kv.layer_ids)
    assert info.device.type == "cpu"
    assert info.dtype == torch.uint8
    assert info.page_size == page_size
    assert info.kv_cache_dim == LATENT_BYTES
    assert info.scales is None and not info.packed_fp8
    assert all(row[0].nbytes == LATENT_BYTES for row in info.buffers)

    index = config.pool_infos[PoolName.INDEXER]
    info = index.buffer_info
    assert index.indices_from_pool == PoolName.KV
    assert index.layer_ids == (0,)
    info.validate(layer_ids=index.layer_ids)
    assert info.key_layout is IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES
    assert info.compress_ratio == 1
    assert info.page_size == info.kernel_page_size == page_size
    assert info.page_bytes == info.kernel_page_bytes == index_page_bytes
    assert info.buffers[0][0].nbytes == index_page_bytes
    # Host rows must split into whole token bytes. The kernel checks the
    # full page stride (not bytes per token) for 8-byte divisibility.
    assert divmod(info.kernel_page_bytes, page_size) == (132, 0)
    assert (info.kernel_page_bytes % 8 == 0) == (page_size == 64)


def _byte_rows(tensor):
    return tensor.view(torch.uint8).view(tensor.shape[0], -1)


def _tag_device_rows(pool, state_info):
    latents = [_byte_rows(tensor) for tensor in pool.kv_buffer]
    state = [_byte_rows(tensor[0]) for tensor in (state_info.temporal, *state_info.conv)]
    page_size = pool.page_size
    slots = torch.arange(pool.size + page_size, device="cuda")
    latents[0][:] = (slots + 1).to(torch.uint8)[:, None]
    for i, rows in enumerate([latents[1], *state], start=1):
        # Include a byte-position pattern to catch reordering within a row.
        offsets = torch.arange(rows.shape[1], device="cuda")
        rows[:] = (slots[: rows.shape[0], None] + 67 * i + offsets) % 254

    offsets = torch.arange(INDEX_KEY_BYTES, device="cuda")
    key_bytes = ((slots[:, None] + offsets) % 126 + 1).to(torch.uint8)
    fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    keys = key_bytes.view(fp8_dtype)
    scales = (slots.float() + 0.25)[:, None]
    pool.set_index_k_scale_buffer(0, slots, keys, scales)
    index = pool.index_key_cache.get_local_buffer(0)
    num_pages = slots.numel() // page_size
    if page_size == 1:
        torch.testing.assert_close(
            index[slots],
            torch.cat((key_bytes, scales.view(torch.uint8)), dim=1),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(
            index[:num_pages, INDEX_KEY_BYTES * page_size :],
            scales.view(torch.uint8).view(num_pages, -1),
            rtol=0,
            atol=0,
        )
    index[num_pages:].fill_(254)
    return {
        PoolName.KV: [rows.view(num_pages, page_size, LATENT_BYTES) for rows in latents],
        PoolName.INDEXER: [index],
        PoolName.MAMBA: state,
    }


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm"
)
@pytest.mark.parametrize(
    ("page_size", "io_backend", "layout"),
    [
        (1, "direct", "page_first_direct"),
        (64, "direct", "page_first_direct"),
        (64, "kernel", "page_first"),
    ],
    ids=["legacy-direct", "preshuffle-direct", "preshuffle-kernel"],
)
def test_mxfp4_hicache_l2_roundtrip(runtime, page_size, io_backend, layout):
    if aiter_can_use_preshuffle_paged_mqa() != (page_size == 64):
        pytest.skip(f"page size {page_size} needs its matching preshuffle environment")
    with runtime(
        "cuda",
        hicache_ratio=2,
        hicache_mem_layout=layout,
        hicache_io_backend=io_backend,
    ):
        pool = _pool("cuda", page_size)
        mamba = MambaPool(
            size=SIZE,
            spec_state_size=0,
            cache_params=KimiLinearCacheParams(
                shape=KimiLinearStateShape.create(
                    tp_world_size=1, num_heads=1, head_dim=8
                ),
                dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
                layers=[1],
            ),
            mamba_layer_ids=[1],
            device="cuda",
        )
        if page_size == 1:
            allocator = TokenToKVPoolAllocator(
                pool.size, torch.uint8, "cuda", pool, need_sort=False
            )
        else:
            allocator = PagedTokenToKVPoolAllocator(
                pool.size, page_size, torch.uint8, "cuda", pool, need_sort=False
            )
        allocator.state = StateStorage.empty("state")
        allocator.state.slots = MambaSlotAllocator(SIZE, "cuda")
        group, controller, _ = build_hybrid_mamba_stack(
            params=CacheInitParams(False, None, allocator, page_size),
            kv_pool=pool,
            mamba_pool=mamba,
            full_layer_mapping=KV_LAYERS,
            start_layer=0,
            load_cache_event=Event(),
            storage_backend=None,
        )
        try:
            kv_slots = allocator.alloc(pool.size)
            state_slots = allocator.state_slots.alloc(SIZE)
            assert kv_slots is not None and state_slots is not None
            pages = kv_slots.view(-1, page_size)
            source = pages[[6, 45, 16]].flatten()
            target = pages[[52, 9, 31]].flatten()
            state_source, state_target = state_slots[[3, 58]], state_slots[[24, 1]]
            (state_desc,) = mamba.get_device_pool_infos()
            buffers = _tag_device_rows(pool, state_desc.buffer_info)
            expected = {
                name: [row.clone() for row in rows] for name, rows in buffers.items()
            }

            # Do not let a copy that assumes host slots start at zero pass.
            assert group.alloc(5 * page_size, pool=PoolName.KV) is not None
            assert group.alloc(7, pool=PoolName.MAMBA) is not None
            state_backup = PoolTransfer(PoolName.MAMBA, device_indices=state_source)
            host_slots = controller.write(
                source,
                extra_pools=[
                    PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV),
                    state_backup,
                ],
            )
            assert host_slots is not None and state_backup.host_indices is not None
            controller.ack_write_queue[-1].finish_event.synchronize()

            for name, rows in buffers.items():
                src, dst, host = (
                    (state_source, state_target, state_backup.host_indices)
                    if name == PoolName.MAMBA
                    else (
                        source[::page_size] // page_size,
                        target[::page_size] // page_size,
                        host_slots[::page_size],
                    )
                )
                host_pool = group.get_pool(name)
                for i, host_slot in enumerate(host.tolist()):
                    backed_up = host_pool.get_data_page(host_slot)
                    want_rows = [row[src[i]] for row in expected[name]]
                    if name == PoolName.KV and layout == "page_first":
                        want = torch.stack(want_rows, dim=1).flatten()
                    else:
                        want = torch.cat([row.flatten() for row in want_rows])
                    torch.testing.assert_close(backed_up, want.cpu(), rtol=0, atol=0)
                for row, want in zip(rows, expected[name], strict=True):
                    torch.testing.assert_close(row, want, rtol=0, atol=0)
                    want[dst] = want[src]
                    want[src] = 255
                    row[src] = 255
                    row[dst] = 255

            loaded = controller.load(
                host_slots,
                device_indices=target,
                extra_pools=[
                    PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV),
                    PoolTransfer(
                        PoolName.MAMBA,
                        host_indices=state_backup.host_indices,
                        device_indices=state_target,
                    ),
                ],
            )
            assert loaded is not None
            controller.start_loading()
            controller.ack_load_queue[-1].finish_event.synchronize()

            # Compare every row, including poisoned source slots and neighbours.
            for name, rows in buffers.items():
                for row, want in zip(rows, expected[name], strict=True):
                    torch.testing.assert_close(row, want, rtol=0, atol=0)
        finally:
            torch.cuda.synchronize()
            group.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
