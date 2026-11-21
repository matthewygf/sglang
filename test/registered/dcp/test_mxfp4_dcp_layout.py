"""Check the gfx95 prefill gather on CPU, including ranks with no live rows."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.layers.attention.dsa import dsa_backend_hip
from sglang.srt.layers.dcp.layout import filter_dcp_local_chunk_kv_indices
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _gather_rank(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, world_size=world, rank=rank)
    try:
        with get_parallel().override(
            dcp_enabled=True, dcp_size=world, dcp_rank=rank, dcp_group=dist
        ):
            for page in (1, 64):
                span = world * page
                for lengths in ([1], [span + 1, 3]):
                    # Allocate the two requests from separate, nonadjacent pages.
                    p = torch.arange(lengths[0])
                    first = torch.tensor([6, 2])[p // span] * span + p % span
                    slots = torch.cat(
                        [first] + [9 * span + torch.arange(n) for n in lengths[1:]]
                    )
                    lens = torch.tensor(lengths)
                    rows = torch.randint(
                        256,
                        (len(slots), 1, 400),
                        dtype=torch.uint8,
                        generator=torch.Generator().manual_seed(42),
                    )
                    own = slots % world == rank
                    pool = torch.zeros((12 * page, 1, 400), dtype=torch.uint8)
                    pool[slots[own] // world] = rows[own]
                    read_ids = (
                        filter_dcp_local_chunk_kv_indices(
                            slots, torch.zeros_like(lens), lens
                        )
                        // world
                    )
                    gathered = dsa_backend_hip.dcp_gather_kv_cache(
                        SimpleNamespace(kv_cache_dim=400, physical_page_size=page),
                        pool,
                        SimpleNamespace(seq_lens_cpu=lens),
                        SimpleNamespace(dcp_local_kv_rows=read_ids),
                    )
                    assert torch.equal(gathered[: len(slots)], rows)
                    assert gathered.shape == (len(slots) + (-len(slots) % page), 1, 400)
                    assert gathered[len(slots) :].count_nonzero() == 0
    finally:
        dist.destroy_process_group()


def test_mxfp4_gather_preserves_rows(tmp_path):
    world = 2
    torch.multiprocessing.spawn(
        _gather_rank, args=(world, (tmp_path / "gloo").as_uri()), nprocs=world
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
