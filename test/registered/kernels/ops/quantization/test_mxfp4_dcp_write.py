"""Compare the production MXFP4 scatter across CPU-interpreted Triton ranks."""

import os
import subprocess
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _check_rows(world):
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.kernels.ops.quantization.dsa_quant_mxfp4 import set_mla_kv_buffer_mxfp4

    # Three nonadjacent allocator pages. The last holds just one live token.
    length = 2 * world + 1
    slots = torch.zeros(length + 1, dtype=torch.int64)
    prefix = world - 1
    alloc_extend_kernel[(1,)](
        torch.tensor([0]),
        torch.tensor([prefix]),
        torch.tensor([-1]),
        torch.tensor([3]),
        slots[:prefix],
        world,
    )
    alloc_extend_kernel[(1,)](
        torch.tensor([prefix]),
        torch.tensor([length]),
        slots[prefix - 1 : prefix],
        torch.tensor([1, 6]),
        slots[prefix:],
        world,
    )
    assert torch.equal(slots[:-1] % world, torch.arange(length) % world)
    assert slots[-1] == 0
    gen = torch.Generator().manual_seed(31)
    x = torch.randn((length + 1, 1, 512), generator=gen).bfloat16()
    rope = torch.randn((length + 1, 1, 64), generator=gen).bfloat16()
    x[-1] = rope[-1] = float("nan")
    reference = torch.full((length + 1, 1, 400), 0xAD, dtype=torch.uint8)
    set_mla_kv_buffer_mxfp4(reference, torch.arange(length + 1).roll(-1), x, rope)

    def write(flip_rank):
        shards = torch.full((world, 7, 1, 400), 0xAD, dtype=torch.uint8)
        for rank, cache in enumerate(shards):
            writer_rank = 1 if flip_rank and rank == 0 else rank
            set_mla_kv_buffer_mxfp4(
                cache, slots, x, rope, dcp_world_size=world, dcp_rank=writer_rank
            )
            owned = slots[:-1][slots[:-1] % world == writer_rank] // world
            unused = ~torch.isin(torch.arange(len(cache)), owned)
            assert (cache[unused] == 0xAD).all()  # includes the reserved sink
        return shards[slots[:-1] % world, slots[:-1] // world]

    assert torch.equal(write(False), reference[1:])
    assert not torch.equal(write(True), reference[1:])


def test_mxfp4_dcp_rows_equal_dcp1():
    # Interpret only in the child. Other tests in this process may use a GPU JIT.
    subprocess.run(
        [sys.executable, __file__, "--interpret"],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        check=True,
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--interpret"]:
        for world in (2, 8):
            _check_rows(world)
    else:
        sys.exit(pytest.main([__file__, "-v"]))
