"""Option B prefill checks. Default: CPU-only T0a/T0b in the pinned GPU-free image.

Expose this checkout as `sglang` on PYTHONPATH. GPU tests are explicit:
  --test T1 T2 T3 T4 T5 bench
"""

import argparse
import importlib
import math
import os
from pathlib import Path
import py_compile
import shlex
import subprocess
import sysconfig
import tempfile
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
DSA = "sglang.kernels.ops.attention.dsa."
MOD = "sparse_mla_prefill_mxfp4_gfx950"
GATE = "SGLANG_MXFP4_NATIVE_PREFILL_GFX950"
D, DR, TOPK = 512, 64, 2048
SM = 1 / math.sqrt(D + DR)


def t0a():
    assert not torch.cuda.is_available(), "T0a requires a GPU-free environment"
    dsa = ROOT / "kernels/ops/attention/dsa"
    files = [Path(__file__), dsa / f"{MOD}.py", dsa / "triton_sparse_mla.py"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(files):
            py_compile.compile(str(path), cfile=f"{tmp}/{i}.pyc", doraise=True)
    for path in files[1:]:
        mod = importlib.import_module(DSA + path.stem)
        assert Path(mod.__file__).resolve() == path.resolve(), mod.__file__
    print("T0a PASS: py_compile (3 files), native wrapper and dispatcher import from the isolated copy")


def t0b():
    assert not torch.cuda.is_available(), "T0b requires a GPU-free environment"
    header = ROOT / "kernels/jit/csrc/attention" / f"{MOD}.cuh"
    includes = (
        ROOT / "kernels/jit/include",
        Path("/sgl-workspace/aiter/csrc/include"),
        Path(sysconfig.get_path("purelib")) / "tvm_ffi/include",
    )
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "prefill_mxfp4.hip"
        probe.write_text(
            f'#include "{header}"\n'
            'namespace sglang::sparse_mla_prefill_mxfp4_gfx950 {\n'
            '#if !__has_builtin(__builtin_amdgcn_ds_read_tr4_b64_v2i32)\n'
            '#error gfx950 tr4 builtin is required\n'
            '#endif\n'
            'template __global__ void sparse_mla_prefill_mxfp4_kernel<false>(const Params, uint32_t);\n'
            'template __global__ void sparse_mla_prefill_mxfp4_kernel<true>(const Params, uint32_t);\n'
            '}\n'
        )
        cmd = [
            "hipcc", "-x", "hip", "--offload-arch=gfx950", "--cuda-device-only",
            "-DUSE_ROCM", "-std=c++20", "-fsyntax-only",
            *(f"-I{path}" for path in includes), str(probe),
        ]
        print(shlex.join(cmd), flush=True)
        result = subprocess.run(cmd, check=True)
        print(f"T0b PASS: gfx950 tr4 + scaled FP4 builtins, split and unsplit device instantiations, exit {result.returncode}")


def inputs(T, H, topk, stride=400):
    from sglang.kernels.ops.quantization.dsa_quant_mxfp4 import set_mla_kv_buffer_mxfp4

    N = max(topk, T) + 17
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(20261126)

    def noise(*shape):
        return torch.randn(shape, device=device, generator=generator).bfloat16()

    q, qr = noise(T, H, D), noise(T, H, DR)
    k, kr = noise(N, 1, D), noise(N, 1, DR)
    # Vary E8M0 scales both across keys and across all 16 blocks.
    exps = torch.randint(-2, 3, (N, 1, 16), device=device, generator=generator)
    k = torch.ldexp(k.unflatten(-1, (16, 32)), exps[..., None]).flatten(-2)
    # The production writer reserves slot zero and writes contiguous rows.
    rows = torch.empty((N + 1, 1, 400), device=device, dtype=torch.uint8)
    loc = torch.arange(1, N + 1, device=device, dtype=torch.int32)
    set_mla_kv_buffer_mxfp4(rows, loc, k, kr)
    rows = rows[1:]
    if stride != 400:
        backing = torch.full((N, 1, stride), 0xA5, device=device, dtype=torch.uint8)
        rows = backing[..., :400].copy_(rows)
    # Each row selects a distinct cyclic window over a permuted pool.
    perm = torch.randperm(N, generator=generator, device=device, dtype=torch.int32)
    window = (torch.arange(T, device=device)[:, None] + torch.arange(topk, device=device)) % N
    ix = perm[window].unsqueeze(1)
    return q, qr, rows, ix


def reference(q, qr, rows, indices, sm=SM):
    """FP64 dense torch attention on the bytes emitted by the production writer."""
    rows = rows[:, 0]
    packed = rows[:, :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    mag = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=q.device, dtype=torch.float64)
    v = mag[(codes & 7).long()] * torch.where(codes >= 8, -1, 1)
    v = torch.ldexp(v, (rows[:, 256:272].int() - 127).repeat_interleave(32, 1))
    kr = rows[:, 272:].contiguous().view(torch.bfloat16).double()
    ix = indices.squeeze(1) if indices.ndim == 3 else indices
    valid = ix >= 0
    slots = ix.clamp_min(0).long()
    selected = v[slots]
    logits = (q.double() @ selected.mT + qr.double() @ kr[slots].mT) * sm
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    lse = logits.logsumexp(-1)
    lse_safe = torch.where(valid.any(-1)[:, None], lse, 0.0)
    out = (logits - lse_safe.unsqueeze(-1)).exp() @ selected
    return out.unsqueeze(0), lse


def native(*args, sm=SM, splits=None):
    mod = importlib.import_module(DSA + MOD)
    if splits is None:
        return mod.sparse_mla_prefill_mxfp4_gfx950(*args, sm, return_lse=True)
    return mod.sparse_mla_split_mxfp4_gfx950(*args, sm, splits, return_lse=True)


def triton_single(*args):
    mod = importlib.import_module(DSA + "triton_sparse_mla")
    return mod._triton_sparse_mla_fwd_single(*args, SM, return_lse=True, kv_format="mxfp4")


def errors(got, want):
    out, lse = got
    out_ref, lse_ref = want
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert out.shape == out_ref.shape and lse.shape == lse_ref.shape
    assert torch.isfinite(out).all()
    assert not torch.isnan(lse).any() and not torch.isposinf(lse).any()
    masked = torch.isneginf(lse_ref)
    assert torch.equal(torch.isneginf(lse), masked)
    assert (out[0, masked] == 0).all()
    # Both paths round PV operands and final output to BF16, so allow a relative
    # component for larger values (e.g. one BF16 ULP at 8 is 0.0625).
    torch.testing.assert_close(out.float(), out_ref.float(), atol=.016, rtol=.008)
    oe = (out.float() - out_ref.float()).abs().max().item()
    le = (lse[~masked] - lse_ref[~masked]).abs().max().item() if (~masked).any() else 0.0
    assert le <= .004, le
    return f"max|Δout|={oe:.6g}, max|ΔlnLSE|={le:.6g}"


def t1():
    for H in (1, 8, 12, 16):
        for topk in (1, 31, 32, 33, 2048):
            args = inputs(5, H, topk)
            got, want = native(*args), reference(*args)
            if topk == 1:
                # P=1: PV must reproduce the exact selected vector for every head.
                torch.testing.assert_close(got[0], want[0].bfloat16(), rtol=0, atol=0)
            print(f"T1 PASS H={H} topk={topk}: {errors(got, want)}")


def t2():
    for T in (4096, 8192):
        for H in (12, 16):
            args = inputs(T, H, TOPK)
            print(f"T2 PASS T={T} H={H}: {errors(native(*args), triton_single(*args))}")


def t3():
    for stride in (400, 416):
        q, qr, rows, ix = inputs(17, 12, 65, stride=stride)
        assert rows.stride(0) == stride
        ix[0].fill_(-1)
        ix[1, 0, ::3] = -1
        ix[2, 0, :-1] = -1  # Entirely masked tiles before a valid key in the final tile.
        args = q, qr, rows, ix
        print(f"T3 PASS T=17 stride={stride}: {errors(native(*args), reference(*args))}")
    # Exercise N * stride, not (N - 1) * stride + 400. No giant allocation needed.
    mod = importlib.import_module(DSA + MOD)
    supports = mod.supports_sparse_mla_prefill_mxfp4_gfx950
    q_meta = torch.empty((17, 12, D), device="meta", dtype=torch.bfloat16)
    qr_meta = torch.empty((17, 12, DR), device="meta", dtype=torch.bfloat16)
    for stride, expected in ((0x40000000 - 16, True), (0x40000000, False)):
        kv = torch.empty_strided((2, 1, 400), (stride, 400, 1), device="meta", dtype=torch.uint8)
        assert supports(q_meta, qr_meta, kv, D) is expected
    # Also exercise the C++ guard with a one-row tensor before any kernel launches.
    kv = rows[:1].as_strided((1, 1, 400), (0x80000000, 400, 1))
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    try:
        mod._jit_module().run(q, qr, kv, ix.squeeze(1), out, lse, SM, 1)
    except (RuntimeError, ValueError) as exc:
        assert "31-bit buffer offset" in str(exc), str(exc)
    else:
        raise AssertionError("Expected native 31-bit offset guard to reject oversized stride")
    print("T3 PASS: 31-bit guard (Python product boundary and direct native rejection)")


def t4():
    q, qr, kv, ix = inputs(17, 16, 97, stride=416)
    ix[0].fill_(-1)
    ix[1, 0, :64] = -1
    args = q, qr, kv, ix
    unsplit = native(*args)
    want = reference(*args)
    for splits in (1, 2, 3, 8):  # Uneven tile groups and empty splits.
        got = native(*args, splits=splits)
        print(f"T4 PASS splits={splits}: vs run {errors(got, unsplit)}; vs fp64 {errors(got, want)}")


def t5():
    mod = importlib.import_module(DSA + MOD)
    dispatch = importlib.import_module(DSA + "triton_sparse_mla")
    q, qr, kv, ix = inputs(17, 12, 33)
    native_result, single_result, split_result = object(), object(), object()
    with (
        patch.object(dispatch, "_cu_count", return_value=16),
        patch.object(dispatch, "_sparse_mla_block_k", return_value=32),
        patch.object(dispatch, "is_gfx95_supported", return_value=True) as gfx95,
        patch.object(mod, "sparse_mla_prefill_mxfp4_gfx950", return_value=native_result) as run,
        patch.object(dispatch, "_triton_sparse_mla_fwd_single", return_value=single_result),
        patch.object(dispatch, "_triton_sparse_mla_fwd_splitk", return_value=split_result),
        patch.dict(os.environ),
    ):
        fwd = dispatch.triton_sparse_mla_fwd
        opts = dict(kv_format="mxfp4", return_lse=True)
        os.environ.pop(GATE, None)
        assert fwd(q, qr, kv, ix, SM, **opts) is single_result
        run.assert_not_called()
        os.environ[GATE] = "1"
        assert fwd(q, qr, kv, ix, SM, **opts) is native_result
        run.assert_called_once_with(q, qr, kv, ix, SM, d_v=D, return_lse=True)
        run.reset_mock()
        assert fwd(q[:16], qr[:16], kv, ix[:16], SM, **opts) is split_result
        kv_bad_stride = torch.empty((kv.shape[0], 1, 402), device=kv.device, dtype=kv.dtype)[..., :400]
        assert fwd(q, qr, kv_bad_stride, ix, SM, **opts) is single_result
        kv_bad_base = torch.empty((kv.shape[0], 1, 416), device=kv.device, dtype=kv.dtype)[..., 1:401]
        assert fwd(q, qr, kv_bad_base, ix, SM, **opts) is single_result
        lengths = torch.full((17,), 1, dtype=torch.int32, device=q.device)
        assert fwd(q, qr, kv, ix, SM, topk_length=lengths, **opts) is split_result
        gfx95.return_value = False
        assert fwd(q, qr, kv, ix, SM, **opts) is single_result
        gfx95.return_value = True
        os.environ[GATE] = "0"
        assert fwd(q, qr, kv, ix, SM, **opts) is single_result
        run.assert_not_called()
    print("T5 PASS: opt-in/off, CU threshold, layout, explicit lengths and arch fallback")


def benchmark():
    from triton.testing import do_bench_cudagraph

    for H in (12, 16):
        args = inputs(8192, H, TOPK)
        parity = errors(native(*args), triton_single(*args))
        torch.cuda.synchronize()
        a = do_bench_cudagraph(lambda: native(*args)) * 1000
        b = do_bench_cudagraph(lambda: triton_single(*args)) * 1000
        print(f"bench T=8192 H={H} topk=2048: native={a:.2f} us Triton={b:.2f} us speedup={b/a:.3f}x; {parity}")


def main():
    tests = dict(T0a=t0a, T0b=t0b, T1=t1, T2=t2, T3=t3, T4=t4, T5=t5, bench=benchmark)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", nargs="+", choices=tests, default=["T0a", "T0b"])
    opts = parser.parse_args()
    if any(test not in ("T0a", "T0b") for test in opts.test):
        assert torch.version.hip and torch.cuda.is_available(), "GPU tests require ROCm"
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
        assert arch == "gfx950", arch
    for test in opts.test:
        tests[test]()


if __name__ == "__main__":
    main()
