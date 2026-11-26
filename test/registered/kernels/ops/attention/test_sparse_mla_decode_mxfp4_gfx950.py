"""Option B validation. Default: CPU-only T0a/T0b/T0c.

Run T0c with TRITON_INTERPRET=1 (set before importing Triton).

Run against an importable copy of this tree as `sglang` (e.g. a symlink in
PYTHONPATH). GPU tests are explicitly selected: --test T1 T2 T3 T4 T5 bench.
T1 can be rerun after flipping the header's kQkLayoutN (use a fresh JIT cache).
For T6: torchrun --nproc-per-node=8 test_sparse_mla_decode_mxfp4_gfx950.py --test T6
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

import torch

D, DR, TOPK = 512, 64, 2048
SM = 1 / math.sqrt(D + DR)
MODULE = "sglang.kernels.ops.attention.dsa.sparse_mla_decode_mxfp4_gfx950"
HEADER = "attention/sparse_mla_decode_mxfp4_gfx950.cuh"
ROOT = Path(__file__).resolve().parent


def magnitudes(device):
    return torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=device)


def pack_t1_t2_patterns(codes, exponents, rope, stride):
    """Pack exact analytical E2M1 patterns for T1/T2 only."""
    n = codes.shape[0]
    backing = torch.full((n, 1, stride), 0xA5, dtype=torch.uint8, device=codes.device)
    rows = backing[..., :400]
    rows[:, 0, :256] = codes[:, ::2] | (codes[:, 1::2] << 4)
    rows[:, 0, 256:272] = (exponents + 127).to(torch.uint8)
    rows[:, 0, 272:] = rope.contiguous().view(torch.uint8)
    return rows


def unpack(rows):
    rows = rows[:, 0]
    packed = rows[:, :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    mag = magnitudes(rows.device)[(codes & 7).long()]
    # Follow the cache codec: both zero codes become +0.
    signed = torch.where((codes > 8) & (mag != 0), -mag, mag)
    k = torch.ldexp(signed, (rows[:, 256:272].int() - 127).repeat_interleave(32, 1))
    return k.bfloat16(), rows[:, 272:].clone().view(torch.bfloat16)


def mxq(q):
    x = q.float().unflatten(-1, (16, 32))
    exponent = (x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 448).log2().ceil()
    exponent = exponent.clamp(-127, 127).int()
    fp8 = torch.ldexp(x, -exponent).to(torch.float8_e4m3fn).float()
    return torch.ldexp(fp8, exponent).flatten(-2)


def dense(q, qr, k, kr, indices, lengths, sm=SM):
    """Plain FP32 attention on supplied operands; no Triton or native kernel."""
    R, H, _ = q.shape
    out = torch.zeros((1, R, H, D), device=q.device, dtype=torch.float32)
    lse = torch.full((R, H), -torch.inf, device=q.device, dtype=torch.float32)
    for r, length in enumerate(lengths.tolist()[:R]):
        if length == 0:
            continue
        ix = indices[r, 0, :length].long()
        v = k[ix].float()
        logits = (q[r].float() @ v.T + qr[r].float() @ kr[ix].float().T) * sm
        out[0, r] = logits.softmax(-1) @ v
        lse[r] = logits.logsumexp(-1)
    return out, lse


def reference(q, qr, rows, ix, lengths, sm=SM):
    k, kr = unpack(rows)
    return dense(mxq(q), qr, k, kr, ix, lengths, sm)


def inputs(R, H, lengths, stride=400, device="cuda", seed=140, n=TOPK):
    g = torch.Generator(device=device).manual_seed(seed)

    def noise(*shape):
        return torch.randn(shape, generator=g, device=device).bfloat16()

    q, qr, k, kr = noise(R, H, D), noise(R, H, DR), noise(n, D), noise(n, DR)
    from sglang.kernels.ops.quantization.dsa_quant_mxfp4 import (
        set_mla_kv_buffer_mxfp4,
    )

    # The production writer requires contiguous storage and reserves slot 0.
    packed = torch.empty((n + 1, 1, 400), dtype=torch.uint8, device=device)
    loc = torch.arange(1, n + 1, dtype=torch.int32, device=device)
    set_mla_kv_buffer_mxfp4(packed, loc, k.unsqueeze(1), kr.unsqueeze(1))
    rows = packed[1:]
    if stride != 400:
        # Exercise padded physical strides without changing the production bytes.
        backing = torch.full((n, 1, stride), 0xA5, dtype=torch.uint8, device=device)
        rows = backing[..., :400].copy_(rows)
    lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    assert len(lengths) == R
    perm = torch.stack([torch.randperm(n, generator=g, device=device) for _ in range(R)])
    ix = torch.full((R, 1, TOPK), -1, device=device, dtype=torch.int32)
    ix[:, 0, :n] = perm.masked_fill(torch.arange(n, device=device) >= lengths[:, None], -1)
    return (q, qr, rows, ix, lengths), (k, kr)


def errors(got, want, max_out, max_lse):
    o, l = got
    ref_o, ref_l = want
    assert torch.isfinite(o).all()
    assert not torch.isnan(l).any() and not torch.isposinf(l).any()
    torch.testing.assert_close(torch.isneginf(l), torch.isneginf(ref_l))
    finite = torch.isfinite(ref_l)
    assert (o[0, ~finite] == 0).all()
    oe = (o.float() - ref_o.float()).abs().max().item()
    le = (l[finite] - ref_l[finite]).abs().max().item() if finite.any() else 0.0
    assert oe <= max_out and le <= max_lse, (oe, le, max_out, max_lse)
    return f"max|Δo|={oe:.6g}, max|ΔlnLSE|={le:.6g}"


def t0a():
    files = [
        Path(__file__), ROOT / "kernels/ops/attention/dsa/sparse_mla_decode_mxfp4_gfx950.py",
        ROOT / "srt/layers/attention/dsa/dsa_backend_hip.py",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(files):
            py_compile.compile(str(path), cfile=f"{tmp}/{i}.pyc", doraise=True)
    assert not torch.cuda.is_available(), "T0a requires a GPU-free environment"
    mod = importlib.import_module(MODULE)
    assert Path(mod.__file__).resolve() == files[1].resolve(), mod.__file__
    assert not mod.supports_sparse_mla_decode_mxfp4_gfx950(
        torch.empty(1, 16, D, dtype=torch.bfloat16), torch.empty(1, 1, 400, dtype=torch.uint8), D
    )
    print("T0a PASS: py_compile (3 files), isolated native-wrapper import, CPU guard rejected")


def t0b():
    assert not torch.cuda.is_available(), "T0b requires a GPU-free environment"
    header = ROOT / "kernels/jit/csrc" / HEADER
    toggle = "constexpr bool kQkLayoutN = true;"
    text = header.read_text()
    assert text.count(toggle) == 1
    includes = (
        ROOT / "kernels/jit/include",
        Path("/sgl-workspace/aiter/csrc/include"),
        Path(sysconfig.get_path("purelib")) / "tvm_ffi/include",
    )
    cmd = [
        "hipcc", "-x", "hip", "--offload-arch=gfx950", "--cuda-device-only",
        "-DUSE_ROCM", "-std=c++20", "-fsyntax-only",
        *(f"-I{path}" for path in includes),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        alt_header = Path(tmp) / header.name
        alt_header.write_text(text.replace(toggle, "constexpr bool kQkLayoutN = false;"))
        for layout, source in (("N", header), ("H", alt_header)):
            probe = Path(tmp) / f"layout_{layout}.hip"
            probe.write_text(f'#include "{source}"\n')
            full_cmd = [*cmd, str(probe)]
            print(shlex.join(full_cmd), flush=True)
            result = subprocess.run(full_cmd, check=True)
            print(f"T0b PASS: gfx950 device syntax, layout {layout}, exit {result.returncode}")


def t0c():
    assert not torch.cuda.is_available(), "T0c requires a GPU-free environment"
    assert os.environ.get("TRITON_INTERPRET") == "1", "Run T0c with TRITON_INTERPRET=1"
    torch.set_num_threads(2)
    for stride in (400, 416):
        args, (original, kr) = inputs(4, 16, [0, 1, 31, 97], stride, "cpu", n=128)
        q, qr, rows, ix, lengths = args
        k, decoded_rope = unpack(rows)
        torch.testing.assert_close(decoded_rope, kr, rtol=0, atol=0)
        got = reference(*args)
        same_kv = dense(q, qr, k, kr, ix, lengths)
        unquantized = dense(q, qr, original, kr, ix, lengths)
        # A fixed absolute output tolerance is not a quantization invariant:
        # for logit perturbations bounded by delta, softmax total variation is
        # at most tanh(delta/2), so the per-channel output bound scales with
        # the range of selected values. Check this and the 1-Lipschitz LSE bound.
        max_bound = 0.0
        q_delta = mxq(q) - q.float()
        for r, n in enumerate(lengths.tolist()):
            if n == 0:
                continue
            v = k[ix[r, 0, :n].long()].float()
            delta = (q_delta[r] @ v.T * SM).abs().amax(-1)
            bound = (v.amax(0) - v.amin(0)) * (delta[:, None] / 2).tanh()
            assert ((got[0][0, r] - same_kv[0][0, r]).abs() <= bound + 1e-6).all()
            assert ((got[1][r] - same_kv[1][r]).abs() <= delta + 1e-6).all()
            max_bound = max(max_bound, bound.max().item())
        quant_q = errors(got, same_kv, max_bound + 1e-6, .05)
        # An exactly MXFP8-representable BF16 lattice must not perturb attention.
        lattice_q = (q.sign() * 0.5).bfloat16()
        torch.testing.assert_close(mxq(lattice_q), lattice_q.float(), rtol=0, atol=0)
        for actual, expected in zip(
            reference(lattice_q, qr, rows, ix, lengths), dense(lattice_q, qr, k, kr, ix, lengths)
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # K quantization also perturbs logits; derive the bound from the total
        # QK error rather than treating a lossy codec as numerically identical.
        total_lse_bound = 0.0
        for r, n in enumerate(lengths.tolist()):
            if n == 0:
                continue
            slots = ix[r, 0, :n].long()
            v = k[slots].float()
            logit_delta = (q_delta[r] @ v.T + q[r].float() @ (v - original[slots].float()).T) * SM
            delta = logit_delta.abs().amax(-1)
            assert ((got[1][r] - unquantized[1][r]).abs() <= delta + 1e-6).all()
            total_lse_bound = max(total_lse_bound, delta.max().item())
        quant_total = errors(got, unquantized, .7, total_lse_bound + 1e-6)
        torch.testing.assert_close(got[0][0, 1], k[ix[1, 0, 0]].float().expand(16, D), rtol=0, atol=0)
        z = torch.zeros_like(q), torch.zeros_like(qr)
        _, lse = reference(*z, rows, ix, lengths)
        torch.testing.assert_close(lse, lengths.float().log()[:, None].expand_as(lse))
        print(f"T0c PASS stride={stride}: BF16 Q/same KV {quant_q}; dense original {quant_total}")


def native(q, qr, rows, ix, lengths, *, sm=SM, splits=None):
    return importlib.import_module(MODULE).sparse_mla_decode_mxfp4_gfx950(
        q, qr, rows, ix, sm, lengths, expected_topk=256, num_splits=splits,
    )


def triton_decode(q, qr, rows, ix, lengths):
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import triton_sparse_mla_decode_splitk

    return triton_sparse_mla_decode_splitk(
        q, qr, rows, ix, SM, lengths=lengths, return_lse=True, expected_topk=256,
        natural_lse=True, kv_format="mxfp4",
    )


def t1():
    # One token per row => natural LSE equals the raw score. Sweep all 512 Q
    # basis columns against all 512 K basis rows, with varied signs/magnitudes.
    d = torch.arange(D, device="cuda")
    vals = 1 + d % 7 + (d % 2) * 8
    codes = torch.zeros((D, D), dtype=torch.uint8, device=d.device)
    codes[d, d] = vals.to(torch.uint8)
    exponents = (d[:, None] + 3 * d[None, :16]) % 9 - 4
    rows = pack_t1_t2_patterns(codes, exponents, torch.zeros(D, DR, device=d.device, dtype=torch.bfloat16), 416)
    diag = magnitudes(d.device)[vals & 7] * torch.where(vals >= 8, -1, 1)
    diag = torch.ldexp(diag, exponents[d, d // 32])
    qr = torch.zeros((D, 16, DR), device=d.device, dtype=torch.bfloat16)
    lengths = torch.ones(D, device=d.device, dtype=torch.int32)
    ix = d.int().reshape(D, 1, 1)
    eye = torch.eye(D, device=d.device, dtype=torch.bfloat16)
    amp = torch.ldexp(torch.where(d % 3 == 0, -1., 1.), (d % 7 - 3).int())
    for start in range(0, D, 16):
        q = (eye[start:start + 16] * amp[start:start + 16, None]).bfloat16()
        _, lse = native(q.expand(D, -1, -1), qr, rows, ix, lengths, sm=1, splits=1)
        want = (d[:, None] == d[None, start:start + 16]) * (diag * amp)[:, None]
        torch.testing.assert_close(lse, want, rtol=2e-6, atol=1e-6)
    print("T1 PASS: 512x512 one-hot logits, signed FP4/FP8 and distinct block scales")


def t2():
    # P=1 and exactly one selected key: the PV path must reproduce each BF16 V
    # bit (including canonical zero) over all nibbles/scales and wave-chunks.
    r = torch.arange(61, device="cuda")
    d = torch.arange(D, device=r.device)
    codes = ((d[None, :] * 7 + d[None, :] // 16 + r[:, None]) % 16).to(torch.uint8)
    exps = (r[:, None] + 7 * d[None, :16]) % 61 - 30
    qr = torch.zeros((len(r), 16, DR), device=r.device, dtype=torch.bfloat16)
    q = torch.zeros((len(r), 16, D), device=r.device, dtype=torch.bfloat16)
    ix = r.int().reshape(-1, 1, 1)
    lengths = torch.ones_like(r, dtype=torch.int32)
    for stride in (400, 416):
        rows = pack_t1_t2_patterns(codes, exps, qr[:, 0], stride)
        want, _ = unpack(rows)
        o, lse = native(q, qr, rows, ix, lengths, splits=1)
        torch.testing.assert_close(o[0].view(torch.int16), want[:, None].expand_as(o[0]).view(torch.int16), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=0)
    print("T2 PASS: native V unpack + unit-weight PV bit-exact at strides 400/416")


def t3():
    for R in (32, 192, 288):
        for stride in (400, 416):
            args, _ = inputs(R, 96, [256 - (r % 9) for r in range(R)], stride)
            got = native(*args)
            ref_error = errors(got, reference(*args), .012, .004)
            triton_error = errors(got, triton_decode(*args), .055, .05)
            print(f"T3 PASS R={R} H=96 stride={stride}: MX ref {ref_error}; Triton {triton_error}")


def t4():
    for R in (1, 4, 8, 16):
        args, _ = inputs(R, 96, ([257, 0, 33, 191] * 4)[:R])
        ref = reference(*args)
        unsplit = native(*args, splits=1)
        for S in (2, 4, 8):
            got = native(*args, splits=S)
            ref_error = errors(got, ref, .015, .004)
            split_error = errors(got, unsplit, .012, .004)
            print(f"T4 PASS R={R} S={S}: MX ref {ref_error}; vs unsplit {split_error}")


def t5():
    lengths = [0, 1, 31, 32, 33, 64, 65, 96, 97, 2048]
    for H in (16, 128):
        args, _ = inputs(len(lengths), H, lengths, 416)
        ref = reference(*args)
        for S in (1, 8):
            print(f"T5 PASS H={H} S={S}: {errors(native(*args, splits=S), ref, .018, .005)}")
        q, qr, rows, ix, ln = args
        o, lse = native(q[:0], qr[:0], rows, ix[:0], ln[:0])
        assert o.shape == (1, 0, H, D) and lse.shape == (0, H)


def t6():
    import torch.distributed as dist
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce

    assert int(os.environ.get("WORLD_SIZE", "0")) == 8, "T6 requires torchrun with eight ranks"
    dist.init_process_group("nccl")  # RCCL on ROCm
    try:
        rank = dist.get_rank()
        R, H = 4, 96
        args, _ = inputs(R, H, [0, 1, 97, TOPK], 416)
        q, qr, rows, ix, lengths = args
        indices = torch.full_like(ix, -1)
        local_rows = torch.empty((TOPK // 8, 1, 416), dtype=rows.dtype, device=rows.device)[..., :400]
        local_rows.copy_(rows[rank::8])
        counts = []
        for r, n in enumerate(lengths.tolist()):
            selected = ix[r, 0, :n]
            local_slots = selected[selected % 8 == rank] // 8
            counts.append(local_slots.numel())
            indices[r, 0, :local_slots.numel()] = local_slots
        ln = torch.tensor(counts, dtype=torch.int32, device=rows.device)
        o, lse = native(q, qr, local_rows, indices, ln)

        class Group:
            world_size = 8
            all_to_all_single = staticmethod(dist.all_to_all_single)

        out = dcp_a2a_lse_reduce(o[0], lse, Group(), is_lse_base_on_e=True)
        ls = [torch.empty_like(lse) for _ in range(8)]
        dist.all_gather(ls, lse)
        global_lse = torch.stack(ls).logsumexp(0)
        want_o, want_lse = reference(*args)
        h = slice(rank * (H // 8), (rank + 1) * (H // 8))
        error = errors((out.unsqueeze(0), global_lse[:, h]), (want_o[:, :, h], want_lse[:, h]), .015, .005)
        # Cross-check that the production Triton natural-LSE combine agrees with torch.
        partials = [torch.empty_like(o[0]) for _ in range(8)]
        dist.all_gather(partials, o[0].contiguous())
        _, actual_lse = dcp_lse_combine_triton(torch.stack(partials), torch.stack(ls), True, True)
        torch.testing.assert_close(actual_lse, global_lse, rtol=1e-5, atol=1e-5)
        print(f"T6 PASS rank={rank}: {error}", flush=True)
    finally:
        dist.destroy_process_group()


def benchmark():
    from triton.testing import do_bench_cudagraph

    for R in (1, 4, 8, 16, 32, 192, 288):
        for stride in (400, 416):
            args, _ = inputs(R, 96, [256] * R, stride)
            parity = errors(native(*args), triton_decode(*args), .055, .05)
            torch.cuda.synchronize()
            a = do_bench_cudagraph(lambda: native(*args)) * 1000
            b = do_bench_cudagraph(lambda: triton_decode(*args)) * 1000
            print(f"bench R={R} stride={stride} native={a:.2f} us triton={b:.2f} us speedup={b/a:.3f}x; {parity}")


def main():
    tests = dict(T0a=t0a, T0b=t0b, T0c=t0c, T1=t1, T2=t2, T3=t3, T4=t4, T5=t5, T6=t6, bench=benchmark)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", nargs="+", choices=tests, default=["T0a", "T0b", "T0c"])
    args = parser.parse_args()
    if any(t not in ("T0a", "T0b", "T0c") for t in args.test):
        assert os.environ.get("TRITON_INTERPRET") != "1", "GPU tests require Triton JIT"
        assert torch.version.hip and torch.cuda.is_available(), "GPU tests require ROCm/gfx950"
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
        assert arch == "gfx950", arch
    for test in args.test:
        tests[test]()


if __name__ == "__main__":
    main()
