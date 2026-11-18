"""Warm the real MXFP4 sparse-prefill autotune buckets before pool sizing."""
import logging
import time

import torch

logger = logging.getLogger(__name__)
# Matches the two regimes around triton_sparse_mla._LONG_PREFILL_SEQ_THRESHOLD.
WARMUP_ROWS = (4096, 32768)


def mxfp4_prefill_warmup_rows(*, enabled, kv_cache_dtype_str, attention_backend,
                              prefill_attention_backend, dsa_prefill_backend,
                              is_gfx95, is_draft_worker):
    if (enabled and not is_draft_worker and is_gfx95
            and kv_cache_dtype_str == "mxfp4"
            and (prefill_attention_backend or attention_backend) == "dsa"
            and dsa_prefill_backend == "triton"):
        return WARMUP_ROWS
    return ()


def run_mxfp4_prefill_warmup(device, num_heads, topk, sm_scale, rows, *,
                              fwd=None, d_v=512, d_tail=64, kv_slots=4096):
    if fwd is None:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
        fwd = triton_sparse_mla_fwd
    kv = torch.zeros((kv_slots + 1, 1, 400), dtype=torch.uint8, device=device)
    seconds = {}
    for n in rows:
        start = time.monotonic()
        q = torch.zeros((n, num_heads, d_v + d_tail), dtype=torch.bfloat16, device=device)
        idx = (torch.arange(topk, dtype=torch.int32, device=device) % kv_slots + 1).repeat(n, 1)
        fwd(q[..., :d_v], q[..., d_v:], kv, idx.unsqueeze(1), sm_scale, d_v,
            topk_length=None, max_topk_length=None, kv_format="mxfp4")
        torch.cuda.synchronize(device)
        seconds[n] = time.monotonic() - start
    configs = {}
    try:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import _sparse_mla_fwd_split_dim_kernel
        configs = {str(key): str(config) for key, config in _sparse_mla_fwd_split_dim_kernel.cache.items()}
    except Exception:
        pass
    return {"seconds": seconds, "configs": configs}


def maybe_warmup_mxfp4_sparse_prefill(model_runner):
    start = time.monotonic()
    cleanup = False
    try:
        from sglang.srt.configs.model_config import get_dsa_index_topk
        from sglang.srt.environ import envs
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.runtime_context import get_exec
        from sglang.srt.utils.common import is_gfx95_supported

        args = get_exec().kernel
        rows = mxfp4_prefill_warmup_rows(
            enabled=envs.SGLANG_MXFP4_PREFILL_WARMUP.get(),
            kv_cache_dtype_str=model_runner.kv_cache_dtype_str,
            attention_backend=args.attention_backend,
            prefill_attention_backend=args.prefill_attention_backend,
            dsa_prefill_backend=args.dsa_prefill_backend,
            is_gfx95=is_gfx95_supported(), is_draft_worker=model_runner.is_draft_worker,
        )
        if not rows:
            return
        cleanup = True
        attn = next((m.attn_mqa for m in model_runner.model.modules()
                     if isinstance(getattr(m, "attn_mqa", None), RadixAttention)), None)
        if attn is None:
            logger.info("MXFP4 sparse prefill warmup skipped: no MLA attention found")
            return
        topk = get_dsa_index_topk(model_runner.model_config.hf_config)
        result = run_mxfp4_prefill_warmup(
            model_runner.device, attn.tp_q_head_num, topk, attn.scaling, rows, d_v=attn.v_head_dim)
        logger.info("MXFP4 sparse prefill warmup done: rows=%s heads=%s topk=%s seconds=%s total=%.1fs configs=%s",
                    rows, attn.tp_q_head_num, topk, result["seconds"], time.monotonic() - start, result["configs"])
    except Exception:
        logger.warning("MXFP4 sparse prefill warmup failed after %.1fs; first long prefill will autotune live",
                       time.monotonic() - start, exc_info=True)
    finally:
        if cleanup:
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.warning("MXFP4 sparse prefill warmup cache cleanup failed", exc_info=True)
