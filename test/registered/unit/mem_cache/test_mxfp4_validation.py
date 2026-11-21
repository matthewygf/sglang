"""Launch policy for native MXFP4 device radix, page size, and HiCache L2."""

from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.mxfp4_validation import validate_mxfp4_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def kimi_model_config():
    import torch

    hf = SimpleNamespace(
        architectures=["KimiK31ForConditionalGeneration"],
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_rope_dim=0,
    )
    return SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


def config(**changes):
    """A valid MXFP4 launch config. Mirrors the qualified Kimi-K3.1 launch."""
    cfg = SimpleNamespace(
        kv_cache_dtype="mxfp4",
        speculative_draft_kv_cache_dtype=None,
        model_path="/models/kimi-k31",
        device="cuda",
        tp_size=8,
        attn_dp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        page_size=1,
        disable_radix_cache=False,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        hicache_io_backend="direct",
        hicache_mem_layout="page_first_direct",
        enable_lmcache=False,
        enable_flexkv=False,
        enable_session_radix_cache=False,
        radix_cache_backend=None,
        enable_unified_memory=False,
        enable_page_major_kv_layout=False,
        enable_unified_cache_external_linker=False,
        enable_linker_mla_dedup=False,
        enable_kv_cache_sharding=False,
        disaggregation_mode="null",
        enable_pd_role_switch=False,
        disaggregation_decode_enable_radix_cache=False,
        disaggregation_decode_enable_offload_kvcache=False,
        disaggregation_decode_retraction_backup=None,
        disaggregation_decode_host_receive_threshold=1.0,
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
        quantization_param_path=None,
        prefill_only_disable_kv_cache=False,
        attention_backend="dsa",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        dsa_prefill_backend="triton",
        dsa_decode_backend="triton",
        dsa_paged_mqa_logits_backend="auto",
        enable_dsa_fp4_indexer=False,
        enable_amd_monokernel=False,
    )
    for key, value in changes.items():
        assert hasattr(cfg, key), f"unknown config field {key!r}"
        setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("disable_radix_cache", [False, True])
def test_native_device_radix_is_admitted(disable_radix_cache, final):
    # The behaviour this change introduces: radix is no longer a launch decision.
    validate_mxfp4_args(
        config(disable_radix_cache=disable_radix_cache), is_hip=True, final=final
    )


@pytest.mark.parametrize(
    ("requested_page", "preshuffle", "model_page", "expected_page"),
    [
        (None, False, None, 1),
        (1, False, None, 1),
        (64, True, 64, 64),
        (64, False, None, None),
        (None, True, 64, None),
    ],
    ids=["default", "legacy", "opt-in", "preshuffle-disabled", "model-override"],
)
def test_native_page_size_resolution(
    monkeypatch, requested_page, preshuffle, model_page, expected_page
):
    from sglang.srt.arg_groups import (
        cuda_graph_hook,
        muse_spark_hook,
        pipeline,
    )
    from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
    from sglang.srt.layers.attention.dsa import utils as dsa_utils
    from sglang.srt.runtime_context import override_platform
    from sglang.srt.server_args import ServerArgs

    monkeypatch.delenv("SGLANG_AITER_KV_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr(
        dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: preshuffle
    )
    monkeypatch.setattr(muse_spark_hook, "apply_muse_spark_env_defaults", lambda _: None)
    monkeypatch.setattr(
        cuda_graph_hook, "finalize_cuda_graph_prefill_max_context", lambda _: None
    )

    # Run the real default and final validation without fetching a checkpoint.
    def model_override(hook, args):
        if (
            hook.__name__ == "handle_model_specific_adjustments"
            and model_page is not None
        ):
            declare_resolution(args, "test.model_override", page_size=model_page)

    monkeypatch.setattr(pipeline, "run_hook", model_override)
    args = ServerArgs(
        model_path="/models/kimi-k31",
        device="cuda",
        kv_cache_dtype="mxfp4",
        page_size=requested_page,
        boundary_reduction="rs+rsv",
        attention_backend="dsa",
        dsa_prefill_backend="triton",
        dsa_decode_backend="triton",
    )
    args._model_config = kimi_model_config()
    with override_platform(is_hip=True):
        if expected_page is None:
            with pytest.raises(
                ValueError, match="explicit --page-size 64 with AITER preshuffle"
            ):
                args.resolve_once()
        else:
            args.resolve_once()
            assert resolution_result(args, "page_size") == expected_page


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize(
    ("page_size", "io_backend", "layout"),
    [
        (1, "direct", "page_first_direct"),
        (64, "direct", "page_first_direct"),
        (64, "kernel", "page_first"),
    ],
)
def test_native_hicache_l2_is_admitted(page_size, io_backend, layout, final):
    cfg = config(
        page_size=page_size,
        enable_hierarchical_cache=True,
        hicache_io_backend=io_backend,
        hicache_mem_layout=layout,
    )
    validate_mxfp4_args(cfg, is_hip=True, native_page64=page_size == 64, final=final)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("page_size", [1, 64])
def test_native_fp4_indexer_is_rejected(page_size, final):
    cfg = config(page_size=page_size, enable_dsa_fp4_indexer=True)
    with pytest.raises(ValueError, match="does not support --enable-dsa-fp4-indexer"):
        validate_mxfp4_args(cfg, is_hip=True, native_page64=page_size == 64, final=final)


@pytest.mark.parametrize("final", [False, True])
def test_native_hicache_l3_is_rejected(final):
    cfg = config(enable_hierarchical_cache=True, hicache_storage_backend="file")
    with pytest.raises(ValueError, match="does not support --hicache-storage-backend"):
        validate_mxfp4_args(cfg, is_hip=True, final=final)


@pytest.mark.parametrize(
    ("page_size", "io_backend", "layout"),
    [(1, "kernel", "page_first"), (64, "kernel", "layer_first")],
)
def test_native_hicache_rejects_unsupported_resolved_io(page_size, io_backend, layout):
    cfg = config(
        page_size=page_size,
        enable_hierarchical_cache=True,
        hicache_io_backend=io_backend,
        hicache_mem_layout=layout,
    )
    with pytest.raises(ValueError, match="MXFP4 HiCache requires --hicache-io-backend"):
        validate_mxfp4_args(cfg, is_hip=True, native_page64=page_size == 64, final=True)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("world", [2, 8])
def test_native_dcp_is_admitted(world, page_size, final):
    validate_mxfp4_args(
        config(dcp_size=world, page_size=page_size),
        is_hip=True,
        final=final,
        native_page64=page_size == 64,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dcp_size", 0, "power-of-two --dcp-size"),
        ("dcp_size", 3, "power-of-two --dcp-size"),
        ("dcp_size", 16, "must divide the attention TP width"),
        ("attn_cp_size", 2, "--attn-cp-size 1"),
        ("enable_prefill_cp", True, "no --enable-prefill-cp"),
        ("page_size", 64, "explicit --page-size 64 with AITER preshuffle"),
        (
            "enable_session_radix_cache",
            True,
            "does not support --enable-session-radix-cache",
        ),
        ("radix_cache_backend", "custom", "does not support --radix-cache-backend"),
        ("enable_hisparse", True, "does not support --enable-hisparse"),
        ("enable_lmcache", True, "does not support --enable-lmcache"),
        ("enable_flexkv", True, "does not support --enable-flexkv"),
        ("enable_kv_cache_sharding", True, "does not support --enable-kv-cache-sharding"),
        ("enable_unified_memory", True, "does not support --enable-unified-memory"),
        ("disaggregation_mode", "prefill", "does not support --disaggregation-mode"),
        ("speculative_algorithm", "EAGLE", "does not support --speculative-algorithm"),
        (
            "quantization_param_path",
            "scales.json",
            "does not support --quantization-param-path",
        ),
        (
            "prefill_only_disable_kv_cache",
            True,
            "does not support --prefill-only-disable-kv-cache",
        ),
        (
            "dsa_prefill_backend",
            "triton_sparse_mla",
            "requires --dsa-prefill-backend triton",
        ),
        (
            "dsa_decode_backend",
            "triton_sparse_mla",
            "requires --dsa-decode-backend triton",
        ),
    ],
)
def test_other_guards_still_reject(field, value, message):
    # Regression floor for guards outside the page-64 and HiCache L2 opt-ins.
    with pytest.raises(ValueError, match=message):
        validate_mxfp4_args(config(**{field: value}), is_hip=True, final=True)


@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8_e4m3"])
def test_other_target_dtypes_return_early(dtype):
    # Non-MXFP4 targets exit before touching radix or any later field, so every
    # field after the early return is deliberately absent.
    cfg = SimpleNamespace(
        kv_cache_dtype=dtype, speculative_draft_kv_cache_dtype=None
    )
    validate_mxfp4_args(cfg, is_hip=False, final=True)


def test_mxfp4_draft_is_rejected_even_for_other_target():
    # The draft-pool rejection sits before the non-MXFP4 early return on purpose.
    cfg = SimpleNamespace(
        kv_cache_dtype="fp8_e4m3", speculative_draft_kv_cache_dtype="mxfp4"
    )
    with pytest.raises(ValueError, match="does not support speculative draft pools"):
        validate_mxfp4_args(cfg, is_hip=True, final=True)


def test_in_tree_uint8_row_sizing_requires_the_native_recipe(monkeypatch):
    import torch

    from sglang.srt.mem_cache import kv_cache_configurator as pools
    from sglang.srt.mem_cache import kv_cache_dtype as dtypes

    cfg = config()
    model = kimi_model_config()
    monkeypatch.setattr(dtypes, "_is_hip", True)
    monkeypatch.setattr(pools, "_is_hip", True)
    monkeypatch.setattr(pools, "get_model", lambda: cfg)
    monkeypatch.setattr(pools, "get_exec", lambda: SimpleNamespace(kernel=cfg))
    recipe, storage = dtypes.configure_kv_cache_dtype(
        server_args_kv_cache_dtype="mxfp4",
        quant_config=None,
        model_dtype=model.dtype,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend=None,
    )
    assert (recipe, storage) == ("mxfp4", torch.uint8)
    row_bytes = pools.calculate_mla_kv_cache_dim(
        model_config=model, kv_cache_dtype=storage
    )
    assert row_bytes == 400
    cfg.kv_cache_dtype = "auto"
    with pytest.raises(ValueError, match="requires ROCm Triton DSA with mxfp4"):
        pools.calculate_mla_kv_cache_dim(model_config=model, kv_cache_dtype=storage)


def test_native_qualification_cli():
    from sglang.srt.mem_cache.platform_kv_cache_dtype import (
        registered_kv_cache_storage_dtype,
    )
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(
        "--model-path /models/kimi-k31 --kv-cache-dtype mxfp4 --page-size 64 "
        "--enable-hierarchical-cache --enable-metrics --max-mamba-cache-size 32 "
        "--speculative-accept-threshold-single=1.0 "
        "--speculative-accept-threshold-acc=1.0".split()
    )
    validate_mxfp4_args(args, is_hip=True, native_page64=True)
    assert args.kv_cache_dtype == "mxfp4"
    assert args.page_size == 64 and args.max_mamba_cache_size == 32
    assert args.enable_hierarchical_cache and args.enable_metrics
    assert registered_kv_cache_storage_dtype("mxfp4") is None


@pytest.mark.parametrize(
    "option",
    [
        "--speculative-accept-threshold-single=0.9",
        "--speculative-use-rejection-sampling",
        "--no-speculative-enable-w4a4-mxfp4-megamoe",
        "--speculative-algorithm DFLASH",
    ],
)
def test_non_default_speculative_cli_options_rejected(option):
    from sglang.srt.server_args import prepare_server_args

    args = prepare_server_args(
        "--model-path /models/kimi-k31 --kv-cache-dtype mxfp4".split() + option.split()
    )
    with pytest.raises(ValueError, match="MXFP4 KV cache does not support"):
        validate_mxfp4_args(args, is_hip=True)


def test_speculative_programmatic_option_rejected():
    cfg = config()
    cfg.speculative_eagle_topk = 2
    with pytest.raises(ValueError, match="--speculative-eagle-topk"):
        validate_mxfp4_args(cfg, is_hip=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
