"""Ordinary device radix is allowed for native MXFP4; other launch guards remain.

The MXFP4 launch validator used to require ``--disable-radix-cache``. That was a
policy check, not a kernel invariant: the radix tree stores int64 KV indices and
never reads the packed 400-byte rows, and the configured Triton sparse-MLA reader
already accepts the packed format. This file pins the two properties that matter
after removing it:

1. radix on and radix off are both admitted, in both validator modes;
2. every other guard in the same function still rejects.

    python -m pytest test/registered/unit/mem_cache/test_mxfp4_validation.py -v
"""

from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.mxfp4_validation import validate_mxfp4_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def config(**changes):
    """A valid MXFP4 launch config. Mirrors the qualified Kimi-K3.1 launch."""
    cfg = SimpleNamespace(
        kv_cache_dtype="mxfp4",
        speculative_draft_kv_cache_dtype=None,
        model_path="/models/kimi-k31",
        device="cuda",
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        page_size=1,
        disable_radix_cache=False,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
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
    ("field", "value", "message"),
    [
        ("dcp_size", 2, "requires DCP1"),
        ("attn_cp_size", 2, "requires DCP1"),
        ("enable_prefill_cp", True, "requires DCP1"),
        ("page_size", 64, "requires --page-size 1"),
        (
            "enable_hierarchical_cache",
            True,
            "does not support --enable-hierarchical-cache",
        ),
        (
            "enable_session_radix_cache",
            True,
            "does not support --enable-session-radix-cache",
        ),
        ("radix_cache_backend", "custom", "does not support --radix-cache-backend"),
        (
            "dsa_prefill_backend",
            "triton_sparse_mla",
            "requires --dsa-prefill-backend triton",
        ),
    ],
)
def test_other_guards_still_reject(field, value, message):
    # Regression floor: the change must not widen the MXFP4 envelope. Each case
    # enables radix, so the removed error cannot accidentally satisfy it.
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
