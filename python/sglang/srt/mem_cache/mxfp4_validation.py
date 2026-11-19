# SPDX-License-Identifier: Apache-2.0
"""Pure launch guards for the native Kimi-K3.1 packed latent KV cache."""


def validate_mxfp4_args(cfg, *, is_hip: bool, final: bool = False) -> None:
    if cfg.speculative_draft_kv_cache_dtype == "mxfp4":
        raise ValueError("MXFP4 KV cache does not support speculative draft pools.")
    if cfg.kv_cache_dtype != "mxfp4":
        return
    if not is_hip or (cfg.device != "cuda" and (final or cfg.device is not None)):
        raise ValueError("--kv-cache-dtype mxfp4 requires ROCm on --device cuda.")
    if cfg.model_path.lower() in ("none", "dummy"):
        raise ValueError("--kv-cache-dtype mxfp4 requires a Kimi-K3.1 model.")
    if cfg.dcp_size != 1 or (cfg.attn_cp_size or 1) != 1 or cfg.enable_prefill_cp:
        raise ValueError(
            "MXFP4 KV cache requires DCP1 and no prefill context parallelism."
        )

    unsupported = {
        "--enable-hisparse": cfg.enable_hisparse,
        "--enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "--enable-lmcache": cfg.enable_lmcache,
        "--enable-flexkv": cfg.enable_flexkv,
        "--enable-session-radix-cache": cfg.enable_session_radix_cache,
        "--radix-cache-backend": cfg.radix_cache_backend is not None,
        "--enable-unified-memory": cfg.enable_unified_memory,
        "--enable-page-major-kv-layout": cfg.enable_page_major_kv_layout,
        "--enable-unified-cache-external-linker": cfg.enable_unified_cache_external_linker,
        "--enable-linker-mla-dedup": cfg.enable_linker_mla_dedup,
        "--enable-kv-cache-sharding": cfg.enable_kv_cache_sharding,
        "--disaggregation-mode": cfg.disaggregation_mode != "null",
        "--enable-pd-role-switch": cfg.enable_pd_role_switch,
        "--disaggregation-decode-enable-radix-cache": (
            cfg.disaggregation_decode_enable_radix_cache
        ),
        "--disaggregation-decode-enable-offload-kvcache": (
            cfg.disaggregation_decode_enable_offload_kvcache
        ),
        "--disaggregation-decode-retraction-backup": (
            cfg.disaggregation_decode_retraction_backup not in (None, "none")
        ),
        "--disaggregation-decode-host-receive-threshold": (
            cfg.disaggregation_decode_host_receive_threshold != 1.0
        ),
        "--speculative-algorithm": cfg.speculative_algorithm is not None,
        "--speculative-draft-model-path": cfg.speculative_draft_model_path is not None,
        "--speculative-draft-kv-cache-dtype": (
            cfg.speculative_draft_kv_cache_dtype is not None
        ),
        "--speculative-num-draft-tokens": cfg.speculative_num_draft_tokens is not None,
        "--speculative-num-steps": cfg.speculative_num_steps is not None,
        "--quantization-param-path": cfg.quantization_param_path is not None,
        "--prefill-only-disable-kv-cache": cfg.prefill_only_disable_kv_cache,
    }
    rejected = [flag for flag, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError("MXFP4 KV cache does not support " + ", ".join(rejected) + ".")
    if cfg.page_size != 1 and (final or cfg.page_size is not None):
        raise ValueError("MXFP4 KV cache requires --page-size 1.")
    for field in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        backend = getattr(cfg, field)
        if backend is not None and backend != "dsa":
            raise ValueError(
                f"MXFP4 KV cache requires the DSA attention backend, got {backend}."
            )
    if final and (
        (cfg.prefill_attention_backend or cfg.attention_backend) != "dsa"
        or (cfg.decode_attention_backend or cfg.attention_backend) != "dsa"
    ):
        raise ValueError(
            "MXFP4 KV cache requires resolved DSA prefill and decode attention."
        )
    for field in ("dsa_prefill_backend", "dsa_decode_backend"):
        backend = getattr(cfg, field)
        if backend != "triton" and (final or backend is not None):
            raise ValueError(
                f"MXFP4 KV cache requires --{field.replace('_', '-')} triton."
            )


def validate_mxfp4_model(model_config) -> None:
    hf_config = model_config.hf_config
    architecture = (getattr(hf_config, "architectures", None) or [None])[0]
    if architecture != "KimiK31ForConditionalGeneration":
        raise ValueError("MXFP4 KV cache only supports KimiK31ForConditionalGeneration.")
    if (model_config.kv_lora_rank, model_config.qk_rope_head_dim) != (512, 64):
        raise ValueError("MXFP4 KV cache requires latent512 + RoPE64 (400-byte rows).")
    if str(model_config.dtype) != "torch.bfloat16":
        raise ValueError("MXFP4 KV cache requires BF16 model computation.")
    tc = model_config.hf_text_config
    if (
        getattr(tc, "index_head_dim", None) != 128
        or getattr(tc, "index_n_heads", None) != 32
        or getattr(tc, "index_topk", None) != 2048
        or getattr(tc, "index_kpool", 1) != 1
        or getattr(tc, "index_kpool_compress", False)
        or getattr(tc, "index_rope_dim", None) != 0
    ):
        raise ValueError(
            "MXFP4 KV cache requires the Kimi-K3.1 FP8 NoPE indexer "
            "(dim128, heads32, topk2048, kpool1)."
        )
