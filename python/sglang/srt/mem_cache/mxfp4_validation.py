# SPDX-License-Identifier: Apache-2.0
"""Pure launch guards for the native Kimi-K3.1 packed latent KV cache."""


MXFP4_DFLASH_BATCH64_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def validate_mxfp4_args(
    cfg, *, is_hip: bool, final: bool = False, native_page64: bool = False
) -> None:
    if cfg.speculative_draft_kv_cache_dtype == "mxfp4":
        raise ValueError("MXFP4 KV cache does not support speculative draft pools.")
    if cfg.kv_cache_dtype != "mxfp4":
        return
    if not is_hip or (cfg.device != "cuda" and (final or cfg.device is not None)):
        raise ValueError("--kv-cache-dtype mxfp4 requires ROCm on --device cuda.")
    if cfg.model_path.lower() in ("none", "dummy"):
        raise ValueError("--kv-cache-dtype mxfp4 requires a Kimi-K3.1 model.")
    if (cfg.attn_cp_size or 1) != 1 or cfg.enable_prefill_cp:
        raise ValueError(
            "MXFP4 KV cache requires --attn-cp-size 1 and no --enable-prefill-cp."
        )
    # alloc_extend_kernel uses tl.arange(0, page_size * dcp_size).
    if cfg.dcp_size < 1 or cfg.dcp_size & (cfg.dcp_size - 1):
        raise ValueError("MXFP4 KV cache requires a positive power-of-two --dcp-size.")
    if cfg.dcp_size > 1 and (
        cfg.attn_dp_size < 1 or cfg.tp_size % (cfg.dcp_size * cfg.attn_dp_size)
    ):
        raise ValueError("MXFP4 --dcp-size must divide the attention TP width.")

    dflash = cfg.speculative_algorithm == "DFLASH"
    unsupported = {
        "--enable-hisparse": cfg.enable_hisparse,
        "--hicache-storage-backend": cfg.hicache_storage_backend is not None,
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
        "--speculative-algorithm": cfg.speculative_algorithm is not None and not dflash,
        "--speculative-draft-model-path": (
            cfg.speculative_draft_model_path is not None and not dflash
        ),
        "--speculative-draft-kv-cache-dtype": (
            cfg.speculative_draft_kv_cache_dtype is not None and not dflash
        ),
        "--speculative-num-draft-tokens": (
            cfg.speculative_num_draft_tokens is not None and not (dflash and final)
        ),
        "--speculative-num-steps": (
            cfg.speculative_num_steps is not None and not (dflash and final)
        ),
        "--quantization-param-path": cfg.quantization_param_path is not None,
        "--prefill-only-disable-kv-cache": cfg.prefill_only_disable_kv_cache,
        "--enable-dsa-fp4-indexer": cfg.enable_dsa_fp4_indexer,
    }
    rejected = [flag for flag, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError("MXFP4 KV cache does not support " + ", ".join(rejected) + ".")
    if dflash:
        _validate_mxfp4_dflash(cfg, final=final, native_page64=native_page64)
    if not final:
        # Check parsed values before resolution inherits draft defaults.
        from sglang.srt.arg_groups.fields.spec import Spec

        defaults = Spec()
        for name in Spec.__struct_fields__:
            if dflash and name in (
                "speculative_algorithm",
                "speculative_draft_model_path",
                "speculative_draft_kv_cache_dtype",
                "speculative_dflash_block_size",
                "speculative_draft_attention_backend",
                "speculative_use_block_verification",
            ):
                continue
            if name.startswith("speculative_"):
                default = getattr(defaults, name)
                if getattr(cfg, name, default) != default:
                    raise ValueError(
                        f"MXFP4 KV cache does not support --{name.replace('_', '-')}."
                    )
    if cfg.page_size != (64 if native_page64 else 1) and (
        final or cfg.page_size is not None
    ):
        raise ValueError(
            "MXFP4 KV cache requires --page-size 1 (legacy) or explicit "
            "--page-size 64 with AITER preshuffle available."
        )
    if native_page64 and cfg.dsa_paged_mqa_logits_backend not in ("auto", "aiter"):
        raise ValueError(
            "MXFP4 page 64 requires --dsa-paged-mqa-logits-backend auto or aiter."
        )
    # DFLASH already passed its narrow scope checks above; plugin validation is separate.
    if native_page64 and cfg.enable_amd_monokernel and not dflash:
        raise ValueError("MXFP4 page 64 does not support --enable-amd-monokernel.")
    if final and cfg.enable_hierarchical_cache:
        io_layout = (cfg.hicache_io_backend, cfg.hicache_mem_layout)
        if io_layout != ("direct", "page_first_direct") and not (
            native_page64 and io_layout == ("kernel", "page_first")
        ):
            raise ValueError(
                "MXFP4 HiCache requires --hicache-io-backend direct with "
                "--hicache-mem-layout page_first_direct, or kernel with page_first "
                "at page size 64."
            )
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


def _validate_mxfp4_dflash(cfg, *, final: bool, native_page64: bool) -> None:
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_platform

    if not (
        native_page64
        and get_platform().is_gfx95
        and (cfg.tp_size, cfg.dcp_size) in {(8, 8), (8, 1)}
        and cfg.attn_dp_size == cfg.dp_size == cfg.pp_size == cfg.nnodes == 1
        and cfg.speculative_draft_model_path
        and cfg.speculative_draft_kv_cache_dtype == "fp8_e4m3"
        and cfg.speculative_dflash_block_size == 6
        and cfg.speculative_draft_attention_backend == "triton"
    ):
        raise ValueError(
            "MXFP4 KV cache does not support DFLASH outside the experimental "
            "gfx95 TP8/DCP8 or TP8/DCP1 single-node configuration with explicit page 64, "
            "a draft model, fp8_e4m3 draft KV, block size 6 and Triton draft attention."
        )
    verify_backend = cfg.linear_attn_verify_backend
    if final:
        from sglang.srt.layers.attention.linear.utils import (
            resolve_linear_attn_verify_backend,
        )

        verify_backend = resolve_linear_attn_verify_backend(
            cfg.linear_attn_decode_backend or cfg.linear_attn_backend, verify_backend
        ).value
    if not (
        cfg.enable_linear_replayssm_spec
        and verify_backend in (None, "triton")
        and envs.SGLANG_RAGGED_VERIFY_MODE.get() == "static"
        and envs.SGLANG_SIMULATE_ACC_LEN.get() < 0
    ):
        raise ValueError(
            "MXFP4 DFLASH requires static unsimulated verify with Triton ReplaySSM."
        )
    radix_overlap = (
        not cfg.disable_radix_cache
        and not cfg.disable_overlap_schedule
        and cfg.dcp_comm_backend == "a2a"
        and cfg.enable_amd_monokernel
        and not cfg.enable_mixed_chunk
        and not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get()
        and (
            cfg.speculative_dflash_draft_kv_ring
            if final
            else envs.SGLANG_ENABLE_DFLASH_DRAFT_KV_RING.get()
        )
    )
    if cfg.enable_hierarchical_cache or not (
        radix_overlap or (cfg.disable_radix_cache and cfg.disable_overlap_schedule)
    ):
        raise ValueError(
            "MXFP4 DFLASH requires no radix cache, HiCache or overlap schedule, "
            "except for device radix and overlap together with a2a, AMD monokernel, "
            "the DFLASH draft KV ring, no mixed chunks, and no separate overlap plan stream."
        )
    if radix_overlap and final and (
        cfg.mamba_radix_cache_strategy != "extra_buffer"
        or cfg.mamba_ssm_dtype != "float32"
        or cfg.enable_int8_mamba_checkpoint
    ):
        raise ValueError(
            "MXFP4 DFLASH device radix with overlap requires extra_buffer and "
            "unquantized float32 recurrent checkpoints."
        )
    if cfg.enable_strict_thinking:
        raise ValueError("MXFP4 DFLASH does not support --enable-strict-thinking.")
    if final:
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        graphs = cfg.cuda_graph_config
        buckets = graphs.decode.bs if graphs is not None else None
        if isinstance(buckets, list) and all(type(bs) is int for bs in buckets):
            buckets = tuple(buckets)
        else:
            buckets = ()
        locked = getattr(cfg, "_cuda_graph_config_locked", set())
        explicit_buckets = (Phase.DECODE, "bs") in locked
        max_batch64 = MXFP4_DFLASH_BATCH64_BUCKETS[-1]
        automatic_batch64 = (
            not explicit_buckets
            and buckets[:1] == (1,)
            and buckets[-1] == max_batch64
            and buckets == tuple(sorted(set(buckets)))
        )
        explicit_multi_batch = (
            explicit_buckets
            and (Phase.DECODE, "backend") in locked
            and buckets in ((1, 2), MXFP4_DFLASH_BATCH64_BUCKETS)
        )
        multi_batch = automatic_batch64 or explicit_multi_batch
        if not (
            cfg.speculative_num_draft_tokens == 6
            and cfg.speculative_num_steps == cfg.speculative_eagle_topk == 1
            and cfg.speculative_attention_mode == "decode"
            and graphs is not None
            and graphs.prefill.backend == Backend.DISABLED
            and (
                graphs.decode.backend == Backend.DISABLED
                or (
                    graphs.decode.backend == Backend.FULL
                    and (
                        buckets == (1,)
                        or (
                            multi_batch
                            and type(graphs.decode.max_bs) is int
                            and graphs.decode.max_bs == buckets[-1]
                        )
                    )
                )
            )
        ):
            raise ValueError(
                "MXFP4 DFLASH must resolve to block-6 decode verification with "
                "prefill CUDA graphs disabled, decode CUDA graphs disabled or full "
                "with [1], explicit [1, 2], explicit [1, 2, 4, 8, 16, 32, 64], "
                "or automatic buckets through batch 64."
            )
        if graphs.decode.backend == Backend.FULL and buckets == (1, 2):
            max_bs = buckets[-1]
            requests = cfg.max_running_requests
            state = cfg.max_mamba_cache_size
            min_state = max_bs
            if radix_overlap:
                from sglang.srt.mem_cache.allocation_sizing import (
                    get_device_mamba_radix_slots_per_request,
                )

                min_state *= get_device_mamba_radix_slots_per_request(
                    extra_buffer=True, overlap=True
                )
            if not (
                type(requests) is int
                and requests >= max_bs
                and (state is None or (type(state) is int and state >= min_state))
            ):
                raise ValueError(
                    "MXFP4 DFLASH with full decode CUDA graphs requires capacity "
                    f"for {max_bs} requests and {min_state} state slots. Set "
                    f"--max-running-requests to at least {max_bs} and leave "
                    "--max-mamba-cache-size unset for automatic sizing, or "
                    f"set it to at least {min_state}."
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
