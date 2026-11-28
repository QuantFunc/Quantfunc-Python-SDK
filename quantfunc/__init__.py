"""QuantFunc Python SDK — layered acceleration runtime for diffusion models.

This top-level facade uses PEP 562 lazy attribute loading so ``import quantfunc``
succeeds in minimal environments without eagerly importing optional dependencies
such as ``diffusers``, ``transformers``, ``accelerate``, ``huggingface_hub``, or
``modelscope``.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.1"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # L0: quantfunc.native
    "QuantFuncError": ("quantfunc.native.errors", "QuantFuncError"),
    "InvalidArgError": ("quantfunc.native.errors", "InvalidArgError"),
    "BackendError": ("quantfunc.native.errors", "BackendError"),
    "OutOfMemoryError": ("quantfunc.native.errors", "OutOfMemoryError"),
    "EngineIOError": ("quantfunc.native.errors", "EngineIOError"),
    "InternalEngineError": ("quantfunc.native.errors", "InternalEngineError"),
    "EngineBusyError": ("quantfunc.native.errors", "EngineBusyError"),
    "AuthError": ("quantfunc.native.errors", "AuthError"),
    "UnsupportedError": ("quantfunc.native.errors", "UnsupportedError"),
    "EngineNotInstallable": ("quantfunc.native.errors", "EngineNotInstallable"),
    "default_engine_dir": ("quantfunc.native.installer", "default_engine_dir"),
    "UpdateInfo": ("quantfunc.native.installer", "UpdateInfo"),
    "check_engine_update": ("quantfunc.native.installer", "check_engine_update"),
    "start_engine_install": ("quantfunc.native.installer", "start_engine_install"),
    "install_engine": ("quantfunc.native.installer", "install_engine"),
    "engine_install_status": ("quantfunc.native.installer", "engine_install_status"),
    "has_ready_so": ("quantfunc.native.installer", "has_ready_so"),
    "resolve_so_path": ("quantfunc.native.installer", "resolve_so_path"),
    "load_lib": ("quantfunc.native.loader", "load_lib"),
    "set_log_level": ("quantfunc.native.loader", "set_log_level"),
    "get_log_level": ("quantfunc.native.loader", "get_log_level"),
    "query_device_caps": ("quantfunc.native.loader", "query_device_caps"),
    "QFEngineHandle": ("quantfunc.native.handle", "QFEngineHandle"),
    # L1: quantfunc.core
    "SolConfig": ("quantfunc.core.sol", "SolConfig"),
    "ATTENTION_BACKENDS": ("quantfunc.core.config", "ATTENTION_BACKENDS"),
    "normalize_attention_backend": ("quantfunc.core.config", "normalize_attention_backend"),
    "CacheConfig": ("quantfunc.core.config", "CacheConfig"),
    "QualityConfig": ("quantfunc.core.config", "QualityConfig"),
    "EngineConfig": ("quantfunc.core.config", "EngineConfig"),
    "AuthConfig": ("quantfunc.core.config", "AuthConfig"),
    "resolve_auth": ("quantfunc.core.config", "resolve_auth"),
    "validate_api_key": ("quantfunc.core.config", "validate_api_key"),
    "redact_api_key": ("quantfunc.core.config", "redact_api_key"),
    "save_api_key": ("quantfunc.core.config", "save_api_key"),
    "StagedSchedule": ("quantfunc.core.session", "StagedSchedule"),
    "SamplingSession": ("quantfunc.core.session", "SamplingSession"),
    "AutoSessionTracker": ("quantfunc.core.session", "AutoSessionTracker"),
    "LoRAEntry": ("quantfunc.core.lora", "LoRAEntry"),
    "LoRAManager": ("quantfunc.core.lora", "LoRAManager"),
    "convert_lora_file": ("quantfunc.core.lora", "convert_lora_file"),
    "detect_lora_format": ("quantfunc.core.lora", "detect_lora_format"),
    "UNSET": ("quantfunc.core.overrides", "UNSET"),
    "RuntimeOverrides": ("quantfunc.core.overrides", "RuntimeOverrides"),
    "PresetMetadata": ("quantfunc.core.checkpoint", "PresetMetadata"),
    "resolve_checkpoint": ("quantfunc.core.checkpoint", "resolve_checkpoint"),
    "resolve_transformer_config": ("quantfunc.core.checkpoint", "resolve_transformer_config"),
    "ModelSpec": ("quantfunc.core.registry", "ModelSpec"),
    "register_model": ("quantfunc.core.registry", "register_model"),
    "get_model_class": ("quantfunc.core.registry", "get_model_class"),
    "list_registered_models": ("quantfunc.core.registry", "list_registered_models"),
    "detect_model_arch": ("quantfunc.core.registry", "detect_model_arch"),
    # L2: quantfunc.models
    "QuantFuncBaseTransformer": ("quantfunc.models.base", "QuantFuncBaseTransformer"),
    "QuantFuncKrea2Transformer2DModel": ("quantfunc.models.krea2", "QuantFuncKrea2Transformer2DModel"),
    "QuantFuncQwenImage21Transformer2DModel": (
        "quantfunc.models.qwenimage21",
        "QuantFuncQwenImage21Transformer2DModel",
    ),
    "QuantFuncLTX25Transformer3DModel": ("quantfunc.models.ltx25", "QuantFuncLTX25Transformer3DModel"),
    "QuantFuncMiniMaxH3Transformer3DModel": (
        "quantfunc.models.minimax_h3",
        "QuantFuncMiniMaxH3Transformer3DModel",
    ),
    "AutoQuantFuncTransformer": ("quantfunc.models.auto", "AutoQuantFuncTransformer"),
    # L3a: quantfunc.accelerate
    "attach_session_hooks": ("quantfunc.accelerate.hooks", "attach_session_hooks"),
    "detach_session_hooks": ("quantfunc.accelerate.hooks", "detach_session_hooks"),
    "pipe_session": ("quantfunc.accelerate.hooks", "pipe_session"),
    "apply_cache_on_pipe": ("quantfunc.accelerate.hooks", "apply_cache_on_pipe"),
    "apply_sol_on_pipe": ("quantfunc.accelerate.hooks", "apply_sol_on_pipe"),
    "enable_offload": ("quantfunc.accelerate.hooks", "enable_offload"),
    "request_scope": ("quantfunc.accelerate.serving", "request_scope"),
    # L3b: quantfunc.sampling
    "LatentBundle": ("quantfunc.sampling.bundle", "LatentBundle"),
    "split_sigmas": ("quantfunc.sampling.sigmas", "split_sigmas"),
    "denoise_sigmas": ("quantfunc.sampling.sigmas", "denoise_sigmas"),
    "build_linear_sigmas": ("quantfunc.sampling.sigmas", "build_linear_sigmas"),
    "FlowMatchRenoise": ("quantfunc.sampling.renoise", "FlowMatchRenoise"),
    "FlowMatchResMultistepScheduler": (
        "quantfunc.sampling.res_multistep",
        "FlowMatchResMultistepScheduler",
    ),
    "use_res_multistep_scheduler": (
        "quantfunc.sampling.res_multistep",
        "use_res_multistep_scheduler",
    ),
    "LatentUpscaler": ("quantfunc.sampling.upscalers", "LatentUpscaler"),
    "InterpolateLatentUpscaler": ("quantfunc.sampling.upscalers", "InterpolateLatentUpscaler"),
    "video_only_upscaler": ("quantfunc.sampling.upscalers", "video_only_upscaler"),
    "StageSpec": ("quantfunc.sampling.runner", "StageSpec"),
    "StageResult": ("quantfunc.sampling.runner", "StageResult"),
    "MultiStageRunner": ("quantfunc.sampling.runner", "MultiStageRunner"),
    "run_stages": ("quantfunc.sampling.runner", "run_stages"),
    # L4a: quantfunc.pipelines
    "load_krea2_pipeline": ("quantfunc.pipelines.krea2", "load_krea2_pipeline"),
    "make_krea2_two_stage_stages": ("quantfunc.pipelines.krea2", "make_krea2_two_stage_stages"),
    "encode_qwen3_vl_pre_norm": ("quantfunc.pipelines.qwenimage21", "encode_qwen3_vl_pre_norm"),
    "load_qwenimage21_pipeline": ("quantfunc.pipelines.qwenimage21", "load_qwenimage21_pipeline"),
    "LTX2NeuralLatentUpscaler": ("quantfunc.pipelines.ltx25", "LTX2NeuralLatentUpscaler"),
    "ensure_deterministic_scheduler": ("quantfunc.pipelines.ltx25", "ensure_deterministic_scheduler"),
    "make_ltx25_two_stage_stages": ("quantfunc.pipelines.ltx25", "make_ltx25_two_stage_stages"),
    "load_ltx25_pipeline": ("quantfunc.pipelines.ltx25", "load_ltx25_pipeline"),
    "reuse_h3_stage2_conditioning": ("quantfunc.pipelines.minimax_h3", "reuse_h3_stage2_conditioning"),
    "make_minimax_h3_two_stage_stages": (
        "quantfunc.pipelines.minimax_h3",
        "make_minimax_h3_two_stage_stages",
    ),
    "load_minimax_h3_pipeline": ("quantfunc.pipelines.minimax_h3", "load_minimax_h3_pipeline"),
    "ServingPipeline": ("quantfunc.pipelines.serving", "ServingPipeline"),
}

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {"native", "core", "models", "accelerate", "sampling", "pipelines", "cli"}
)

__all__ = ["__version__", *sorted(_LAZY_EXPORTS.keys())]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"quantfunc.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'quantfunc' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | set(_LAZY_SUBMODULES))
