"""L1 model-agnostic runtime core: Sol, Config, Auth, Session, LoRA, Checkpoint/Presets, and Registry."""

from __future__ import annotations

from .checkpoint import (
    PresetMetadata,
    bundled_preset_dir,
    bundled_presets_root,
    list_bundled_presets,
    name_matches_hints,
    resolve_checkpoint,
    resolve_transformer_config,
    sniff_safetensors_header,
)
from .config import (
    ATTENTION_BACKENDS,
    DEFAULT_MAX_CONTEXT_SEQ,
    DEFAULT_QUALITY_RATIO,
    DEFAULT_SERVER_URL,
    DEFAULT_TIMESTEP_SCALE,
    MAX_VRAM_BUDGET_MB,
    OFFLOAD_MODES,
    PRODUCER_STREAM_MODES,
    WIRE_QUALITY_RATIO_KEY,
    AuthConfig,
    CacheConfig,
    EngineConfig,
    QualityConfig,
    default_cache_dir,
    default_config_path,
    normalize_attention_backend,
    redact_api_key,
    resolve_auth,
    save_api_key,
    validate_api_key,
)
from .lora import (
    LoRAEntry,
    LoRAManager,
    convert_lora_file,
    detect_lora_format,
    unsupported_lora_kind,
)
from .registry import (
    ModelSpec,
    detect_model_arch,
    get_model_class,
    get_model_spec,
    list_registered_models,
    register_model,
    unregister_model,
)
from .session import (
    DEFAULT_AUTO_SESSION_MAX_STEPS,
    AutoSessionTracker,
    CtxKeyAssigner,
    SamplingSession,
    StagedSchedule,
)
from .sol import (
    SolConfig,
    from_legacy,
    legacy_tau,
    percent_to_sigma,
)

__all__ = [
    # sol
    "SolConfig",
    "legacy_tau",
    "from_legacy",
    "percent_to_sigma",
    # config
    "DEFAULT_TIMESTEP_SCALE",
    "DEFAULT_MAX_CONTEXT_SEQ",
    "DEFAULT_QUALITY_RATIO",
    "DEFAULT_SERVER_URL",
    "WIRE_QUALITY_RATIO_KEY",
    "ATTENTION_BACKENDS",
    "OFFLOAD_MODES",
    "PRODUCER_STREAM_MODES",
    "MAX_VRAM_BUDGET_MB",
    "default_config_path",
    "default_cache_dir",
    "normalize_attention_backend",
    "CacheConfig",
    "QualityConfig",
    "EngineConfig",
    "AuthConfig",
    "resolve_auth",
    "validate_api_key",
    "redact_api_key",
    "save_api_key",
    # session
    "DEFAULT_AUTO_SESSION_MAX_STEPS",
    "CtxKeyAssigner",
    "StagedSchedule",
    "SamplingSession",
    "AutoSessionTracker",
    # lora
    "LoRAEntry",
    "detect_lora_format",
    "unsupported_lora_kind",
    "convert_lora_file",
    "LoRAManager",
    # checkpoint
    "PresetMetadata",
    "name_matches_hints",
    "sniff_safetensors_header",
    "bundled_presets_root",
    "list_bundled_presets",
    "bundled_preset_dir",
    "resolve_checkpoint",
    "resolve_transformer_config",
    # registry
    "ModelSpec",
    "register_model",
    "unregister_model",
    "get_model_spec",
    "get_model_class",
    "list_registered_models",
    "detect_model_arch",
]
