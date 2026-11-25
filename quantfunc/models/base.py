"""Base drop-in PyTorch/diffusers transformer backed by the QuantFunc PC Engine."""

from __future__ import annotations

import ctypes
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from quantfunc.core.checkpoint import (
    PresetMetadata,
    bundled_preset_dir,
    resolve_checkpoint,
    resolve_transformer_config,
)
from quantfunc.core.config import (
    DEFAULT_MAX_CONTEXT_SEQ,
    DEFAULT_QUALITY_RATIO,
    DEFAULT_TIMESTEP_SCALE,
    WIRE_QUALITY_RATIO_KEY,
    AuthConfig,
    CacheConfig,
    EngineConfig,
    normalize_attention_backend,
    resolve_auth,
)
from quantfunc.core.lora import LoRAEntry, LoRAManager, normalize_lora_entries
from quantfunc.core.registry import get_model_spec
from quantfunc.core.session import (
    DEFAULT_AUTO_SESSION_MAX_STEPS,
    AutoSessionTracker,
    SamplingSession,
    StagedSchedule,
)
from quantfunc.core.sol import SolConfig
from quantfunc.native.capi import (
    QF_DTYPE_BF16,
    QF_DTYPE_FP16,
    QF_DTYPE_FP32,
    QF_UINT64_MAX,
    ColdNeed,
    QFStepAvParams,
    QFStepParams,
    QFStepRefsParams,
)
from quantfunc.native.handle import QFEngineHandle, estimate_vram, make_create_params
from quantfunc.native.loader import load_lib

logger = logging.getLogger(__name__)

_CUDA_STREAM_LEGACY: int = 1  # cudaStreamLegacy sentinel (0x1) for GPU-side default-stream sync
_MAX_SESSION_CTX_KEEPALIVE: int = 16

_TORCH_TO_QF_DTYPE: dict[torch.dtype, int] = {
    torch.float32: QF_DTYPE_FP32,
    torch.float16: QF_DTYPE_FP16,
    torch.bfloat16: QF_DTYPE_BF16,
}

try:
    from diffusers.configuration_utils import ConfigMixin as _BaseConfigMixin
    from diffusers.models.cache_utils import CacheMixin as _BaseCacheMixin
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.modeling_utils import ModelMixin as _BaseModelMixin

    _HAS_DIFFUSERS = True
except ImportError:
    _HAS_DIFFUSERS = False

    class _FrozenConfigDict(OrderedDict):  # type: ignore[type-arg]
        """Minimal dict-like frozen config container when diffusers is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for key, value in self.items():
                super().__setattr__(key, value)
            super().__setattr__("_frozen", True)

        def __setattr__(self, name: str, value: Any) -> None:
            if getattr(self, "_frozen", False):
                raise AttributeError(f"Cannot mutate frozen config attribute {name!r}")
            super().__setattr__(name, value)

        def __setitem__(self, key: str, value: Any) -> None:
            if getattr(self, "_frozen", False):
                raise TypeError(f"Cannot mutate frozen config key {key!r}")
            super().__setitem__(key, value)

    class Transformer2DModelOutput(OrderedDict):  # type: ignore[no-redef,type-arg]
        """Minimal drop-in output container when diffusers is not installed."""

        def __init__(self, sample: torch.Tensor) -> None:
            super().__init__(sample=sample)
            self.sample = sample

        def __getitem__(self, item: Any) -> Any:
            if isinstance(item, int):
                return tuple(self.values())[item]
            return super().__getitem__(item)

    class _BaseModelMixin(nn.Module):  # type: ignore[no-redef]
        """Pure-PyTorch shim matching diffusers ModelMixin."""

    class _BaseConfigMixin:  # type: ignore[no-redef]
        """Pure-PyTorch shim matching diffusers ConfigMixin."""

        config_name: str | None = "config.json"

        def register_to_config(self, **kwargs: Any) -> None:
            kwargs.pop("kwargs", None)
            prev = dict(getattr(self, "_internal_dict", {}))
            prev.update(kwargs)
            self._internal_dict = _FrozenConfigDict(prev)

        @property
        def config(self) -> Mapping[str, Any]:
            return self._internal_dict

    class _BaseCacheMixin:  # type: ignore[no-redef]
        """Pure-PyTorch shim matching diffusers CacheMixin."""

        _cache_config: Any = None

        @property
        def is_cache_enabled(self) -> bool:
            return self._cache_config is not None

        def enable_cache(self, config: Any) -> None:
            self._cache_config = config

        def disable_cache(self) -> None:
            self._cache_config = None

        def _reset_stateful_cache(self, recurse: bool = True) -> None:
            _ = recurse

        @contextmanager
        def cache_context(self, name: str):
            _ = name
            yield


def _torch_dtype_to_qf(dtype: torch.dtype) -> int:
    """Map a floating-point torch.dtype to the engine C-API dtype enum."""
    code = _TORCH_TO_QF_DTYPE.get(dtype)
    if code is None:
        raise ValueError(
            f"Unsupported tensor dtype {dtype!r}; expected float32, float16, or bfloat16"
        )
    return code


def _extract_execution_device(hook: Any, seen: set[int]) -> torch.device | None:
    """Recursively inspect accelerate hooks (including SequentialHook) for execution_device."""
    if hook is None:
        return None
    obj_id = id(hook)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    exec_dev = getattr(hook, "execution_device", None)
    if exec_dev is not None:
        return torch.device(exec_dev)

    sub_hooks = getattr(hook, "hooks", None)
    if isinstance(sub_hooks, Sequence) and not isinstance(sub_hooks, str | bytes):
        for sub in sub_hooks:
            found = _extract_execution_device(sub, seen)
            if found is not None:
                return found

    inner = getattr(hook, "hook", None)
    if inner is not None:
        return _extract_execution_device(inner, seen)
    return None


def _parse_to_args(*args: Any, **kwargs: Any) -> tuple[torch.device | None, torch.dtype | None]:
    """Parse PyTorch Module.to(*args, **kwargs) into (device, dtype)."""
    try:
        parsed = torch._C._nn._parse_to(*args, **kwargs)
        return parsed[0], parsed[1]
    except Exception:
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        if "device" in kwargs and kwargs["device"] is not None:
            device = torch.device(kwargs["device"])
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            dtype = kwargs["dtype"]
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, torch.Tensor):
                device = arg.device
                dtype = arg.dtype
            elif isinstance(arg, torch.device | str | int):
                device = torch.device(arg)
        return device, dtype


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that value is a strictly positive integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_finite_nonneg(value: Any, name: str) -> float:
    """Validate that value is a finite non-negative float (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return fval


class QuantFuncBaseTransformer(_BaseModelMixin, _BaseConfigMixin, _BaseCacheMixin):
    """Base drop-in transformer delegating compute to a QuantFunc PC Engine handle."""

    config_name: str = "config.json"
    model_arch: str = ""
    _is_quantfunc_transformer: bool = True
    _supports_gradient_checkpointing: bool = False
    _no_split_modules: list[str] = []

    def __init__(
        self,
        *,
        handle: QFEngineHandle,
        config: Mapping[str, Any],
        preset: PresetMetadata | None = None,
        weights_path: str | os.PathLike[str] | None = None,
        engine_config: EngineConfig | None = None,
        auth_config: AuthConfig | None = None,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        device_id: int = 0,
        extra_create_config: Mapping[str, Any] | None = None,
        lib: Any | None = None,
    ) -> None:
        super().__init__()
        self.register_to_config(**dict(config))
        eff_dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
        self._qf_device_anchor = nn.Parameter(
            torch.empty(0, dtype=eff_dtype),
            requires_grad=False,
        )

        self._handle: QFEngineHandle = handle
        self._lib: Any = lib if lib is not None else getattr(handle, "lib", None)
        self._preset: PresetMetadata | None = preset
        self._weights_path: str | None = str(weights_path) if weights_path is not None else None
        self._device_id: int = int(device_id)
        self._cuda_target_device: torch.device | None = None
        self._engine_config: EngineConfig = (
            engine_config if engine_config is not None else EngineConfig(device_id=self._device_id)
        )
        self._auth_config: AuthConfig | None = auth_config
        self._extra_create_config: dict[str, Any] = dict(extra_create_config or {})

        self._lora_manager: LoRAManager = LoRAManager(
            self._handle,
            on_before_rebind=self._end_session_if_open,
        )

        self._default_steps: int = int(DEFAULT_AUTO_SESSION_MAX_STEPS)
        self._session_ctx_keepalive: dict[int, torch.Tensor] = {}
        self._session_tracker: AutoSessionTracker = AutoSessionTracker(
            self._handle,
            default_steps=self._default_steps,
            on_session_reset=self._session_ctx_keepalive.clear,
        )
        self._staged_schedule: StagedSchedule | None = None
        self._sol_sigmas: list[float] | None = None
        self._session_saw_scaled_timestep: bool = False
        self._scaled_timestep_session_ref: Any = None
        self._pending_scaled_timestep_for_open: bool = False

        self._cache_config: CacheConfig = CacheConfig()
        self._sol_config: SolConfig | None = None
        self._quality_enhance: bool = False
        self._attention_backend: str = normalize_attention_backend(
            getattr(self._engine_config, "attention_backend", "auto")
        )
        self._producer_stream_mode: str = str(
            getattr(self._engine_config, "producer_stream_mode", "torch_current")
        )
        self._active_cache_context: str | None = None
        self._scratch: dict[str, torch.Tensor] = {}
        self._closed: bool = False

    # ── Construction factories ──────────────────────────────────────────────

    @classmethod
    def _load_preset_metadata(cls, arch: str | None) -> PresetMetadata | None:
        if not arch:
            return None
        try:
            pdir = bundled_preset_dir(arch)
        except Exception:
            return None
        preset_file = Path(pdir) / "preset.json"
        if preset_file.is_file():
            loader = getattr(PresetMetadata, "from_file", None) or getattr(
                PresetMetadata, "load", None
            )
            if callable(loader):
                return loader(preset_file)
        return None

    @classmethod
    def _build_extra_create_config(
        cls,
        *,
        weights_path: str | None,
        config: Mapping[str, Any],
        config_path: str | None,
        preset: PresetMetadata | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = (weights_path, config, config_path, preset, kwargs)
        return {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        config_path: str | os.PathLike[str] | None = None,
        config: Mapping[str, Any] | None = None,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        device_id: int = 0,
        attention_backend: str = "auto",
        offload_mode: str = "auto",
        pinned_memory: bool = False,
        vram_budget_mb: int = 0,
        act_scale_g32: bool | None = None,
        glu_fuse: bool | None = None,
        producer_stream_mode: str = "torch_current",
        api_key: str | None = None,
        server_url: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        source: str = "auto",
        revision: str | None = None,
        subfolder: str | None = None,
        auto_install: bool = True,
        so_path: str | os.PathLike[str] | None = None,
        lib: Any | None = None,
        handle: QFEngineHandle | None = None,
        preset: PresetMetadata | None = None,
        quality_enhance: bool = False,
        block_cache: float = 0.0,
        step_cache: float = 0.0,
        sol_config: SolConfig | None = None,
        loras: Sequence[Any] | None = None,
        default_steps: int | None = None,
        **kwargs: Any,
    ) -> QuantFuncBaseTransformer:
        """Load a QuantFunc transformer from a local file/dir, HuggingFace Hub, or ModelScope."""
        arch = cls.model_arch or str(kwargs.pop("model_arch", "") or "")
        spec = None
        if arch:
            try:
                spec = get_model_spec(arch)
            except KeyError:
                spec = None
        engine_arch = spec.engine_arch if spec is not None else (arch or "auto")
        default_cfg_path = spec.default_config_path if spec is not None else None
        resolved_preset = preset if preset is not None else cls._load_preset_metadata(arch)

        raw_path = str(pretrained_model_name_or_path)
        if handle is not None and not os.path.exists(raw_path):
            resolved_weights = raw_path
        elif os.path.isfile(raw_path):
            resolved_weights = str(Path(raw_path).resolve())
        else:
            file_hints = (
                getattr(resolved_preset, "file_hints", ())
                if resolved_preset is not None
                else (spec.file_hints if spec is not None else ())
            )
            resolved_weights = str(
                resolve_checkpoint(
                    pretrained_model_name_or_path,
                    source=source,
                    revision=revision,
                    subfolder=subfolder,
                    file_hints=file_hints,
                )
            )

        resolved_cfg: dict[str, Any]
        resolved_cfg_path: str | None
        if config is not None:
            resolved_cfg = dict(config)
            resolved_cfg_path = str(config_path) if config_path is not None else None
            if resolved_cfg_path is None and default_cfg_path is not None and Path(default_cfg_path).is_file():
                resolved_cfg_path = str(Path(default_cfg_path).resolve())
            if resolved_cfg_path is None and arch:
                try:
                    candidate = Path(bundled_preset_dir(arch)) / "transformer" / "config.json"
                    if candidate.is_file():
                        resolved_cfg_path = str(candidate)
                except Exception:
                    resolved_cfg_path = None
        else:
            cfg_res = resolve_transformer_config(
                arch=arch or None,
                config_path=config_path,
                weights_path=resolved_weights,
                default_config_path=default_cfg_path,
            )
            if isinstance(cfg_res, tuple) and len(cfg_res) == 2:
                resolved_cfg = dict(cfg_res[0])
                resolved_cfg_path = str(cfg_res[1]) if cfg_res[1] is not None else None
            else:
                resolved_cfg = dict(cfg_res)
                resolved_cfg_path = str(config_path) if config_path is not None else None

        norm_backend = normalize_attention_backend(attention_backend)
        eng_cfg = EngineConfig(
            model_arch=engine_arch,
            attention_backend=norm_backend,
            offload_mode=offload_mode,
            pinned_memory=pinned_memory,
            vram_budget_mb=vram_budget_mb,
            device_id=device_id,
            act_scale_g32=act_scale_g32,
            glu_fuse=glu_fuse,
            producer_stream_mode=producer_stream_mode,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        auth_cfg = resolve_auth(
            api_key=api_key,
            server_url=server_url,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

        extra_create = cls._build_extra_create_config(
            weights_path=resolved_weights,
            config=resolved_cfg,
            config_path=resolved_cfg_path,
            preset=resolved_preset,
            kwargs=kwargs,
        )
        if resolved_cfg_path is not None and "config_path" not in extra_create:
            extra_create["config_path"] = resolved_cfg_path

        if handle is None:
            native_lib = (
                lib
                if lib is not None
                else load_lib(
                    device_id=device_id,
                    so_path=str(so_path) if so_path is not None else None,
                    auto_install=auto_install,
                )
            )
            create_dict = cls._assemble_create_dict(
                arch=engine_arch,
                engine_config=eng_cfg,
                auth_config=auth_cfg,
                extra_create_config=extra_create,
            )
            live_handle = QFEngineHandle.create(
                native_lib,
                weights_path=resolved_weights,
                device_id=device_id,
                config_json=create_dict,
            )
        else:
            live_handle = handle
            native_lib = lib if lib is not None else getattr(handle, "lib", None)

        model = cls(
            handle=live_handle,
            config=resolved_cfg,
            preset=resolved_preset,
            weights_path=resolved_weights,
            engine_config=eng_cfg,
            auth_config=auth_cfg,
            torch_dtype=torch_dtype,
            device_id=device_id,
            extra_create_config=extra_create,
            lib=native_lib,
            **kwargs,
        )
        model.set_attention_backend(norm_backend)
        model.set_quality_enhance(bool(quality_enhance))
        if block_cache or step_cache:
            model.set_cache(block_cache=block_cache, step_cache=step_cache)
        if sol_config is not None:
            model.set_sol(sol_config)
        if default_steps is not None:
            model.set_default_steps(default_steps)
        if loras:
            model.set_loras(loras)
        return model

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str | os.PathLike[str],
        **kwargs: Any,
    ) -> QuantFuncBaseTransformer:
        """Load a QuantFunc transformer from a single safetensors checkpoint file."""
        return cls.from_pretrained(pretrained_model_link_or_path, **kwargs)

    @staticmethod
    def _assemble_create_dict(
        *,
        arch: str,
        engine_config: EngineConfig,
        auth_config: AuthConfig | None,
        extra_create_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        create_dict = dict(engine_config.to_create_dict())
        if arch:
            create_dict["model_arch"] = arch
        if auth_config is not None:
            if getattr(auth_config, "server_url", None):
                create_dict["server_url"] = auth_config.server_url
            if getattr(auth_config, "api_key", None):
                create_dict["api_key"] = auth_config.api_key
            if getattr(auth_config, "cache_dir", None) and "cache_dir" not in create_dict:
                create_dict["cache_dir"] = auth_config.cache_dir
        for key, value in extra_create_config.items():
            if value is not None:
                create_dict[key] = value
        return create_dict

    def _build_create_dict(self) -> dict[str, Any]:
        return self._assemble_create_dict(
            arch=self._engine_config.model_arch or self.model_arch,
            engine_config=self._engine_config,
            auth_config=self._auth_config,
            extra_create_config=self._extra_create_config,
        )

    # ── Device, dtype, offload & meta-safe .to() ───────────────────────────

    @property
    def device(self) -> torch.device:
        """Resolve effective execution device without returning 'meta' under sequential offload."""
        anchor_dev = self._qf_device_anchor.device
        if anchor_dev.type == "meta":
            hf_hook = getattr(self, "_hf_hook", None)
            if hf_hook is not None:
                exec_dev = _extract_execution_device(hf_hook, set())
                if exec_dev is not None:
                    return exec_dev
            if self._cuda_target_device is not None:
                return self._cuda_target_device
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self._device_id}")
            return torch.device("cpu")
        if anchor_dev.type == "cpu" and self._cuda_target_device is not None:
            return self._cuda_target_device
        return anchor_dev

    @property
    def dtype(self) -> torch.dtype:
        """Return the module's active activation dtype from the 0-byte anchor."""
        return self._qf_device_anchor.dtype

    def to(self, *args: Any, **kwargs: Any) -> QuantFuncBaseTransformer:
        """Move/cast transformer while managing engine VRAM, cross-GPU rebuilds, and meta safety."""
        target_dev, target_dtype = _parse_to_args(*args, **kwargs)

        if target_dev is not None and target_dev.type == "cpu":
            self._cuda_target_device = None
            self._end_session_if_open()
            self.unload_vram(QF_UINT64_MAX)
        elif target_dev is not None and target_dev.type == "cuda":
            new_dev_id = int(target_dev.index) if target_dev.index is not None else 0
            if new_dev_id != self._device_id:
                self._recreate_handle_on_device(new_dev_id)
        elif target_dev is not None and target_dev.type == "meta":
            self._session_ctx_keepalive.clear()

        anchor = self._qf_device_anchor
        eff_dtype = target_dtype if target_dtype is not None else anchor.dtype

        if anchor.device.type == "meta":
            eff_dev = target_dev if target_dev is not None else anchor.device
            if eff_dev.type == "cuda" and not torch.cuda.is_available():
                new_idx = int(eff_dev.index) if eff_dev.index is not None else self._device_id
                self._cuda_target_device = torch.device(f"cuda:{new_idx}")
                self._qf_device_anchor = nn.Parameter(
                    torch.empty(0, device="cpu", dtype=eff_dtype),
                    requires_grad=False,
                )
            else:
                if eff_dev.type != "meta":
                    self._cuda_target_device = None
                self._qf_device_anchor = nn.Parameter(
                    torch.empty(0, device=eff_dev, dtype=eff_dtype),
                    requires_grad=False,
                )
            return self

        if target_dev is not None and target_dev.type == "cuda" and not torch.cuda.is_available():
            new_idx = int(target_dev.index) if target_dev.index is not None else self._device_id
            self._cuda_target_device = torch.device(f"cuda:{new_idx}")
            self._qf_device_anchor = nn.Parameter(
                torch.empty(0, device="cpu", dtype=eff_dtype),
                requires_grad=False,
            )
            return self

        if target_dev is not None and target_dev.type == "meta":
            self._cuda_target_device = None
            self._qf_device_anchor = nn.Parameter(
                torch.empty(0, device="meta", dtype=eff_dtype),
                requires_grad=False,
            )
            return self

        if target_dev is not None:
            self._cuda_target_device = None
        return nn.Module.to(self, *args, **kwargs)

    def _recreate_handle_on_device(self, new_device_id: int) -> None:
        """Destroy current engine handle and recreate it on a new CUDA device index."""
        self._end_session_if_open()
        old_lora_mgr = self._lora_manager
        saved_loras = list(old_lora_mgr.active_loras)
        saved_cache_dir = old_lora_mgr.cache_dir
        saved_registry = old_lora_mgr.lora_registry
        saved_roots = old_lora_mgr.allowed_lora_roots
        saved_strict = old_lora_mgr.strict_registry
        if self._lib is None or not self._weights_path:
            raise RuntimeError(
                "Cannot migrate transformer to another GPU without native library and weights_path"
            )
        old_handle = self._handle
        if old_handle is not None:
            old_handle.destroy()

        self._device_id = int(new_device_id)
        self._engine_config = EngineConfig(
            model_arch=self._engine_config.model_arch,
            attention_backend=self._attention_backend,
            offload_mode=self._engine_config.offload_mode,
            pinned_memory=self._engine_config.pinned_memory,
            vram_budget_mb=self._engine_config.vram_budget_mb,
            device_id=self._device_id,
            act_scale_g32=self._engine_config.act_scale_g32,
            glu_fuse=self._engine_config.glu_fuse,
            producer_stream_mode=self._producer_stream_mode,
            cache_dir=self._engine_config.cache_dir,
        )
        create_dict = self._build_create_dict()
        new_handle = QFEngineHandle.create(
            self._lib,
            weights_path=self._weights_path,
            device_id=self._device_id,
            config_json=create_dict,
        )
        self._handle = new_handle
        self._lora_manager = LoRAManager(
            new_handle,
            cache_dir=saved_cache_dir,
            lora_registry=saved_registry,
            allowed_lora_roots=saved_roots,
            strict_registry=saved_strict,
            on_before_rebind=self._end_session_if_open,
        )
        self._session_tracker = AutoSessionTracker(
            new_handle,
            default_steps=self._default_steps,
            on_session_reset=self._session_ctx_keepalive.clear,
        )
        if self._staged_schedule is not None:
            self._session_tracker.set_schedule(self._staged_schedule)
        if saved_loras:
            self.set_loras(saved_loras, force=True)

    # ── CacheMixin integration ──────────────────────────────────────────────

    @property
    def is_cache_enabled(self) -> bool:
        cfg = self._cache_config
        if isinstance(cfg, CacheConfig):
            return bool(cfg.step_cache > 0.0 or cfg.block_cache > 0.0)
        return cfg is not None

    def enable_cache(self, config: Any) -> None:
        if isinstance(config, CacheConfig):
            self.set_cache(block_cache=config.block_cache, step_cache=config.step_cache)
            return
        thresh = getattr(config, "threshold", None)
        if thresh is not None:
            self.set_cache(block_cache=float(thresh))
            return
        raise ValueError(
            f"Unsupported cache config {type(config).__name__}; use set_cache(block_cache=..., step_cache=...)"
        )

    def disable_cache(self) -> None:
        self.set_cache(block_cache=0.0, step_cache=0.0)

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        _ = recurse

    @contextmanager
    def cache_context(self, name: str):
        prev = self._active_cache_context
        self._active_cache_context = str(name)
        try:
            yield
        finally:
            self._active_cache_context = prev

    # ── LoRA management ─────────────────────────────────────────────────────

    def configure_lora_policy(
        self,
        *,
        lora_registry: Any = ...,
        allowed_lora_roots: Any = ...,
        strict_registry: Any = ...,
    ) -> None:
        """Configure default LoRA alias registry, allowed roots, and strict registry mode."""
        self._lora_manager.configure_policy(
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
        )

    def load_lora(
        self,
        path: str | os.PathLike[str],
        scale: float = 1.0,
        *,
        force: bool = False,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool | None = None,
    ) -> None:
        """Hot-load a single LoRA adapter into the live engine handle."""
        self.set_loras(
            [LoRAEntry(path=os.fspath(path), scale=scale)],
            force=force,
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
        )

    def set_loras(
        self,
        loras: Sequence[
            LoRAEntry
            | tuple[str | os.PathLike[str], float]
            | list[Any]
            | Mapping[str, Any]
            | str
            | os.PathLike[str]
        ],
        *,
        force: bool = False,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool | None = None,
    ) -> None:
        """Hot-swap the active LoRA stack without rebuilding the engine handle (O(1) no-op on identical stack)."""
        entries = normalize_lora_entries(loras)
        eff_model_path = (
            self._weights_path
            if (self._weights_path is not None and os.path.isfile(self._weights_path))
            else None
        )
        self._lora_manager.set_loras(
            entries,
            model_path=eff_model_path,
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
            force=force,
        )

    def unload_loras(self, *, force: bool = False) -> None:
        """Clear all active LoRA adapters by sending [] to the engine (O(1) no-op when already empty)."""
        self._lora_manager.unload_loras(force=force)

    def load_lora_adapter(self, *args: Any, **kwargs: Any) -> None:
        """Unsupported diffusers PEFT entrypoint; directs users to native load_lora()."""
        _ = (args, kwargs)
        raise NotImplementedError(
            "QuantFunc transformers load LoRA adapters natively via the engine. "
            "Use transformer.load_lora(path, scale=1.0) or transformer.set_loras([(path, scale), ...]) instead."
        )

    # ── Runtime dials ───────────────────────────────────────────────────────

    @property
    def quality_enhance(self) -> bool:
        return self._quality_enhance

    @quality_enhance.setter
    def quality_enhance(self, enabled: bool) -> None:
        self.set_quality_enhance(enabled)

    @property
    def attention_backend(self) -> str:
        return self._attention_backend

    @attention_backend.setter
    def attention_backend(self, backend: str) -> None:
        self.set_attention_backend(backend)

    @property
    def block_cache(self) -> float:
        return self._cache_config.block_cache

    @block_cache.setter
    def block_cache(self, threshold: float) -> None:
        self.set_cache(block_cache=threshold)

    @property
    def step_cache(self) -> float:
        return self._cache_config.step_cache

    @step_cache.setter
    def step_cache(self, threshold: float) -> None:
        self.set_cache(step_cache=threshold)

    @property
    def sol_config(self) -> SolConfig | None:
        return self._sol_config

    @sol_config.setter
    def sol_config(self, config: SolConfig | None) -> None:
        self.set_sol(config)

    @property
    def active_loras(self) -> list[LoRAEntry]:
        return list(self._lora_manager.active_loras)

    def set_cache(
        self,
        block_cache: float | None = None,
        step_cache: float | None = None,
    ) -> None:
        """Configure engine step-cache and block-cache thresholds."""
        new_block = (
            self._cache_config.block_cache
            if block_cache is None
            else _validate_finite_nonneg(block_cache, "block_cache")
        )
        new_step = (
            self._cache_config.step_cache
            if step_cache is None
            else _validate_finite_nonneg(step_cache, "step_cache")
        )
        self._cache_config = CacheConfig(step_cache=new_step, block_cache=new_block)

    def set_sol(self, sol_config: SolConfig | None) -> None:
        """Configure sparse attention policy for subsequent sessions."""
        if sol_config is not None and not isinstance(sol_config, SolConfig):
            raise TypeError(f"sol_config must be a SolConfig or None, got {type(sol_config).__name__}")
        self._sol_config = sol_config

    def set_attention_backend(self, backend: str) -> None:
        """Set attention backend dial ('auto', 'int8', 'i4xi8', 'int4', 'fp16')."""
        self._attention_backend = normalize_attention_backend(backend)

    def set_quality_enhance(self, enabled: bool) -> None:
        """Toggle quality enhancement mode (True forces full ratio 1.0)."""
        self._quality_enhance = bool(enabled)

    def set_default_steps(self, n: int) -> None:
        """Override Tier-2 AutoSessionTracker fallback step budget."""
        val = _validate_positive_int(n, "default_steps")
        self._default_steps = val
        self._session_tracker.set_default_steps(val)

    def set_staged_schedule(self, schedule: StagedSchedule | None) -> None:
        """Bind or clear a Tier-1 StagedSchedule pushed by pipeline hooks."""
        self._staged_schedule = schedule
        self._reset_timestep_scale_tracking()
        if schedule is not None and schedule.sigmas is not None:
            self._sol_sigmas = [float(s) for s in schedule.sigmas]
        else:
            self._sol_sigmas = None
        if hasattr(self._session_tracker, "set_schedule"):
            self._session_tracker.set_schedule(schedule)
        elif hasattr(self._session_tracker, "stage_schedule"):
            self._session_tracker.stage_schedule(schedule)

    def set_hook_step_cursor(self, idx: int) -> None:
        """Update the external Tier-1 step cursor pushed by pipeline scheduler hooks."""
        self._session_tracker.set_step_cursor(idx)

    def set_sol_sigmas(self, sigmas: Sequence[float] | None) -> None:
        """Update active sigma schedule used for SolConfig window conversion."""
        self._sol_sigmas = [float(s) for s in sigmas] if sigmas is not None else None

    def _effective_quality_ratio(
        self,
        has_refs: bool = False,
        batch: int = 1,
        single_cond_group: bool = True,
    ) -> float:
        """Compute model-appropriate quality ratio for session options."""
        _ = (has_refs, batch, single_cond_group)
        if self._quality_enhance:
            return 1.0
        if self._preset is not None and self._preset.default_quality_ratio is not None:
            return float(self._preset.default_quality_ratio)
        return float(DEFAULT_QUALITY_RATIO)

    def _build_base_session_options(
        self,
        *,
        has_refs: bool = False,
        batch: int = 1,
        single_cond_group: bool = True,
        sigmas: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "attention_backend": self._attention_backend,
            WIRE_QUALITY_RATIO_KEY: float(
                self._effective_quality_ratio(
                    has_refs=has_refs,
                    batch=batch,
                    single_cond_group=single_cond_group,
                )
            ),
        }
        if self._cache_config.step_cache > 0.0:
            opts["step_cache_thresh"] = float(self._cache_config.step_cache)
        if self._cache_config.block_cache > 0.0:
            opts["block_cache_thresh"] = float(self._cache_config.block_cache)
        if self._sol_config is not None and self._sol_config.enabled:
            eff_sigmas = sigmas
            if eff_sigmas is None:
                eff_sigmas = self._sol_sigmas
            if eff_sigmas is None and self._staged_schedule is not None:
                eff_sigmas = self._staged_schedule.sigmas
            sparse = self._sol_config.to_native(sigmas=eff_sigmas)
            if sparse is not None:
                opts["sparse"] = sparse
        return opts

    # ── Session & VRAM lifecycle ────────────────────────────────────────────

    def sampling_session(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        max_context_dims: tuple[int, int, int] = (0, 0, 0),
        cond_dtype: torch.dtype | int | None = None,
        options: Mapping[str, Any] | None = None,
        sigmas: Sequence[float] | None = None,
        do_cfg: bool = False,
        has_refs: bool = False,
        batch: int = 1,
    ) -> SamplingSession:
        """Create an explicit Tier-3 SamplingSession context manager."""
        qf_cond_dtype = (
            _torch_dtype_to_qf(self.dtype)
            if cond_dtype is None
            else (
                _torch_dtype_to_qf(cond_dtype)
                if isinstance(cond_dtype, torch.dtype)
                else int(cond_dtype)
            )
        )
        merged_opts = self._build_base_session_options(
            has_refs=has_refs,
            batch=batch,
            single_cond_group=not do_cfg and batch == 1,
            sigmas=sigmas,
        )
        if options:
            merged_opts.update(dict(options))
        return SamplingSession(
            handle=self._handle,
            tracker=self._session_tracker,
            width=int(width),
            height=int(height),
            num_steps=int(num_steps),
            max_context_dims=tuple(int(d) for d in max_context_dims),
            cond_dtype=qf_cond_dtype,
            options=merged_opts,
            sigmas=sigmas,
            do_cfg=do_cfg,
            on_session_end=self._session_ctx_keepalive.clear,
        )

    def end_session_if_open(self) -> tuple[bool, bool]:
        """Close any active engine session and clear scratch buffers."""
        return self._end_session_if_open()

    def reset_request_state(self) -> None:
        """Clear per-request transient session/schedule/scratch state without unloading weights or LoRAs."""
        if not self._closed:
            self._end_session_if_open()
        else:
            self._release_scratch()
            self._reset_timestep_scale_tracking()
        self.set_staged_schedule(None)
        self._sol_sigmas = None
        self._active_cache_context = None
        self._release_scratch()

    def _reset_timestep_scale_tracking(self) -> None:
        self._session_saw_scaled_timestep = False
        self._scaled_timestep_session_ref = None
        self._pending_scaled_timestep_for_open = False

    def _end_session_if_open(self) -> tuple[bool, bool]:
        self._release_scratch()
        self._session_ctx_keepalive.clear()
        self._reset_timestep_scale_tracking()
        if self._session_tracker is not None and hasattr(self._session_tracker, "end_session_if_open"):
            return self._session_tracker.end_session_if_open()
        if self._handle is not None:
            return self._handle.end_session_if_open()
        return (False, True)

    def unload_vram(self, bytes_requested: int = QF_UINT64_MAX) -> int:
        """Unload up to bytes_requested of engine GPU memory."""
        self._release_scratch()
        if self._handle is None:
            return 0
        return int(self._handle.unload_vram(bytes_requested))

    def resident_vram_bytes(self) -> int:
        """Query current engine resident VRAM bytes."""
        if self._handle is None:
            return 0
        return int(self._handle.resident_vram_bytes())

    def estimate_vram(self) -> ColdNeed:
        """Run header-only VRAM estimation via qf_xfm_estimate_vram."""
        if self._lib is None or not self._weights_path:
            raise RuntimeError("estimate_vram requires a loaded native library and weights_path")
        params = make_create_params(
            weights_path=self._weights_path,
            device_id=self._device_id,
            config_json=self._build_create_dict(),
        )
        return estimate_vram(self._lib, params)

    def close(self) -> None:
        """Close any active session, free scratch buffers, and destroy the engine handle."""
        if self._closed:
            return
        self._release_scratch()
        try:
            self._end_session_if_open()
        finally:
            if self._handle is not None:
                self._handle.destroy()
            self._closed = True

    def __del__(self) -> None:
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass

    # ── Shared internal helpers for L2 subclasses ───────────────────────────

    def _ensure_scratch(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buf = self._scratch.get(key)
        target_shape = tuple(int(d) for d in shape)
        if buf is None or tuple(buf.shape) != target_shape or buf.dtype != dtype or buf.device != device:
            buf = torch.empty(target_shape, dtype=dtype, device=device)
            self._scratch[key] = buf
        return buf

    def _release_scratch(self) -> None:
        self._scratch.clear()
        self._session_ctx_keepalive.clear()

    def _resolve_producer_stream(self, tensor: torch.Tensor) -> int:
        if self._producer_stream_mode == "torch_current" and tensor.is_cuda:
            raw = int(torch.cuda.current_stream(tensor.device).cuda_stream)
            return raw if raw != 0 else _CUDA_STREAM_LEGACY
        return 0

    def _producer_stream_ptr(self, tensor: torch.Tensor) -> int:
        return self._resolve_producer_stream(tensor)

    def _spatial_scale(self) -> int:
        if self._preset is not None:
            sc = getattr(self._preset, "spatial_compression", None)
            if sc is not None and int(sc) > 0:
                return int(sc)
            vsf = getattr(self._preset, "vae_scale_factors", None)
            if vsf:
                return int(vsf[-1])
        vsf_cfg = self.config.get("vae_scale_factors")
        if vsf_cfg:
            return int(vsf_cfg[-1])
        sc_cfg = self.config.get("spatial_compression")
        if sc_cfg is not None and int(sc_cfg) > 0:
            return int(sc_cfg)
        raise ValueError(
            f"{type(self).__name__}: missing spatial_compression / vae_scale_factors in preset and config"
        )

    def _temporal_scale(self) -> int:
        if self._preset is not None:
            tc = getattr(self._preset, "temporal_compression", None)
            if tc is not None and int(tc) > 0:
                return int(tc)
            vsf = getattr(self._preset, "vae_scale_factors", None)
            if vsf and len(vsf) >= 3:
                return int(vsf[0])
        vsf_cfg = self.config.get("vae_scale_factors")
        if vsf_cfg and len(vsf_cfg) >= 3:
            return int(vsf_cfg[0])
        tc_cfg = self.config.get("temporal_compression")
        if tc_cfg is not None and int(tc_cfg) > 0:
            return int(tc_cfg)
        raise ValueError(
            f"{type(self).__name__}: missing temporal_compression / vae_scale_factors in preset and config"
        )

    def _timestep_scale_multiplier(self) -> float:
        raw = self.config.get("timestep_scale_multiplier")
        if raw is None:
            return float(DEFAULT_TIMESTEP_SCALE)
        val = float(raw)
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"timestep_scale_multiplier must be positive and finite, got {raw!r}")
        return val

    def _normalize_scalar_sigma(self, value: float) -> float:
        fval = float(value)
        if not math.isfinite(fval) or fval < 0.0:
            raise ValueError(f"Non-finite or negative sigma/timestep value: {value!r}")
        cur_sess = getattr(self._handle, "current_session", None)
        if self._scaled_timestep_session_ref is not None and cur_sess is not self._scaled_timestep_session_ref:
            self._reset_timestep_scale_tracking()
        scale = self._timestep_scale_multiplier()
        sched_scale = self._staged_schedule.timestep_scale if self._staged_schedule is not None else None
        eff_scale = sched_scale if (sched_scale is not None and sched_scale > 1.0) else scale
        if fval > 1.0:
            self._session_saw_scaled_timestep = True
            self._pending_scaled_timestep_for_open = True
            if cur_sess is not None:
                self._scaled_timestep_session_ref = cur_sess
            return fval / eff_scale
        if fval > 0.0 and ((sched_scale is not None and sched_scale > 1.0) or self._session_saw_scaled_timestep):
            return fval / eff_scale
        return fval

    def _extract_batch_sigmas(self, timestep: Any, batch: int) -> list[float]:
        if torch.is_tensor(timestep):
            flat = timestep.detach().float().reshape(-1)
            if flat.numel() == 0:
                raise ValueError("timestep tensor must be non-empty")
            vals = flat.cpu().tolist()
            if len(vals) == 1:
                v = self._normalize_scalar_sigma(float(vals[0]))
                return [v] * batch
            if len(vals) >= batch:
                return [self._normalize_scalar_sigma(float(vals[i])) for i in range(batch)]
            raise ValueError(f"timestep length {len(vals)} does not match batch size {batch}")
        v = self._normalize_scalar_sigma(float(timestep))
        return [v] * batch

    def _context_fingerprint(
        self,
        ctx_item: torch.Tensor | Sequence[torch.Tensor | None],
        batch_idx: int,
    ) -> Any:
        """Build a stable conditioning-group identity token for CtxKeyAssigner with 0-sync fast path.

        Contract:
        - When ``cache_context(name)`` is active, returns ``(name, batch_idx)``.
        - Normal path (up to 16 distinct caller tensor ``data_ptr()`` values per session):
          retains references in ``_session_ctx_keepalive`` (cleared on session end / scratch release)
          to prevent allocator address reuse (ABA) and fingerprints via
          ``(data_ptr, _version, shape, stride, dtype, device)`` with zero GPU->CPU ``.item()`` syncs.
          Standard PyTorch in-place ops (``mul_``, ``copy_``, ``zero_``, etc.) automatically bump
          ``_version`` and rotate the key; mutations bypassing ``_version`` (such as
          ``tensor.data.copy_()``) must use an explicit ``cache_context(...)``.
          Inference tensors created under ``torch.inference_mode()`` have no ``_version`` counter
          (accessing ``t._version`` raises ``RuntimeError``) and map to ``-1`` via
          ``t.is_inference()`` (0-sync C++ ``TensorImpl`` flag check). They cannot be mutated
          in-place outside ``inference_mode``; any in-place mutation inside ``inference_mode``
          bypasses ``_version`` and must use an explicit ``cache_context(...)`` (consistent with
          the ``tensor.data.copy_()`` contract).
        - Overflow fallback (>16 distinct ``data_ptr()`` in a single session): never evicts existing
          keepalive entries (preventing ABA collisions) and falls back for the overflow call to a
          lightweight 3-scalar content fingerprint ``(flat[0], flat[mid], flat[-1])``.
        """
        if self._active_cache_context is not None:
            return (self._active_cache_context, batch_idx)

        raw_tensors: tuple[torch.Tensor, ...]
        if torch.is_tensor(ctx_item):
            raw_tensors = (ctx_item,)
        else:
            raw_tensors = tuple(t for t in ctx_item if t is not None)

        new_ptrs: dict[int, torch.Tensor] = {}
        for t in raw_tensors:
            ptr = int(t.data_ptr())
            if ptr not in self._session_ctx_keepalive and ptr not in new_ptrs:
                new_ptrs[ptr] = t

        if len(self._session_ctx_keepalive) + len(new_ptrs) <= _MAX_SESSION_CTX_KEEPALIVE:
            if new_ptrs:
                self._session_ctx_keepalive.update(new_ptrs)
            return (
                "ptr",
                tuple(
                    (
                        int(t.data_ptr()),
                        -1 if t.is_inference() else int(getattr(t, "_version", 0)),
                        tuple(t.shape),
                        tuple(t.stride()),
                        t.dtype,
                        t.device,
                    )
                    for t in raw_tensors
                ),
                batch_idx,
            )

        val_sig: list[tuple[Any, ...]] = []
        for t in raw_tensors:
            flat = t.detach().reshape(-1)
            n = flat.numel()
            if n == 0:
                sample = (0.0, 0.0, 0.0)
            else:
                mid = n // 2
                sample = (
                    float(flat[0].item()),
                    float(flat[mid].item()),
                    float(flat[-1].item()),
                )
            val_sig.append((tuple(t.shape), t.dtype, sample))
        return ("val", tuple(val_sig), batch_idx)

    def _max_context_dims_for(self, ctx_item: torch.Tensor) -> tuple[int, int, int]:
        seq = max(int(ctx_item.shape[1]), int(DEFAULT_MAX_CONTEXT_SEQ))
        return (1, seq, int(ctx_item.shape[2]))

    def _session_is_open(self) -> bool:
        return getattr(self._handle, "current_session", None) is not None

    def _abort_session_on_error(self) -> None:
        try:
            if self._handle is not None:
                self._handle.end_session_if_open()
        finally:
            try:
                if hasattr(self._session_tracker, "on_session_closed"):
                    self._session_tracker.on_session_closed()
            finally:
                self._session_ctx_keepalive.clear()
                self._reset_timestep_scale_tracking()

    @contextmanager
    def _step_error_cleanup(self) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self._abort_session_on_error()
            raise

    def _ensure_session_step(
        self,
        *,
        width: int,
        height: int,
        sigma: float,
        max_context_dims: tuple[int, int, int],
        cond_dtype: int,
        options: dict[str, Any],
        geometry_key: Any,
        ctx_identity: Any = None,
        ctx_identity_factory: Callable[[], Any] | None = None,
        producer_stream: int = 0,
        allow_busy_retry: bool = True,
        before_begin: Callable[[], None] | None = None,
        after_begin: Callable[[], None] | None = None,
    ) -> tuple[int, int, int]:
        """Ensure an active engine session via AutoSessionTracker and return (step_idx, total_steps, ctx_key)."""

        def _wrapped_after_begin() -> None:
            self._session_saw_scaled_timestep = self._pending_scaled_timestep_for_open
            self._scaled_timestep_session_ref = getattr(self._handle, "current_session", None)
            self._pending_scaled_timestep_for_open = False
            if after_begin is not None:
                after_begin()

        with self._step_error_cleanup():
            return self._session_tracker.prepare_step(
                width=int(width),
                height=int(height),
                sigma=float(sigma),
                max_context_dims=max_context_dims,
                cond_dtype=int(cond_dtype),
                options=options,
                geometry_key=geometry_key,
                ctx_identity=ctx_identity,
                ctx_identity_factory=ctx_identity_factory,
                producer_stream=int(producer_stream),
                allow_busy_retry=allow_busy_retry,
                before_begin=before_begin,
                after_begin=_wrapped_after_begin,
            )

    def _call_step_guarded(self, step_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._step_error_cleanup():
            return step_fn(*args, **kwargs)

    @staticmethod
    def _build_step_params(
        *,
        latent_in: torch.Tensor,
        velocity_out: torch.Tensor,
        sigma: float,
        step_index: int,
        total_steps: int,
        context: torch.Tensor | None,
        cfg_context_key: int,
        producer_stream: int = 0,
    ) -> QFStepParams:
        p = QFStepParams()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.struct_size = ctypes.sizeof(QFStepParams)
        p.latent_in = int(latent_in.data_ptr())
        p.velocity_out = int(velocity_out.data_ptr())
        p.velocity_out_capacity = int(velocity_out.numel() * velocity_out.element_size())
        for idx, dim in enumerate(latent_in.shape):
            p.dims[idx] = int(dim)
        p.dtype = _torch_dtype_to_qf(latent_in.dtype)
        p.sigma = float(sigma)
        p.step_index = int(step_index)
        p.total_steps = int(total_steps)
        if context is not None:
            p.context = int(context.data_ptr())
            p.context_dims = (ctypes.c_int32 * 3)(*(int(d) for d in context.shape))
            p.context_dtype = _torch_dtype_to_qf(context.dtype)
        p.cfg_context_key = int(cfg_context_key)
        p.producer_stream = int(producer_stream)
        return p

    def _run_step(
        self,
        *,
        latent_in: torch.Tensor,
        velocity_out: torch.Tensor,
        sigma: float,
        step_index: int,
        total_steps: int,
        context: torch.Tensor | None,
        cfg_context_key: int,
        producer_stream: int = 0,
    ) -> None:
        with self._step_error_cleanup():
            params = self._build_step_params(
                latent_in=latent_in,
                velocity_out=velocity_out,
                sigma=sigma,
                step_index=step_index,
                total_steps=total_steps,
                context=context,
                cfg_context_key=cfg_context_key,
                producer_stream=producer_stream,
            )
            self._call_step_guarded(self._handle.step, params)

    def _run_step_av(
        self,
        *,
        latent_in: torch.Tensor,
        velocity_out: torch.Tensor,
        audio_latent_in: torch.Tensor,
        audio_velocity_out: torch.Tensor,
        sigma: float,
        step_index: int,
        total_steps: int,
        context: torch.Tensor | None,
        cfg_context_key: int,
        audio_scale: float = 1.0,
        audio_reference_active: int = 0,
        producer_stream: int = 0,
    ) -> None:
        with self._step_error_cleanup():
            base = self._build_step_params(
                latent_in=latent_in,
                velocity_out=velocity_out,
                sigma=sigma,
                step_index=step_index,
                total_steps=total_steps,
                context=context,
                cfg_context_key=cfg_context_key,
                producer_stream=producer_stream,
            )
            av = QFStepAvParams()
            ctypes.memset(ctypes.byref(av), 0, ctypes.sizeof(av))
            av.struct_size = ctypes.sizeof(QFStepAvParams)
            av.base = base
            av.audio_latent_in = int(audio_latent_in.data_ptr())
            av.audio_velocity_out = int(audio_velocity_out.data_ptr())
            av.audio_velocity_out_capacity = int(
                audio_velocity_out.numel() * audio_velocity_out.element_size()
            )
            for idx, dim in enumerate(audio_latent_in.shape):
                av.audio_dims[idx] = int(dim)
            av.audio_dtype = _torch_dtype_to_qf(audio_latent_in.dtype)
            av.audio_scale = float(audio_scale)
            av.audio_reference_active = int(audio_reference_active)
            self._call_step_guarded(self._handle.step_av, av)

    def _run_step_refs(
        self,
        *,
        latent_in: torch.Tensor,
        velocity_out: torch.Tensor,
        sigma: float,
        step_index: int,
        total_steps: int,
        context: torch.Tensor | None,
        ref_latents: Sequence[torch.Tensor],
        image_slots: Sequence[int],
        cfg_context_key: int,
        producer_stream: int = 0,
    ) -> None:
        with self._step_error_cleanup():
            base = self._build_step_params(
                latent_in=latent_in,
                velocity_out=velocity_out,
                sigma=sigma,
                step_index=step_index,
                total_steps=total_steps,
                context=context,
                cfg_context_key=cfg_context_key,
                producer_stream=producer_stream,
            )
            n_refs = len(ref_latents)
            ptrs = (ctypes.c_void_p * n_refs)(*(int(t.data_ptr()) for t in ref_latents))
            flat_dims = [int(d) for t in ref_latents for d in t.shape]
            dims_arr = (ctypes.c_int32 * len(flat_dims))(*flat_dims)
            slots_arr = (ctypes.c_int32 * n_refs)(*(int(s) for s in image_slots))

            rp = QFStepRefsParams()
            ctypes.memset(ctypes.byref(rp), 0, ctypes.sizeof(rp))
            rp.struct_size = ctypes.sizeof(QFStepRefsParams)
            rp.base = base
            rp.num_refs = int(n_refs)
            rp.ref_latents = ctypes.cast(ptrs, ctypes.POINTER(ctypes.c_void_p))
            rp.ref_dims = ctypes.cast(dims_arr, ctypes.POINTER(ctypes.c_int32))
            rp.image_slots = ctypes.cast(slots_arr, ctypes.POINTER(ctypes.c_int32))
            # Keep backing ctypes buffers alive for the duration of the native call.
            rp._keep = (ptrs, dims_arr, slots_arr, ref_latents)
            self._call_step_guarded(self._handle.step_refs, rp)
