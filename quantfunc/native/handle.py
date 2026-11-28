"""RAII native transformer handle, strict C-API JSON allowlists, and VRAM/session bindings."""

from __future__ import annotations

import atexit
import ctypes
import json
import math
import operator
import os
import sys
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .capi import (
    QF_ERR_INTERNAL,
    QF_ERR_INVALID_ARG,
    QF_OK,
    QF_UINT64_MAX,
    ColdNeed,
    QFSessionBeginParams,
    QFStepAvParams,
    QFStepParams,
    QFStepRefsParams,
    QFVramEstimate,
    QFXfmCreateParams,
)
from .errors import InvalidArgError, raise_for_status, status_to_exception
from .loader import last_err

MAX_CREATE_JSON_BYTES = 256 * 1024
MAX_SESSION_JSON_BYTES = 1024 * 1024
MAX_API_JSON_DEPTH = 16

_WIRE_QUALITY_RATIO_KEY = "cat_keep_ratio"

# Strict 12-key create config_json allowlist
CREATE_CONFIG_ALLOWLIST: frozenset[str] = frozenset(
    {
        "model_arch",
        "config_path",
        "connectors_path",
        "attention_backend",
        "offload_mode",
        "pinned_memory",
        "vram_budget_mb",
        "act_scale_g32",
        "glu_fuse",
        "server_url",
        "api_key",
        "cache_dir",
    }
)

# 25 runtime/retired keys forbidden anywhere inside create config_json (including nested structures)
FORBIDDEN_CREATE_KEYS: frozenset[str] = frozenset(
    {
        "cache_mode",
        "cache_thresh",
        "step_cache",
        "block_cache",
        "step_cache_thresh",
        "block_cache_thresh",
        _WIRE_QUALITY_RATIO_KEY,
        "cat_external_signal",
        "sparse",
        "sparse_cdf",
        "sol",
        "sol_s2",
        "s2_cubin_path",
        "quality",
        "video_enhance",
        "audio_enhance",
        "kernel_plugin_path",
        "denoise_only",
        "transformer_precision",
        "lora",
        "controlnet",
        "tiled_vae",
        "text_precision",
        "cpu_offload",
        "layer_offload",
    }
)

# Strict 19-key session options_json allowlist
SESSION_OPTIONS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "attention_backend",
        "step_cache_thresh",
        "block_cache_thresh",
        _WIRE_QUALITY_RATIO_KEY,
        "cat_external_signal",
        "sparse",
        "num_frames",
        "fps",
        "audio_dims",
        "av_sigma_shift_video",
        "av_sigma_shift_audio",
        "av_unprocessed_ctx",
        "av_conds",
        "text_vision_row_ranges",
        "frame_t_scales",
        "audio_frame_t_scales",
        "stg_blocks",
        "disable_av_cross_attn",
        "idle_timeout_ms",
    }
)

# 11 legacy session option keys explicitly rejected before hitting C-API
REJECTED_LEGACY_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "audio_enhance",
        "extra_audio_steps",
        "sol_s2",
        "s2_cubin_path",
        "controlnet",
        "tiled_vae",
        "text_precision",
        "cpu_offload",
        "layer_offload",
        "sol",
        "sparse_cdf",
    }
)

# 14 allowed subkeys inside session options_json["sparse"]
SPARSE_SUBKEY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "enabled",
        "selector",
        "tau",
        "top_p",
        "cdf",
        "min_tokens",
        "first_layers_dense",
        "first_steps_dense",
        "first_steps_dense_frac",
        "plan_interval",
        "auto_dense_guard",
        "require_permute",
        "window",
        "sink_conditioning",
    }
)

# Pending reference-audio owners survive handle GC until explicit native cleanup completes.
# Strict acyclic lock invariant: no path holding _destroy_lock ever acquires _LIVE_HANDLES_LOCK;
# both are RLock, and registry discards while any _destroy_lock is held defer via _THREAD_DESTROY_STATE.
_LIVE_HANDLES_LOCK = threading.RLock()
_LIVE_HANDLES: weakref.WeakSet[QFEngineHandle] = weakref.WeakSet()
_REFERENCE_AUDIO_PENDING_OWNERS: set[QFEngineHandle] = set()
_REGISTRY_SNAPSHOT_DEPTH = 0
_DEFERRED_REG_DISCARDS: list[tuple[QFEngineHandle, bool, bool]] = []
_THREAD_DESTROY_STATE = threading.local()


def _thread_destroy_depth() -> int:
    return int(getattr(_THREAD_DESTROY_STATE, "depth", 0))


def _enter_destroy_scope() -> None:
    _THREAD_DESTROY_STATE.depth = _thread_destroy_depth() + 1


def _exit_destroy_scope() -> list[tuple[QFEngineHandle, bool, bool]]:
    cur = _thread_destroy_depth()
    new_depth = cur - 1 if cur > 1 else 0
    _THREAD_DESTROY_STATE.depth = new_depth
    if new_depth > 0:
        return []
    pending: list[tuple[QFEngineHandle, bool, bool]] | None = getattr(
        _THREAD_DESTROY_STATE, "pending_discards", None
    )
    if not pending:
        return []
    _THREAD_DESTROY_STATE.pending_discards = None
    return pending


def _apply_registry_discards_locked(items: Sequence[tuple[QFEngineHandle, bool, bool]]) -> None:
    for item, discard_live, discard_pending in items:
        if discard_live:
            _LIVE_HANDLES.discard(item)
        if discard_pending and not item.reference_audio_pending():
            _REFERENCE_AUDIO_PENDING_OWNERS.discard(item)


def _flush_registry_discards(items: Sequence[tuple[QFEngineHandle, bool, bool]]) -> None:
    with _LIVE_HANDLES_LOCK:
        if _REGISTRY_SNAPSHOT_DEPTH > 0:
            _DEFERRED_REG_DISCARDS.extend(items)
            return
        _apply_registry_discards_locked(items)


def _queue_or_flush_discards(items: Sequence[tuple[QFEngineHandle, bool, bool]]) -> None:
    if _thread_destroy_depth() > 0:
        pending: list[tuple[QFEngineHandle, bool, bool]] | None = getattr(
            _THREAD_DESTROY_STATE, "pending_discards", None
        )
        if pending is None:
            pending = []
            _THREAD_DESTROY_STATE.pending_discards = pending
        pending.extend(items)
        return
    _flush_registry_discards(items)


def _enc(value: str | bytes | os.PathLike[str]) -> bytes:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    return os.fspath(value).encode("utf-8")


def _default_sdk_cache_dir() -> str:
    env = os.environ.get("QF_CACHE_DIR", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.abspath(os.path.join(base, "quantfunc"))


def _check_json_depth_and_finiteness(obj: Any, *, max_depth: int, label: str, depth: int = 0) -> None:
    if isinstance(obj, dict):
        next_depth = depth + 1
        if next_depth > max_depth:
            raise InvalidArgError(f"{label} nesting depth exceeds limit ({max_depth})")
        for k, v in obj.items():
            if not isinstance(k, str):
                raise InvalidArgError(f"{label} object keys must be strings")
            _check_json_depth_and_finiteness(v, max_depth=max_depth, label=label, depth=next_depth)
    elif isinstance(obj, list):
        next_depth = depth + 1
        if next_depth > max_depth:
            raise InvalidArgError(f"{label} nesting depth exceeds limit ({max_depth})")
        for item in obj:
            _check_json_depth_and_finiteness(item, max_depth=max_depth, label=label, depth=next_depth)
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise InvalidArgError(f"{label} contains non-finite floating-point value")


def _find_forbidden_create_key(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_CREATE_KEYS:
                return k
            nested = _find_forbidden_create_key(v)
            if nested is not None:
                return nested
    elif isinstance(obj, list):
        for item in obj:
            nested = _find_forbidden_create_key(item)
            if nested is not None:
                return nested
    return None


def _parse_json_payload(
    payload: Mapping[str, Any] | str | bytes | bytearray,
    *,
    max_bytes: int,
    max_depth: int,
    label: str,
) -> dict[str, Any]:
    if isinstance(payload, str | bytes | bytearray):
        raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        if len(raw) > max_bytes:
            raise InvalidArgError(f"{label} byte size ({len(raw)}) exceeds limit ({max_bytes})")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidArgError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    elif isinstance(payload, Mapping):
        parsed = dict(payload)
    else:
        raise InvalidArgError(f"{label} must be a mapping, JSON string, or bytes")
    if not isinstance(parsed, dict):
        raise InvalidArgError(f"{label} root must be a JSON object")
    _check_json_depth_and_finiteness(parsed, max_depth=max_depth, label=label)
    return parsed


def validate_create_config(
    config_json: Mapping[str, Any] | str | bytes | bytearray | None,
) -> bytes | None:
    """Validate create ``config_json`` against the 12-key allowlist, 25 forbidden keys, size, and depth bounds."""
    if config_json is None:
        return None
    if isinstance(config_json, str | bytes | bytearray) and not config_json:
        return None
    parsed = _parse_json_payload(
        config_json,
        max_bytes=MAX_CREATE_JSON_BYTES,
        max_depth=MAX_API_JSON_DEPTH,
        label="create config_json",
    )
    bad_nested = _find_forbidden_create_key(parsed)
    if bad_nested is not None:
        raise InvalidArgError(
            f"Forbidden runtime or retired key {bad_nested!r} must never appear in create config_json"
        )
    disallowed = sorted(k for k in parsed if k not in CREATE_CONFIG_ALLOWLIST)
    if disallowed:
        raise InvalidArgError(f"Disallowed or unknown create config_json key(s): {', '.join(disallowed)}")
    try:
        encoded = json.dumps(parsed, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidArgError(f"create config_json is not JSON-serializable: {exc}") from exc
    if len(encoded) > MAX_CREATE_JSON_BYTES:
        raise InvalidArgError(
            f"create config_json byte size ({len(encoded)}) exceeds limit ({MAX_CREATE_JSON_BYTES})"
        )
    return encoded


def validate_session_options(
    options_json: Mapping[str, Any] | str | bytes | bytearray | None,
) -> bytes | None:
    """Validate session ``options_json`` against the 19-key allowlist, 11 legacy rejections, size, and depth bounds."""
    if options_json is None:
        return None
    if isinstance(options_json, str | bytes | bytearray) and not options_json:
        return None
    parsed = _parse_json_payload(
        options_json,
        max_bytes=MAX_SESSION_JSON_BYTES,
        max_depth=MAX_API_JSON_DEPTH,
        label="session options_json",
    )
    for k in parsed:
        if k in REJECTED_LEGACY_OPTION_KEYS:
            raise InvalidArgError(f"Removed legacy session option key is not supported: {k!r}")
        if k not in SESSION_OPTIONS_ALLOWLIST:
            raise InvalidArgError(f"Unknown or disallowed session options_json key: {k!r}")
    sparse = parsed.get("sparse")
    if sparse is not None:
        if not isinstance(sparse, dict):
            raise InvalidArgError("session options_json.sparse must be an object or null")
        bad_sparse = sorted(k for k in sparse if k not in SPARSE_SUBKEY_ALLOWLIST)
        if bad_sparse:
            raise InvalidArgError(f"Unknown session options_json.sparse key(s): {', '.join(bad_sparse)}")
    try:
        encoded = json.dumps(parsed, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidArgError(f"session options_json is not JSON-serializable: {exc}") from exc
    if len(encoded) > MAX_SESSION_JSON_BYTES:
        raise InvalidArgError(
            f"session options_json byte size ({len(encoded)}) exceeds limit ({MAX_SESSION_JSON_BYTES})"
        )
    return encoded


def make_create_params(
    *,
    weights_path: str | os.PathLike[str] | None = None,
    model_path: str | os.PathLike[str] | None = None,
    device_id: int = 0,
    config_json: Mapping[str, Any] | str | bytes | bytearray | None = None,
    config: Mapping[str, Any] | str | bytes | bytearray | None = None,
) -> QFXfmCreateParams:
    """Build a validated ``QFXfmCreateParams`` without loading a native library or mutating caller dicts."""
    if isinstance(device_id, bool):
        raise ValueError("device_id must be a nonnegative int32, not bool")
    dev = operator.index(device_id)
    if not 0 <= dev < (1 << 31):
        raise ValueError("device_id must fit nonnegative int32")
    eff_weights = weights_path if weights_path is not None else model_path
    if eff_weights is None:
        raise ValueError("weights_path is required")
    wpath_bytes = _enc(eff_weights)
    if not wpath_bytes:
        raise ValueError("weights_path must be non-empty")

    eff_cfg = config_json if config_json is not None else config
    # Validate caller payload first before injecting default cache_dir so size/depth/allowlist errors trigger cleanly.
    validated = validate_create_config(eff_cfg)
    cfg_dict: dict[str, Any] = json.loads(validated.decode("utf-8")) if validated is not None else {}
    if "cache_dir" not in cfg_dict:
        cfg_dict["cache_dir"] = _default_sdk_cache_dir()
    final_cfg = validate_create_config(cfg_dict)

    params = QFXfmCreateParams(struct_size=ctypes.sizeof(QFXfmCreateParams), device_id=dev)
    keep = [wpath_bytes]
    params.weights_path = keep[0]
    if final_cfg is not None:
        keep.append(final_cfg)
        params.config_json = keep[-1]
    params._keep = keep
    return params


def estimate_vram(lib: Any, create_params: QFXfmCreateParams) -> ColdNeed:
    """Header-only cold VRAM estimate via ``qf_xfm_estimate_vram`` (zero GPU allocation)."""
    if not isinstance(create_params, QFXfmCreateParams):
        raise TypeError("estimate_vram requires QFXfmCreateParams")
    fn = getattr(lib, "qf_xfm_estimate_vram", None)
    if fn is None:
        return ColdNeed(None, "qf_xfm_estimate_vram symbol unavailable")
    est = QFVramEstimate(struct_size=ctypes.sizeof(QFVramEstimate))
    st = fn(ctypes.byref(create_params), ctypes.byref(est))
    if st == QF_OK and (est.resident_weight_bytes or est.working_set_floor_bytes):
        return ColdNeed(
            int(est.resident_weight_bytes),
            None,
            int(est.min_offload_bytes),
            int(est.working_set_floor_bytes),
            int(est.recommended_offload_mode),
        )
    if st == QF_ERR_INVALID_ARG:
        exc = status_to_exception(st, context="qf_xfm_estimate_vram", detail=last_err(lib))
        assert exc is not None
        raise exc
    return ColdNeed(None, f"status {st}: {last_err(lib)}")


def create_transformer(
    lib: Any,
    *,
    create_params: QFXfmCreateParams | None = None,
    **create_kwargs: Any,
) -> ctypes.c_void_p:
    """Create a native ``qf_xfm_t`` handle via ``qf_xfm_create``."""
    if create_params is not None and create_kwargs:
        raise ValueError("Pass either create_params or create keywords, not both")
    params = make_create_params(**create_kwargs) if create_params is None else create_params
    if not isinstance(params, QFXfmCreateParams):
        raise TypeError("create_transformer requires QFXfmCreateParams")
    if params.config_json:
        validate_create_config(params.config_json)
    handle = ctypes.c_void_p()
    st = lib.qf_xfm_create(ctypes.byref(params), ctypes.byref(handle))
    if st != QF_OK or not handle:
        eff_status = st if st != QF_OK else QF_ERR_INTERNAL
        raise_for_status(eff_status, context="qf_xfm_create", detail=last_err(lib))
    return handle


def _normalize_lora_payload(loras: Sequence[Any] | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in loras or ():
        if isinstance(item, Mapping):
            path = os.fspath(item["path"])
            scale = float(item.get("scale", 1.0))
        elif hasattr(item, "path") and hasattr(item, "scale"):
            path = os.fspath(item.path)
            scale = float(item.scale)
        elif isinstance(item, tuple | list) and len(item) == 2:
            path = os.fspath(item[0])
            scale = float(item[1])
        else:
            raise TypeError(f"Unsupported LoRA entry specification: {item!r}")
        if not math.isfinite(scale):
            raise ValueError(f"LoRA scale must be finite, got {scale!r}")
        payload.append({"path": path, "scale": scale})
    return payload


class QFEngineHandle:
    """RAII owner of a native ``qf_xfm_t`` handle and its single active session."""

    @classmethod
    def create(
        cls,
        lib: Any,
        weights_path: str | os.PathLike[str] | None = None,
        create_cfg: Mapping[str, Any] | str | bytes | None = None,
        device_id: int = 0,
        *,
        config_json: Mapping[str, Any] | str | bytes | None = None,
        create_params: QFXfmCreateParams | None = None,
        capacity_bytes: int = 0,
        host_weight_bytes: int = 0,
    ) -> QFEngineHandle:
        eff_cfg = create_cfg if create_cfg is not None else config_json
        if create_params is not None:
            if weights_path is not None or eff_cfg is not None:
                raise ValueError("Pass either create_params or weights_path/create_cfg, not both")
            xfm = create_transformer(lib, create_params=create_params)
            eff_dev = int(create_params.device_id)
        else:
            if weights_path is None:
                raise ValueError("weights_path is required when create_params is not provided")
            xfm = create_transformer(
                lib,
                weights_path=weights_path,
                device_id=device_id,
                config_json=eff_cfg,
            )
            eff_dev = int(device_id)
        return cls(
            lib,
            xfm,
            device_id=eff_dev,
            capacity_bytes=capacity_bytes,
            host_weight_bytes=host_weight_bytes,
        )

    def __init__(
        self,
        lib: Any,
        xfm: ctypes.c_void_p | int | None,
        *,
        device_id: int = 0,
        capacity_bytes: int = 0,
        host_weight_bytes: int = 0,
    ) -> None:
        self.lib = lib
        self.xfm = xfm
        self.device_id = int(device_id)
        self.capacity_bytes = int(capacity_bytes)
        self.host_weight_bytes = int(host_weight_bytes if host_weight_bytes else capacity_bytes)
        self.current_session: ctypes.c_void_p | None = None
        self.step_count: int = 0
        self.applied_lora_sig: str = "[]"
        self._begin_keep: QFSessionBeginParams | None = None
        self._reference_audio_owner: Any = None
        self._reference_audio_pending: bool = False
        self._destroy_lock = threading.RLock()
        self._destroying: bool = False
        self._sys = sys
        if self.xfm is not None:
            with _LIVE_HANDLES_LOCK:
                _LIVE_HANDLES.add(self)

    def __enter__(self) -> QFEngineHandle:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()

    def __del__(self) -> None:
        sys_mod = getattr(self, "_sys", None) or globals().get("sys")
        if sys_mod is None or getattr(sys_mod, "is_finalizing", lambda: True)():
            return
        if (
            getattr(self, "_destroying", False)
            or getattr(self, "_reference_audio_pending", False)
            or getattr(self, "xfm", None) is None
        ):
            return
        self._run_destroy_scope(
            self._destroy_unlocked,
            nonblocking=True,
            suppress_errors=True,
            guard=lambda: (
                getattr(self, "xfm", None) is not None
                and not getattr(self, "_reference_audio_pending", False)
            ),
        )

    def _require_live(self, op: str) -> None:
        if self.xfm is None:
            raise RuntimeError(f"QFEngineHandle.{op}: no live native transformer handle")

    def _require_session(self, op: str) -> ctypes.c_void_p:
        if self.current_session is None:
            raise RuntimeError(f"QFEngineHandle.{op}: no active native session")
        return self.current_session

    def set_api_key(self, api_key: str) -> None:
        """Hot-switch the active API key in place via ``qf_auth_set_api_key``."""
        self._require_live("set_api_key")
        if not isinstance(api_key, str) or not api_key:
            raise InvalidArgError("api_key must be a non-empty string")
        st = self.lib.qf_auth_set_api_key(self.xfm, api_key.encode("utf-8"))
        raise_for_status(st, context="qf_auth_set_api_key", detail=last_err(self.lib))

    def set_loras(self, loras: Sequence[Any] | None) -> None:
        """Hot-swap the active LoRA stack via ``qf_xfm_set_loras`` (``[]`` clears)."""
        self._require_live("set_loras")
        payload = _normalize_lora_payload(loras)
        serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        st = self.lib.qf_xfm_set_loras(self.xfm, serialized.encode("utf-8"))
        raise_for_status(st, context="qf_xfm_set_loras", detail=last_err(self.lib))
        self.applied_lora_sig = serialized

    def unload_vram(self, bytes_requested: int = QF_UINT64_MAX) -> int:
        """Release up to ``bytes_requested`` bytes of engine VRAM via ``qf_xfm_unload_vram``."""
        if self.xfm is None:
            return 0
        if isinstance(bytes_requested, bool):
            raise ValueError("unload_vram bytes_requested must fit uint64, not bool")
        req = operator.index(bytes_requested)
        if not 0 <= req <= QF_UINT64_MAX:
            raise ValueError("unload_vram bytes_requested must fit uint64 [0, 2^64-1]")
        freed = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_unload_vram(self.xfm, req, ctypes.byref(freed))
        raise_for_status(st, context="qf_xfm_unload_vram", detail=last_err(self.lib))
        return int(freed.value)

    def unload_sync(self) -> int:
        """Explicit synchronization fence via ``qf_xfm_unload_vram(0)`` releasing pending reference-audio borrows."""
        self._require_live("unload_sync")
        self.unload_vram(0)
        self.reference_audio_copy_completed()
        self._reference_audio_owner = None
        return QF_OK

    def resident_vram_bytes(self) -> int:
        """Query currently resident engine VRAM bytes via ``qf_xfm_resident_vram_bytes``."""
        if self.xfm is None:
            return 0
        out = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_resident_vram_bytes(self.xfm, ctypes.byref(out))
        raise_for_status(st, context="qf_xfm_resident_vram_bytes", detail=last_err(self.lib))
        return int(out.value)

    def vram_need_bytes(self, latent_shape: Sequence[int]) -> int:
        """Query additional VRAM required for ``latent_shape`` via ``qf_xfm_vram_need_bytes``."""
        if self.xfm is None:
            return 0
        dims = [int(d) for d in latent_shape]
        if not dims or any(d <= 0 for d in dims):
            return 0
        arr = (ctypes.c_int64 * len(dims))(*dims)
        out = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_vram_need_bytes(self.xfm, arr, len(dims), ctypes.byref(out))
        raise_for_status(st, context="qf_xfm_vram_need_bytes", detail=last_err(self.lib))
        return int(out.value)

    def session_begin(
        self,
        width: int,
        height: int,
        num_steps: int,
        max_context_dims: Sequence[int],
        cond_dtype: int,
        options_dict: Mapping[str, Any] | str | bytes | None = None,
        *,
        producer_stream: int = 0,
    ) -> ctypes.c_void_p:
        """Begin a native denoising session with strict ``options_json`` allowlist validation."""
        self._require_live("session_begin")
        if len(max_context_dims) != 3:
            raise InvalidArgError(f"max_context_dims must have length 3, got {len(max_context_dims)}")
        encoded_opts = validate_session_options(options_dict)
        bpx = QFSessionBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(QFSessionBeginParams)
        bpx.width = int(width)
        bpx.height = int(height)
        bpx.num_steps = int(num_steps)
        bpx.max_context_dims = (ctypes.c_int32 * 3)(*(int(v) for v in max_context_dims))
        bpx.cond_dtype = int(cond_dtype)
        bpx.producer_stream = int(producer_stream)
        if encoded_opts is not None:
            bpx._opts = encoded_opts
            bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        self._begin_keep = bpx
        st = self.lib.qf_xfm_session_begin(self.xfm, ctypes.byref(bpx), ctypes.byref(session))
        raise_for_status(st, context="qf_xfm_session_begin", detail=last_err(self.lib))
        self.current_session = session
        return session

    def step(self, params: QFStepParams) -> None:
        """Execute a single denoising step via ``qf_xfm_step``."""
        sess = self._require_session("step")
        if not params.struct_size:
            params.struct_size = ctypes.sizeof(QFStepParams)
        st = self.lib.qf_xfm_step(sess, ctypes.byref(params))
        raise_for_status(st, context="qf_xfm_step", detail=last_err(self.lib))
        self.step_count += 1

    def step_av(self, params: QFStepAvParams) -> None:
        """Execute a joint audio-video denoising step via ``qf_xfm_step_av``."""
        sess = self._require_session("step_av")
        if not params.struct_size:
            params.struct_size = ctypes.sizeof(QFStepAvParams)
        if not params.base.struct_size:
            params.base.struct_size = ctypes.sizeof(QFStepParams)
        st = self.lib.qf_xfm_step_av(sess, ctypes.byref(params))
        raise_for_status(st, context="qf_xfm_step_av", detail=last_err(self.lib))
        self.step_count += 1

    def step_refs(self, params: QFStepRefsParams) -> None:
        """Execute a reference-image denoising step via ``qf_xfm_step_refs``."""
        sess = self._require_session("step_refs")
        if not params.struct_size:
            params.struct_size = ctypes.sizeof(QFStepRefsParams)
        if not params.base.struct_size:
            params.base.struct_size = ctypes.sizeof(QFStepParams)
        st = self.lib.qf_xfm_step_refs(sess, ctypes.byref(params))
        raise_for_status(st, context="qf_xfm_step_refs", detail=last_err(self.lib))
        self.step_count += 1

    def session_end(self) -> None:
        """Close the active session strictly; raises on native failure and retains session handle."""
        if self.current_session is None:
            return
        st = self.lib.qf_xfm_session_end(self.current_session)
        raise_for_status(st, context="qf_xfm_session_end", detail=last_err(self.lib))
        self.current_session = None
        self._reference_audio_owner = None
        self.reference_audio_copy_completed()

    def end_session_if_open(self) -> tuple[bool, bool]:
        """Close any open session cleanly; returns ``(was_open, succeeded)``."""
        sess = getattr(self, "current_session", None)
        if sess is None:
            return (False, True)
        ok = True
        try:
            lib = getattr(self, "lib", None)
            end_fn = getattr(lib, "qf_xfm_session_end", None)
            if not callable(end_fn):
                ok = False
            else:
                st = end_fn(sess)
                if st != QF_OK:
                    ok = False
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            self.current_session = None
            self._reference_audio_owner = None
            self.reference_audio_copy_completed()
        return (True, ok)

    def retain_reference_audio(self, owner: Any) -> None:
        """Pin a borrowed reference-audio owner until native async copy completion is proven."""
        if self.reference_audio_pending():
            raise RuntimeError("quantfunc: prior reference-audio begin has no proven copy completion")
        self._reference_audio_owner = owner
        self._reference_audio_pending = owner is not None
        if owner is not None:
            with _LIVE_HANDLES_LOCK:
                _REFERENCE_AUDIO_PENDING_OWNERS.add(self)

    def reference_audio_pending(self) -> bool:
        return bool(getattr(self, "_reference_audio_pending", False))

    def reference_audio_copy_completed(self) -> None:
        self._reference_audio_pending = False
        if not getattr(self, "_destroying", False):
            _queue_or_flush_discards(((self, False, True),))

    def _destroy_unlocked(self) -> None:
        """Close any active session and destroy ``xfm``; caller must hold ``_destroy_lock`` if present."""
        try:
            self.end_session_if_open()
        finally:
            xfm = getattr(self, "xfm", None)
            self.xfm = None
            if xfm is not None:
                destroy_fn = getattr(getattr(self, "lib", None), "qf_xfm_destroy", None)
                if callable(destroy_fn):
                    destroy_fn(xfm)

    def _run_destroy_scope(
        self,
        body: Callable[[], None],
        *,
        nonblocking: bool = False,
        suppress_errors: bool = False,
        guard: Callable[[], bool] | None = None,
    ) -> None:
        destroyed = False
        _enter_destroy_scope()
        try:
            lock = getattr(self, "_destroy_lock", None)
            acquired = False
            if lock is not None:
                acquired = lock.acquire(blocking=False) if nonblocking else lock.acquire()
                if not acquired:
                    return
            try:
                if getattr(self, "_destroying", False):
                    return
                if guard is not None and not guard():
                    return
                self._destroying = True
                destroyed = True
                try:
                    body()
                except Exception:  # noqa: BLE001
                    if not suppress_errors:
                        raise
                finally:
                    self._destroying = False
            finally:
                if acquired and lock is not None:
                    lock.release()
        finally:
            pending_flush = _exit_destroy_scope()
            if destroyed:
                pending_flush.append((self, True, not self.reference_audio_pending()))
            if pending_flush:
                _queue_or_flush_discards(pending_flush)

    def _require_no_pending_audio(self) -> bool:
        if self.reference_audio_pending():
            raise RuntimeError(
                "quantfunc: reference-audio copy is still pending; call unload_sync() or complete session close first"
            )
        return True

    def destroy(self) -> None:
        """Close any active session and destroy the native ``qf_xfm_t`` handle."""
        self._run_destroy_scope(self._destroy_unlocked, guard=self._require_no_pending_audio)

    def _release_pending_for_shutdown(self) -> Any:
        if not self.reference_audio_pending():
            return None
        for step in (self.unload_sync, self.end_session_if_open):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass
            if not self.reference_audio_pending():
                return None
        owner_keep = getattr(self, "_reference_audio_owner", None)
        self.reference_audio_copy_completed()
        self._reference_audio_owner = None
        return owner_keep

    def _destroy_for_shutdown(self) -> None:
        """Force-retire the handle during interpreter exit while CUDA and Python are healthy."""

        def _body() -> None:
            owner_keep = self._release_pending_for_shutdown()
            try:
                self._destroy_unlocked()
            finally:
                if owner_keep is not None:
                    owner_keep = None

        self._run_destroy_scope(_body)


def _shutdown_all_handles() -> None:
    """Destroy all live native handles before CPython finalizes and CUDA tears down."""
    global _REGISTRY_SNAPSHOT_DEPTH
    with _LIVE_HANDLES_LOCK:
        _REGISTRY_SNAPSHOT_DEPTH += 1
        try:
            seen: set[int] = set()
            handles: list[QFEngineHandle] = []
            for h in _LIVE_HANDLES:
                hid = id(h)
                if hid not in seen:
                    seen.add(hid)
                    handles.append(h)
            for h in _REFERENCE_AUDIO_PENDING_OWNERS:
                hid = id(h)
                if hid not in seen:
                    seen.add(hid)
                    handles.append(h)
        finally:
            _REGISTRY_SNAPSHOT_DEPTH -= 1
            if _REGISTRY_SNAPSHOT_DEPTH == 0 and _DEFERRED_REG_DISCARDS:
                _apply_registry_discards_locked(_DEFERRED_REG_DISCARDS)
                _DEFERRED_REG_DISCARDS.clear()
    for h in handles:
        try:
            h._destroy_for_shutdown()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_shutdown_all_handles)


__all__ = [
    "MAX_CREATE_JSON_BYTES",
    "MAX_SESSION_JSON_BYTES",
    "MAX_API_JSON_DEPTH",
    "FORBIDDEN_CREATE_KEYS",
    "CREATE_CONFIG_ALLOWLIST",
    "SESSION_OPTIONS_ALLOWLIST",
    "REJECTED_LEGACY_OPTION_KEYS",
    "SPARSE_SUBKEY_ALLOWLIST",
    "validate_create_config",
    "validate_session_options",
    "make_create_params",
    "estimate_vram",
    "create_transformer",
    "QFEngineHandle",
]
