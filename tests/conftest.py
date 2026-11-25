"""Shared pure-CPU pytest fixtures and full 25-symbol FakeQFLib C-API simulator."""

from __future__ import annotations

import ctypes
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root is importable even before `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quantfunc.native.capi import (  # noqa: E402
    QF_ATTN_CAP_FP16,
    QF_ATTN_CAP_INT4,
    QF_ATTN_CAP_INT8,
    QF_DTYPE_BF16,
    QF_DTYPE_FP8_E4M3,
    QF_DTYPE_FP16,
    QF_DTYPE_FP32,
    QF_DTYPE_INT8,
    QF_DTYPE_INT32,
    QF_ERR_AUTH,
    QF_ERR_BUSY,
    QF_ERR_INVALID_ARG,
    QF_MAX_STEP_REF_AREA,
    QF_MAX_STEP_REFS,
    QF_OK,
    QFAuthConfig,
    QFDeviceCaps,
    QFSessionBeginParams,
    QFStepAvParams,
    QFStepParams,
    QFStepRefsParams,
    QFVramEstimate,
    QFXfmCreateParams,
)
from quantfunc.native.handle import (  # noqa: E402
    CREATE_CONFIG_ALLOWLIST,
    FORBIDDEN_CREATE_KEYS,
    MAX_API_JSON_DEPTH,
    MAX_CREATE_JSON_BYTES,
    MAX_SESSION_JSON_BYTES,
    REJECTED_LEGACY_OPTION_KEYS,
    SESSION_OPTIONS_ALLOWLIST,
    SPARSE_SUBKEY_ALLOWLIST,
)

_DTYPE_ITEMSIZE: dict[int, int] = {
    QF_DTYPE_FP32: 4,
    QF_DTYPE_FP16: 2,
    QF_DTYPE_BF16: 2,
    QF_DTYPE_INT8: 1,
    QF_DTYPE_INT32: 4,
    QF_DTYPE_FP8_E4M3: 1,
}

_VALID_MODEL_ARCHES = frozenset({"auto", "krea2", "qwenimage21", "ltx25", "minimax_h3"})
_VALID_OFFLOAD_MODES = frozenset({"auto", "resident", "block_offload"})
_VALID_ATTN_BACKENDS = frozenset({"auto", "int8", "int4_pv8", "i4xi8", "int4", "fp16"})


def _deref(arg: Any) -> Any:
    """Unwrap ctypes.byref / ctypes.pointer arguments transparently."""
    if arg is None:
        return None
    if isinstance(arg, ctypes._Pointer):
        return arg.contents
    if hasattr(arg, "_obj"):
        return arg._obj
    return arg


def _ptr_int(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, ctypes.c_void_p):
        return int(val.value or 0)
    return int(val)


def _decode_cstr(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    return str(val)


def _json_depth(obj: Any, depth: int = 0) -> int:
    if isinstance(obj, dict):
        if not obj:
            return depth + 1
        return max(_json_depth(v, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return depth + 1
        return max(_json_depth(v, depth + 1) for v in obj)
    return depth


def _contains_forbidden_create_key(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_CREATE_KEYS:
                return k
            hit = _contains_forbidden_create_key(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = _contains_forbidden_create_key(item)
            if hit is not None:
                return hit
    return None


def _parse_dims(dims_arr: Any, max_n: int) -> list[int] | None:
    out: list[int] = []
    seen_zero = False
    for i in range(max_n):
        d = int(dims_arr[i])
        if d < 0:
            return None
        if d == 0:
            seen_zero = True
        else:
            if seen_zero:
                return None
            out.append(d)
    return out if out else None


def _bf16_to_f32(u16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def _f32_to_bf16(val: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
    return (bits >> 16) & 0xFFFF


def _echo_cpu_buffer(src_ptr: int, dst_ptr: int, numel: int, dtype: int, scale: float) -> None:
    """Deterministically copy CPU tensor bytes from src_ptr to dst_ptr, applying scale."""
    itemsize = _DTYPE_ITEMSIZE.get(dtype, 2)
    nbytes = numel * itemsize
    if nbytes <= 0 or src_ptr == 0 or dst_ptr == 0:
        return
    if scale == 1.0:
        ctypes.memmove(dst_ptr, src_ptr, nbytes)
        return
    raw = ctypes.string_at(src_ptr, nbytes)
    if dtype == QF_DTYPE_FP32:
        vals = struct.unpack(f"<{numel}f", raw)
        out = struct.pack(f"<{numel}f", *(v * scale for v in vals))
    elif dtype == QF_DTYPE_FP16:
        vals = struct.unpack(f"<{numel}e", raw)
        out = struct.pack(f"<{numel}e", *(v * scale for v in vals))
    elif dtype == QF_DTYPE_BF16:
        u16s = struct.unpack(f"<{numel}H", raw)
        out = struct.pack(f"<{numel}H", *(_f32_to_bf16(_bf16_to_f32(u) * scale) for u in u16s))
    else:
        out = raw
    ctypes.memmove(dst_ptr, out, len(out))


class _FakeSym:
    """Callable wrapper supporting ctypes ``restype`` / ``argtypes`` attribute assignment."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class FakeQFLib:
    """Complete pure-CPU simulator of the frozen 25-symbol QuantFunc PC Engine C-API."""

    def __init__(
        self,
        *,
        busy_once: bool = False,
        fail_status: int | None = None,
        fail_session_end_status: int | None = None,
        echo_scale: float = 1.0,
        initial_resident_bytes: int = 8 * (1 << 30),
        estimate_resident_bytes: int = 8 * (1 << 30),
        estimate_min_offload_bytes: int = 2 * (1 << 30),
        estimate_working_floor_bytes: int = 512 * (1 << 20),
        estimate_recommended_offload_mode: int = 0,
    ) -> None:
        self.busy_once: bool = bool(busy_once)
        self.fail_status: int | None = fail_status
        self.fail_session_end_status: int | None = fail_session_end_status
        self.echo_scale: float = float(echo_scale)
        self.initial_resident_bytes: int = int(initial_resident_bytes)
        self.estimate_resident_bytes: int = int(estimate_resident_bytes)
        self.estimate_min_offload_bytes: int = int(estimate_min_offload_bytes)
        self.estimate_working_floor_bytes: int = int(estimate_working_floor_bytes)
        self.estimate_recommended_offload_mode: int = int(estimate_recommended_offload_mode)

        self.resident_bytes: int = 0
        self.log_level: int = 3
        self.log_stderr: int = 0
        self._last_error: bytes = b""
        self._next_xfm_id: int = 0x1000
        self._next_session_id: int = 0x2000
        self._live_xfms: dict[int, dict[str, Any]] = {}
        self._active_sessions: dict[int, dict[str, Any]] = {}
        self._xfm_to_session: dict[int, int] = {}

        # Recorded audit trails for tests
        self.auth_configs: list[dict[str, Any]] = []
        self.api_keys: list[tuple[int, str]] = []
        self.created_configs: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.loras_history: list[list[dict[str, Any]]] = []
        self.unloads: list[int] = []
        self.destroyed: list[int] = []

        # Expose all 25 symbols as bindable callables
        self.qf_version = _FakeSym(self._qf_version)
        self.qf_last_error = _FakeSym(self._qf_last_error)
        self.qf_set_log_level = _FakeSym(self._qf_set_log_level)
        self.qf_get_log_level = _FakeSym(self._qf_get_log_level)
        self.qf_set_log_stderr = _FakeSym(self._qf_set_log_stderr)
        self.qf_auth_configure = _FakeSym(self._qf_auth_configure)
        self.qf_auth_set_api_key = _FakeSym(self._qf_auth_set_api_key)
        self.qf_device_query_caps = _FakeSym(self._qf_device_query_caps)
        self.qf_xfm_estimate_vram = _FakeSym(self._qf_xfm_estimate_vram)
        self.qf_default_xfm_create_params = _FakeSym(self._qf_default_xfm_create_params)
        self.qf_xfm_create = _FakeSym(self._qf_xfm_create)
        self.qf_xfm_destroy = _FakeSym(self._qf_xfm_destroy)
        self.qf_xfm_set_loras = _FakeSym(self._qf_xfm_set_loras)
        self.qf_xfm_unload_vram = _FakeSym(self._qf_xfm_unload_vram)
        self.qf_xfm_resident_vram_bytes = _FakeSym(self._qf_xfm_resident_vram_bytes)
        self.qf_xfm_vram_need_bytes = _FakeSym(self._qf_xfm_vram_need_bytes)
        self.qf_default_session_begin_params = _FakeSym(self._qf_default_session_begin_params)
        self.qf_default_step_params = _FakeSym(self._qf_default_step_params)
        self.qf_default_step_av_params = _FakeSym(self._qf_default_step_av_params)
        self.qf_default_step_refs_params = _FakeSym(self._qf_default_step_refs_params)
        self.qf_xfm_session_begin = _FakeSym(self._qf_xfm_session_begin)
        self.qf_xfm_step = _FakeSym(self._qf_xfm_step)
        self.qf_xfm_step_av = _FakeSym(self._qf_xfm_step_av)
        self.qf_xfm_step_refs = _FakeSym(self._qf_xfm_step_refs)
        self.qf_xfm_session_end = _FakeSym(self._qf_xfm_session_end)

    def _fail(self, code: int, msg: str) -> int:
        self._last_error = msg.encode("utf-8", "replace")
        return code

    def _ok(self) -> int:
        self._last_error = b""
        return QF_OK

    # ── 1..5: Version & Diagnostics ──

    def _qf_version(self) -> bytes:
        return b"1.0.0-fake"

    def _qf_last_error(self) -> bytes:
        return self._last_error

    def _qf_set_log_level(self, level: int) -> None:
        self.log_level = int(level)

    def _qf_get_log_level(self) -> int:
        return self.log_level

    def _qf_set_log_stderr(self, level: int) -> None:
        self.log_stderr = int(level)

    # ── 6..7: Auth ──

    def _qf_auth_configure(self, cfg_ptr: Any) -> int:
        cfg = _deref(cfg_ptr)
        if cfg is None:
            self.auth_configs.append({"default": True})
            return self._ok()
        if int(cfg.struct_size) != ctypes.sizeof(QFAuthConfig):
            return self._fail(QF_ERR_INVALID_ARG, "qf_auth_configure: struct_size mismatch")
        if int(cfg.device_id) < 0:
            return self._fail(QF_ERR_INVALID_ARG, "qf_auth_configure: negative device_id")
        self.auth_configs.append(
            {
                "server_url": _decode_cstr(cfg.server_url),
                "api_key": _decode_cstr(cfg.api_key),
                "cache_dir": _decode_cstr(cfg.cache_dir),
                "device_id": int(cfg.device_id),
            }
        )
        return self._ok()

    def _qf_auth_set_api_key(self, xfm: Any, api_key: Any) -> int:
        key_str = _decode_cstr(api_key)
        if not key_str:
            return self._fail(QF_ERR_INVALID_ARG, "qf_auth_set_api_key: empty api_key")
        xfm_id = _ptr_int(xfm)
        if xfm_id != 0 and xfm_id not in self._live_xfms:
            return self._fail(QF_ERR_INVALID_ARG, "qf_auth_set_api_key: invalid or destroyed handle")
        self.api_keys.append((xfm_id, key_str))
        return self._ok()

    # ── 8..9: Device & VRAM Probe ──

    def _qf_device_query_caps(self, device_id: int, out_ptr: Any) -> int:
        out = _deref(out_ptr)
        if int(device_id) < 0 or out is None or int(out.struct_size) != ctypes.sizeof(QFDeviceCaps):
            return self._fail(QF_ERR_INVALID_ARG, "qf_device_query_caps: invalid arguments or struct_size")
        out.backend_type = 0
        out.device_id = int(device_id)
        out.arch_code = 89
        out.compute_unit_count = 142
        out.total_vram_bytes = 48 * (1 << 30)
        out.free_vram_bytes = 44 * (1 << 30)
        out.attn_caps_mask = QF_ATTN_CAP_FP16 | QF_ATTN_CAP_INT8 | QF_ATTN_CAP_INT4
        out.backend_feature_flags = 1
        out.device_name = b"Fake-GPU-SM89"
        out.uuid_hex = b"0" * 64
        out.plugin_path = b"/fake/libquantfunc_kernels.so"
        return self._ok()

    def _validate_create_json(self, raw_json: Any) -> tuple[dict[str, Any] | None, int, str]:
        if raw_json is None or raw_json == b"" or raw_json == "":
            return {}, QF_OK, ""
        raw_bytes = raw_json if isinstance(raw_json, bytes) else str(raw_json).encode("utf-8")
        if len(raw_bytes) > MAX_CREATE_JSON_BYTES:
            return None, QF_ERR_INVALID_ARG, "config_json exceeds 256 KiB limit"
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return None, QF_ERR_INVALID_ARG, f"config_json parse error: {exc}"
        if not isinstance(parsed, dict):
            return None, QF_ERR_INVALID_ARG, "config_json must be a JSON object"
        if _json_depth(parsed) > MAX_API_JSON_DEPTH:
            return None, QF_ERR_INVALID_ARG, "config_json exceeds max depth 16"
        bad = _contains_forbidden_create_key(parsed)
        if bad is not None:
            return None, QF_ERR_INVALID_ARG, f"forbidden key in config_json: {bad}"
        for k, v in parsed.items():
            if k not in CREATE_CONFIG_ALLOWLIST:
                return None, QF_ERR_INVALID_ARG, f"disallowed config_json key: {k}"
            if k == "model_arch" and v not in _VALID_MODEL_ARCHES:
                return None, QF_ERR_INVALID_ARG, f"unsupported config_json.model_arch: {v}"
            if k == "offload_mode" and v not in _VALID_OFFLOAD_MODES:
                return None, QF_ERR_INVALID_ARG, f"invalid config_json.offload_mode: {v}"
            if k == "attention_backend" and v not in _VALID_ATTN_BACKENDS:
                return None, QF_ERR_INVALID_ARG, f"invalid config_json.attention_backend: {v}"
            if k == "vram_budget_mb" and (isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > (1 << 44) - 1):
                return None, QF_ERR_INVALID_ARG, "invalid config_json.vram_budget_mb"
        return parsed, QF_OK, ""

    def _qf_xfm_estimate_vram(self, params_ptr: Any, out_ptr: Any) -> int:
        params = _deref(params_ptr)
        out = _deref(out_ptr)
        if (
            params is None
            or int(params.struct_size) != ctypes.sizeof(QFXfmCreateParams)
            or out is None
            or int(out.struct_size) != ctypes.sizeof(QFVramEstimate)
            or not _decode_cstr(params.weights_path)
            or int(params.device_id) < 0
        ):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_estimate_vram: invalid arguments")
        parsed, st, err = self._validate_create_json(params.config_json)
        if st != QF_OK or parsed is None:
            return self._fail(st, err)
        out.resident_weight_bytes = self.estimate_resident_bytes
        out.min_offload_bytes = self.estimate_min_offload_bytes
        out.working_set_floor_bytes = self.estimate_working_floor_bytes
        out.recommended_offload_mode = self.estimate_recommended_offload_mode
        out.reserved = 0
        return self._ok()

    # ── 10..12: Transformer Lifecycle ──

    def _qf_default_xfm_create_params(self) -> QFXfmCreateParams:
        return QFXfmCreateParams(struct_size=ctypes.sizeof(QFXfmCreateParams))

    def _qf_xfm_create(self, params_ptr: Any, out_xfm_ptr: Any) -> int:
        if self.fail_status is not None:
            code = self.fail_status
            if code == QF_ERR_AUTH:
                return self._fail(code, "injected auth failure")
            return self._fail(code, f"injected create failure status={code}")
        params = _deref(params_ptr)
        out_xfm = _deref(out_xfm_ptr)
        if (
            params is None
            or int(params.struct_size) != ctypes.sizeof(QFXfmCreateParams)
            or out_xfm is None
            or not _decode_cstr(params.weights_path)
            or int(params.device_id) < 0
        ):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_create: invalid arguments")
        parsed, st, err = self._validate_create_json(params.config_json)
        if st != QF_OK or parsed is None:
            return self._fail(st, err)

        self._next_xfm_id += 1
        hid = self._next_xfm_id
        rec = {
            "handle_id": hid,
            "weights_path": _decode_cstr(params.weights_path),
            "device_id": int(params.device_id),
            "config": dict(parsed),
            "model_arch": str(parsed.get("model_arch", "auto")),
        }
        self._live_xfms[hid] = rec
        self.created_configs.append(rec)
        self.resident_bytes = self.initial_resident_bytes
        out_xfm.value = hid
        return self._ok()

    def _qf_xfm_destroy(self, xfm: Any) -> None:
        hid = _ptr_int(xfm)
        if hid in self._live_xfms:
            sid = self._xfm_to_session.pop(hid, None)
            if sid is not None:
                self._active_sessions.pop(sid, None)
            del self._live_xfms[hid]
            self.destroyed.append(hid)
            if not self._live_xfms:
                self.resident_bytes = 0

    # ── 13..16: LoRA & VRAM Management ──

    def _qf_xfm_set_loras(self, xfm: Any, loras_json: Any) -> int:
        hid = _ptr_int(xfm)
        if hid not in self._live_xfms:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_set_loras: invalid handle")
        raw = _decode_cstr(loras_json)
        try:
            parsed = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            return self._fail(QF_ERR_INVALID_ARG, f"qf_xfm_set_loras: invalid json: {exc}")
        if not isinstance(parsed, list):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_set_loras: expected JSON array")
        self.loras_history.append(parsed)
        return self._ok()

    def _qf_xfm_unload_vram(self, xfm: Any, bytes_requested: int, out_freed_ptr: Any) -> int:
        hid = _ptr_int(xfm)
        out_freed = _deref(out_freed_ptr)
        if hid not in self._live_xfms or out_freed is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_unload_vram: invalid arguments")
        req = int(bytes_requested)
        freed = min(self.resident_bytes, req)
        self.resident_bytes -= freed
        out_freed.value = freed
        self.unloads.append(req)
        return self._ok()

    def _qf_xfm_resident_vram_bytes(self, xfm: Any, out_bytes_ptr: Any) -> int:
        hid = _ptr_int(xfm)
        out_bytes = _deref(out_bytes_ptr)
        if hid not in self._live_xfms or out_bytes is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_resident_vram_bytes: invalid arguments")
        out_bytes.value = self.resident_bytes
        return self._ok()

    def _qf_xfm_vram_need_bytes(self, xfm: Any, latent_dims: Any, ndims: int, out_bytes_ptr: Any) -> int:
        hid = _ptr_int(xfm)
        out_bytes = _deref(out_bytes_ptr)
        if hid not in self._live_xfms or out_bytes is None or int(ndims) <= 0:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_vram_need_bytes: invalid arguments")
        prod = 1
        for i in range(int(ndims)):
            d = int(latent_dims[i])
            if d <= 0:
                return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_vram_need_bytes: non-positive dim")
            prod *= d
        out_bytes.value = prod * 16 + (64 * 1024 * 1024)
        return self._ok()

    # ── 17..20: Session Parameter Factories ──

    def _qf_default_session_begin_params(self) -> QFSessionBeginParams:
        return QFSessionBeginParams(struct_size=ctypes.sizeof(QFSessionBeginParams))

    def _qf_default_step_params(self) -> QFStepParams:
        return QFStepParams(struct_size=ctypes.sizeof(QFStepParams))

    def _qf_default_step_av_params(self) -> QFStepAvParams:
        p = QFStepAvParams(struct_size=ctypes.sizeof(QFStepAvParams))
        p.base.struct_size = ctypes.sizeof(QFStepParams)
        p.audio_scale = 1.0
        return p

    def _qf_default_step_refs_params(self) -> QFStepRefsParams:
        p = QFStepRefsParams(struct_size=ctypes.sizeof(QFStepRefsParams))
        p.base.struct_size = ctypes.sizeof(QFStepParams)
        return p

    # ── 21..25: Session & Denoise Steps ──

    def _validate_session_json(
        self,
        raw_json: Any,
        model_arch: str,
    ) -> tuple[dict[str, Any] | None, int, str]:
        if raw_json is None or raw_json == b"" or raw_json == "":
            return {}, QF_OK, ""
        raw_bytes = raw_json if isinstance(raw_json, bytes) else str(raw_json).encode("utf-8")
        if len(raw_bytes) > MAX_SESSION_JSON_BYTES:
            return None, QF_ERR_INVALID_ARG, "options_json exceeds 1 MiB limit"
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return None, QF_ERR_INVALID_ARG, f"options_json parse error: {exc}"
        if not isinstance(parsed, dict):
            return None, QF_ERR_INVALID_ARG, "options_json must be a JSON object"
        if _json_depth(parsed) > MAX_API_JSON_DEPTH:
            return None, QF_ERR_INVALID_ARG, "options_json exceeds max depth 16"
        for k, v in parsed.items():
            if k in REJECTED_LEGACY_OPTION_KEYS:
                return None, QF_ERR_INVALID_ARG, f"removed legacy option key: {k}"
            if k not in SESSION_OPTIONS_ALLOWLIST:
                return None, QF_ERR_INVALID_ARG, f"unknown options_json key: {k}"
            if k == "audio_dims" and model_arch in ("krea2", "qwenimage21"):
                return None, QF_ERR_INVALID_ARG, f"audio_dims refused on image-only model: {model_arch}"
            if k == "sparse" and v is not None:
                if not isinstance(v, dict):
                    return None, QF_ERR_INVALID_ARG, "options_json.sparse must be object or null"
                for sk in v:
                    if sk not in SPARSE_SUBKEY_ALLOWLIST:
                        return None, QF_ERR_INVALID_ARG, f"unknown sparse key: {sk}"
            if k in ("av_sigma_shift_video", "av_sigma_shift_audio"):
                if not isinstance(v, int | float) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                    return None, QF_ERR_INVALID_ARG, f"invalid {k}"
        return parsed, QF_OK, ""

    def _qf_xfm_session_begin(self, xfm: Any, params_ptr: Any, out_session_ptr: Any) -> int:
        hid = _ptr_int(xfm)
        params = _deref(params_ptr)
        out_sess = _deref(out_session_ptr)
        if hid not in self._live_xfms or params is None or out_sess is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_session_begin: invalid arguments")
        if int(params.struct_size) != ctypes.sizeof(QFSessionBeginParams):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_session_begin: struct_size mismatch")
        if self.busy_once:
            self.busy_once = False
            return self._fail(QF_ERR_BUSY, "session busy (injected busy_once)")
        if hid in self._xfm_to_session:
            return self._fail(QF_ERR_BUSY, "transformer already has an active session")
        if int(params.width) <= 0 or int(params.height) <= 0 or int(params.num_steps) <= 0:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_session_begin: width/height/num_steps must be > 0")
        if int(params.cond_dtype) not in range(8):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_session_begin: invalid cond_dtype")

        model_arch = self._live_xfms[hid]["model_arch"]
        opts, st, err = self._validate_session_json(params.options_json, model_arch)
        if st != QF_OK or opts is None:
            return self._fail(st, err)

        self._next_session_id += 1
        sid = self._next_session_id
        sess_rec = {
            "session_id": sid,
            "xfm": hid,
            "width": int(params.width),
            "height": int(params.height),
            "num_steps": int(params.num_steps),
            "max_context_dims": [int(params.max_context_dims[i]) for i in range(3)],
            "cond_dtype": int(params.cond_dtype),
            "producer_stream": int(params.producer_stream),
            "options": dict(opts),
            "ended": False,
        }
        self._active_sessions[sid] = sess_rec
        self._xfm_to_session[hid] = sid
        self.sessions.append(sess_rec)
        out_sess.value = sid
        return self._ok()

    def _validate_base_step(
        self,
        sess_rec: dict[str, Any],
        base: QFStepParams,
    ) -> tuple[list[int] | None, int, str]:
        if int(base.struct_size) != ctypes.sizeof(QFStepParams):
            return None, QF_ERR_INVALID_ARG, "step: base.struct_size mismatch"
        sigma = float(base.sigma)
        if not math.isfinite(sigma) or sigma < 0.0 or sigma > 100.0:
            return None, QF_ERR_INVALID_ARG, f"step: sigma out of [0, 100]: {sigma}"
        if (
            int(base.total_steps) <= 0
            or int(base.total_steps) > int(sess_rec["num_steps"])
            or int(base.step_index) < 0
            or int(base.step_index) >= int(base.total_steps)
        ):
            return None, QF_ERR_INVALID_ARG, "step: step_index/total_steps out of session bounds"
        dims = _parse_dims(base.dims, 5)
        if dims is None:
            return None, QF_ERR_INVALID_ARG, "step: invalid latent dims"
        numel = math.prod(dims)
        itemsize = _DTYPE_ITEMSIZE.get(int(base.dtype))
        if itemsize is None:
            return None, QF_ERR_INVALID_ARG, f"step: unsupported dtype {base.dtype}"
        if int(base.velocity_out_capacity) < numel * itemsize:
            return None, QF_ERR_INVALID_ARG, "step: velocity_out_capacity too small"
        if _ptr_int(base.latent_in) == 0 or _ptr_int(base.velocity_out) == 0:
            return None, QF_ERR_INVALID_ARG, "step: null latent_in or velocity_out"

        max_ctx = sess_rec["max_context_dims"]
        expect_ctx = max_ctx[0] > 0 and max_ctx[1] > 0 and max_ctx[2] > 0
        c_dims = [int(base.context_dims[i]) for i in range(3)]
        if expect_ctx:
            if _ptr_int(base.context) == 0:
                return None, QF_ERR_INVALID_ARG, "step: null context when session expects conditioning"
            if int(base.context_dtype) != int(sess_rec["cond_dtype"]):
                return None, QF_ERR_INVALID_ARG, "step: context_dtype mismatch"
            if c_dims[0] != 1 or c_dims[1] <= 0 or c_dims[1] > max_ctx[1] or c_dims[2] != max_ctx[2]:
                return None, QF_ERR_INVALID_ARG, f"step: context_dims {c_dims} exceeds max {max_ctx}"
        else:
            if _ptr_int(base.context) != 0 or any(d != 0 for d in c_dims):
                return None, QF_ERR_INVALID_ARG, "step: unexpected context when max_context_dims is zero"

        return dims, QF_OK, ""

    def _qf_xfm_step(self, session: Any, params_ptr: Any) -> int:
        sid = _ptr_int(session)
        params = _deref(params_ptr)
        sess_rec = self._active_sessions.get(sid)
        if sess_rec is None or params is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step: invalid or closed session")
        dims, st, err = self._validate_base_step(sess_rec, params)
        if st != QF_OK or dims is None:
            return self._fail(st, err)
        _echo_cpu_buffer(
            _ptr_int(params.latent_in),
            _ptr_int(params.velocity_out),
            math.prod(dims),
            int(params.dtype),
            self.echo_scale,
        )
        self.steps.append(
            {
                "kind": "step",
                "session": sid,
                "dims": list(dims),
                "dtype": int(params.dtype),
                "sigma": float(params.sigma),
                "step_index": int(params.step_index),
                "total_steps": int(params.total_steps),
                "context_dims": [int(params.context_dims[i]) for i in range(3)],
                "context_dtype": int(params.context_dtype),
                "cfg_context_key": int(params.cfg_context_key),
                "producer_stream": int(params.producer_stream),
            }
        )
        return self._ok()

    def _qf_xfm_step_av(self, session: Any, params_ptr: Any) -> int:
        sid = _ptr_int(session)
        params = _deref(params_ptr)
        sess_rec = self._active_sessions.get(sid)
        if sess_rec is None or params is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: invalid or closed session")
        if int(params.struct_size) != ctypes.sizeof(QFStepAvParams):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: struct_size mismatch")
        dims, st, err = self._validate_base_step(sess_rec, params.base)
        if st != QF_OK or dims is None:
            return self._fail(st, err)
        adims = _parse_dims(params.audio_dims, 4)
        if adims is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: invalid audio_dims")
        aitem = _DTYPE_ITEMSIZE.get(int(params.audio_dtype))
        if aitem is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: invalid audio_dtype")
        anbytes = math.prod(adims) * aitem
        if int(params.audio_velocity_out_capacity) < anbytes:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: audio_velocity_out_capacity too small")
        if not math.isfinite(float(params.audio_scale)) or float(params.audio_scale) <= 0.0:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: audio_scale must be positive finite")
        if int(params.audio_reference_active) not in (0, 1):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_av: audio_reference_active must be 0 or 1")

        _echo_cpu_buffer(
            _ptr_int(params.base.latent_in),
            _ptr_int(params.base.velocity_out),
            math.prod(dims),
            int(params.base.dtype),
            self.echo_scale,
        )
        _echo_cpu_buffer(
            _ptr_int(params.audio_latent_in),
            _ptr_int(params.audio_velocity_out),
            math.prod(adims),
            int(params.audio_dtype),
            self.echo_scale,
        )
        self.steps.append(
            {
                "kind": "step_av",
                "session": sid,
                "dims": list(dims),
                "dtype": int(params.base.dtype),
                "sigma": float(params.base.sigma),
                "step_index": int(params.base.step_index),
                "total_steps": int(params.base.total_steps),
                "context_dims": [int(params.base.context_dims[i]) for i in range(3)],
                "context_dtype": int(params.base.context_dtype),
                "cfg_context_key": int(params.base.cfg_context_key),
                "producer_stream": int(params.base.producer_stream),
                "audio_dims": list(adims),
                "audio_dtype": int(params.audio_dtype),
                "audio_scale": float(params.audio_scale),
                "audio_reference_active": int(params.audio_reference_active),
            }
        )
        return self._ok()

    def _qf_xfm_step_refs(self, session: Any, params_ptr: Any) -> int:
        sid = _ptr_int(session)
        params = _deref(params_ptr)
        sess_rec = self._active_sessions.get(sid)
        if sess_rec is None or params is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_refs: invalid or closed session")
        if int(params.struct_size) != ctypes.sizeof(QFStepRefsParams):
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_refs: struct_size mismatch")
        dims, st, err = self._validate_base_step(sess_rec, params.base)
        if st != QF_OK or dims is None:
            return self._fail(st, err)
        nrefs = int(params.num_refs)
        if not 1 <= nrefs <= QF_MAX_STEP_REFS:
            return self._fail(QF_ERR_INVALID_ARG, f"qf_xfm_step_refs: num_refs {nrefs} out of [1, {QF_MAX_STEP_REFS}]")
        if not params.ref_latents or not params.ref_dims or not params.image_slots:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_refs: null ref_latents/ref_dims/image_slots")

        txt_len = int(params.base.context_dims[1])
        ref_dims_list: list[list[int]] = []
        slots_list: list[int] = []
        total_area = 0
        prev_slot = 0
        for i in range(nrefs):
            rptr = _ptr_int(params.ref_latents[i])
            if rptr == 0:
                return self._fail(QF_ERR_INVALID_ARG, f"qf_xfm_step_refs: null ref_latents[{i}]")
            rd = [int(params.ref_dims[4 * i + j]) for j in range(4)]
            if rd[0] != 1 or any(v <= 0 for v in rd):
                return self._fail(QF_ERR_INVALID_ARG, f"qf_xfm_step_refs: invalid ref_dims[{i}]={rd}")
            total_area += rd[2] * rd[3]
            if total_area > QF_MAX_STEP_REF_AREA:
                return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_step_refs: total ref area exceeds 2^24")
            slot = int(params.image_slots[i])
            if slot < prev_slot or slot > txt_len:
                return self._fail(QF_ERR_INVALID_ARG, f"qf_xfm_step_refs: invalid image_slot {slot}")
            prev_slot = slot
            ref_dims_list.append(rd)
            slots_list.append(slot)

        _echo_cpu_buffer(
            _ptr_int(params.base.latent_in),
            _ptr_int(params.base.velocity_out),
            math.prod(dims),
            int(params.base.dtype),
            self.echo_scale,
        )
        self.steps.append(
            {
                "kind": "step_refs",
                "session": sid,
                "dims": list(dims),
                "dtype": int(params.base.dtype),
                "sigma": float(params.base.sigma),
                "step_index": int(params.base.step_index),
                "total_steps": int(params.base.total_steps),
                "context_dims": [int(params.base.context_dims[i]) for i in range(3)],
                "context_dtype": int(params.base.context_dtype),
                "cfg_context_key": int(params.base.cfg_context_key),
                "producer_stream": int(params.base.producer_stream),
                "num_refs": nrefs,
                "ref_dims": ref_dims_list,
                "image_slots": slots_list,
            }
        )
        return self._ok()

    def _qf_xfm_session_end(self, session: Any) -> int:
        if self.fail_session_end_status is not None:
            return self._fail(self.fail_session_end_status, "injected session_end failure")
        sid = _ptr_int(session)
        sess_rec = self._active_sessions.pop(sid, None)
        if sess_rec is None:
            return self._fail(QF_ERR_INVALID_ARG, "qf_xfm_session_end: invalid or already closed session")
        sess_rec["ended"] = True
        self._xfm_to_session.pop(sess_rec["xfm"], None)
        return self._ok()


@pytest.fixture
def fake_qf_lib() -> FakeQFLib:
    """Provide a fresh pure-CPU FakeQFLib instance per test."""
    return FakeQFLib()
