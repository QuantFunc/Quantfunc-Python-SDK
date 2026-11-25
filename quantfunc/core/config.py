"""Model-agnostic engine configuration, attention backends, cache/quality dials, and secure auth management."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any

from quantfunc.native.handle import CREATE_CONFIG_ALLOWLIST
from quantfunc.native.installer import validate_https_url

DEFAULT_TIMESTEP_SCALE: float = 1000.0
DEFAULT_MAX_CONTEXT_SEQ: int = 4096
DEFAULT_QUALITY_RATIO: float = 0.80
DEFAULT_SERVER_URL: str = "https://service.quantfunc.com"

WIRE_QUALITY_RATIO_KEY: str = "cat_keep_ratio"

ATTENTION_BACKENDS: tuple[str, ...] = ("auto", "int8", "i4xi8", "int4", "fp16")
_ATTENTION_BACKEND_ALIASES: dict[str, str] = {
    "int4_pv8": "i4xi8",
    "sage": "int8",
    "fp16_native": "fp16",
    "native": "fp16",
    "flash": "fp16",
}

OFFLOAD_MODES: tuple[str, ...] = ("auto", "resident", "block_offload")
PRODUCER_STREAM_MODES: tuple[str, ...] = ("sync", "torch_current")
MAX_VRAM_BUDGET_MB: int = (1 << 44) - 1  # floor(UINT64_MAX / 1 MiB)
MAX_CACHE_THRESH: float = 100.0

_API_KEY_RE = re.compile(r"qf_[0-9a-f]{64}")
_ENV_API_KEY = "QUANTFUNC_API_KEY"
_ENV_SERVER_URL = "QF_SERVER_URL"
_ENV_CONFIG_PATH = "QF_CONFIG_PATH"
_ENV_LEGACY_KEYFILE = "QF_NATIVE_KEYFILE"
_ENV_CACHE_DIR = "QF_CACHE_DIR"


def default_config_path() -> str:
    """Return user config file path (``QF_CONFIG_PATH`` > ``QF_NATIVE_KEYFILE`` > ``~/.config/quantfunc/config.json``)."""
    for env_name in (_ENV_CONFIG_PATH, _ENV_LEGACY_KEYFILE):
        val = os.environ.get(env_name, "").strip()
        if val:
            return os.path.abspath(os.path.expanduser(val))
    return os.path.abspath(os.path.join(os.path.expanduser("~"), ".config", "quantfunc", "config.json"))


def default_cache_dir() -> str:
    """Return user cache root directory (``QF_CACHE_DIR`` > ``~/.cache/quantfunc``)."""
    env_val = os.environ.get(_ENV_CACHE_DIR, "").strip()
    if env_val:
        return os.path.abspath(os.path.expanduser(env_val))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.abspath(os.path.join(base, "quantfunc"))


def normalize_attention_backend(value: str | None) -> str:
    """Normalize an attention backend identifier or supported compatibility alias."""
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError(f"attention_backend must be a string, got {type(value).__name__}")
    raw = value.strip().lower()
    if not raw:
        return "auto"
    mapped = _ATTENTION_BACKEND_ALIASES.get(raw, raw)
    if mapped not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported attention_backend {value!r}; expected one of {ATTENTION_BACKENDS}"
        )
    return mapped


def _validate_finite_range(value: Any, name: str, lo: float, hi: float, *, strict_lo: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if (strict_lo and not (lo < out <= hi)) or (not strict_lo and not (lo <= out <= hi)):
        bracket = "(" if strict_lo else "["
        raise ValueError(f"{name} must be within {bracket}{lo}, {hi}], got {out!r}")
    return out


@dataclass(frozen=True)
class CacheConfig:
    """Model-agnostic step-cache and block-cache threshold dials."""

    step_cache: float = 0.0
    block_cache: float = 0.0

    def __post_init__(self) -> None:
        sc = _validate_finite_range(self.step_cache, "step_cache", 0.0, MAX_CACHE_THRESH)
        bc = _validate_finite_range(self.block_cache, "block_cache", 0.0, MAX_CACHE_THRESH)
        object.__setattr__(self, "step_cache", sc)
        object.__setattr__(self, "block_cache", bc)

    def to_session_options(self) -> dict[str, float]:
        opts: dict[str, float] = {}
        if self.step_cache > 0.0:
            opts["step_cache_thresh"] = float(self.step_cache)
        if self.block_cache > 0.0:
            opts["block_cache_thresh"] = float(self.block_cache)
        return opts


@dataclass(frozen=True)
class QualityConfig:
    """Quality enhancement mode controlling the native quality ratio wire parameter."""

    quality_enhance: bool = False
    default_ratio: float = DEFAULT_QUALITY_RATIO

    def __post_init__(self) -> None:
        if not isinstance(self.quality_enhance, bool):
            raise ValueError(f"quality_enhance must be bool, got {self.quality_enhance!r}")
        ratio = _validate_finite_range(self.default_ratio, "default_ratio", 0.0, 1.0, strict_lo=True)
        object.__setattr__(self, "default_ratio", ratio)

    def effective_ratio(self) -> float:
        return 1.0 if self.quality_enhance else float(self.default_ratio)


@dataclass
class EngineConfig:
    """Create-time engine configuration covering the strict 12-key C-API create allowlist."""

    model_arch: str = "auto"
    config_path: str | None = None
    connectors_path: str | None = None
    attention_backend: str = "auto"
    offload_mode: str = "auto"
    pinned_memory: bool = False
    vram_budget_mb: int = 0
    act_scale_g32: bool | None = None
    glu_fuse: bool | None = None
    server_url: str | None = None
    api_key: str | None = None
    cache_dir: str | None = None
    device_id: int = 0
    producer_stream_mode: str = "torch_current"

    def __post_init__(self) -> None:
        if not isinstance(self.model_arch, str) or not self.model_arch.strip():
            raise ValueError(f"model_arch must be a non-empty string, got {self.model_arch!r}")
        self.model_arch = self.model_arch.strip()
        self.attention_backend = normalize_attention_backend(self.attention_backend)

        if not isinstance(self.offload_mode, str) or self.offload_mode not in OFFLOAD_MODES:
            raise ValueError(f"offload_mode must be one of {OFFLOAD_MODES}, got {self.offload_mode!r}")

        if not isinstance(self.pinned_memory, bool):
            raise ValueError(f"pinned_memory must be bool, got {self.pinned_memory!r}")

        if isinstance(self.vram_budget_mb, bool) or not isinstance(self.vram_budget_mb, int):
            raise ValueError(f"vram_budget_mb must be an integer, got {self.vram_budget_mb!r}")
        if not 0 <= self.vram_budget_mb <= MAX_VRAM_BUDGET_MB:
            raise ValueError(f"vram_budget_mb must be in [0, {MAX_VRAM_BUDGET_MB}], got {self.vram_budget_mb!r}")

        if self.act_scale_g32 is not None and not isinstance(self.act_scale_g32, bool):
            raise ValueError(f"act_scale_g32 must be bool or None, got {self.act_scale_g32!r}")

        if self.glu_fuse is not None and not isinstance(self.glu_fuse, bool):
            raise ValueError(f"glu_fuse must be bool or None, got {self.glu_fuse!r}")

        if isinstance(self.device_id, bool) or not isinstance(self.device_id, int) or not 0 <= self.device_id < (1 << 31):
            raise ValueError(f"device_id must be a nonnegative int32, got {self.device_id!r}")

        if not isinstance(self.producer_stream_mode, str) or self.producer_stream_mode not in PRODUCER_STREAM_MODES:
            raise ValueError(
                f"producer_stream_mode must be one of {PRODUCER_STREAM_MODES}, got {self.producer_stream_mode!r}"
            )

        if self.server_url is not None and self.server_url != "":
            validate_https_url(self.server_url)

        if self.api_key is not None and self.api_key != "":
            validate_api_key(self.api_key)

    def to_create_dict(self) -> dict[str, Any]:
        """Serialize strictly to the 12-key C-API create ``config_json`` dictionary."""
        raw: dict[str, Any] = {
            "model_arch": self.model_arch,
            "attention_backend": self.attention_backend,
            "offload_mode": self.offload_mode,
            "pinned_memory": self.pinned_memory,
            "vram_budget_mb": self.vram_budget_mb,
        }
        if self.config_path:
            raw["config_path"] = os.fspath(self.config_path)
        if self.connectors_path:
            raw["connectors_path"] = os.fspath(self.connectors_path)
        if self.act_scale_g32 is not None:
            raw["act_scale_g32"] = self.act_scale_g32
        if self.glu_fuse is not None:
            raw["glu_fuse"] = self.glu_fuse
        if self.server_url:
            raw["server_url"] = self.server_url
        if self.api_key:
            raw["api_key"] = self.api_key
        if self.cache_dir:
            raw["cache_dir"] = os.fspath(self.cache_dir)
        return {k: v for k, v in raw.items() if k in CREATE_CONFIG_ALLOWLIST}


# ── API key validation, redaction, resolution & cross-platform atomic save ───


def validate_api_key(key: str) -> str:
    """Validate API key syntax (``qf_`` + 64 lowercase hex chars). Never emits raw input on error."""
    if not isinstance(key, str):
        raise ValueError("Invalid QuantFunc API key: expected string")
    cleaned = key.strip()
    if not _API_KEY_RE.fullmatch(cleaned):
        raise ValueError("Invalid QuantFunc API key format (expected 'qf_' followed by 64 lowercase hex characters)")
    return cleaned


def redact_api_key(key: str | None) -> str:
    """Return a console-safe redacted display preview of an API key (never reveals the secret body)."""
    if not key or not isinstance(key, str):
        return ""
    cleaned = key.strip()
    if not cleaned:
        return ""
    head = "qf_" if cleaned.startswith("qf_") else ""
    body = cleaned[len(head) :]
    if len(body) > 8:
        return f"{head}{body[:4]}...{body[-4:]}"
    return f"{head}***"


@dataclass(frozen=True)
class AuthConfig:
    """Resolved authentication configuration with redacted ``__repr__``."""

    api_key: str = ""
    server_url: str = DEFAULT_SERVER_URL
    cache_dir: str = ""
    source: str = "none"

    def __repr__(self) -> str:
        return (
            f"AuthConfig(api_key={redact_api_key(self.api_key)!r}, "
            f"server_url={self.server_url!r}, cache_dir={self.cache_dir!r}, source={self.source!r})"
        )


def _read_config_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path!r} must contain a JSON object")
    return data


def resolve_auth(
    *,
    api_key: str | None = None,
    server_url: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    keyfile: str | os.PathLike[str] | None = None,
    required: bool = False,
    require_key: bool = False,
) -> AuthConfig:
    """Resolve auth settings with priority: explicit arg > env var > config file > default."""
    cfg_path = os.path.abspath(os.path.expanduser(os.fspath(keyfile))) if keyfile else default_config_path()
    file_cfg: dict[str, Any] = {}
    if os.path.isfile(cfg_path):
        file_cfg = _read_config_json(cfg_path)

    resolved_key = ""
    source = "none"
    if api_key is not None and str(api_key).strip():
        resolved_key = validate_api_key(str(api_key))
        source = "explicit"
    else:
        env_key = os.environ.get(_ENV_API_KEY, "").strip()
        if env_key:
            resolved_key = validate_api_key(env_key)
            source = "env"
        else:
            file_key = file_cfg.get("api_key", "")
            if isinstance(file_key, str) and file_key.strip():
                resolved_key = validate_api_key(file_key)
                source = "config_file"

    if (required or require_key) and not resolved_key:
        raise ValueError(
            "QuantFunc API key is not configured. Set QUANTFUNC_API_KEY or run `quantfunc auth set-key`."
        )

    raw_url = (
        str(server_url).strip()
        if server_url is not None and str(server_url).strip()
        else (
            os.environ.get(_ENV_SERVER_URL, "").strip()
            or str(file_cfg.get("server_url") or "").strip()
            or DEFAULT_SERVER_URL
        )
    )
    resolved_url = validate_https_url(raw_url)

    resolved_cache = (
        os.path.abspath(os.path.expanduser(os.fspath(cache_dir)))
        if cache_dir is not None and str(cache_dir).strip()
        else str(file_cfg.get("cache_dir") or "").strip() or default_cache_dir()
    )

    return AuthConfig(
        api_key=resolved_key,
        server_url=resolved_url,
        cache_dir=resolved_cache,
        source=source,
    )


def save_api_key(
    key: str,
    keyfile: str | os.PathLike[str] | None = None,
    *,
    server_url: str | None = None,
) -> str:
    """Validate and atomically persist an API key to ``config.json`` (POSIX ``0o700`` dir / ``0o600`` file)."""
    validated_key = validate_api_key(key)
    validated_url = validate_https_url(server_url) if server_url is not None and server_url.strip() else None

    target = os.path.abspath(os.path.expanduser(os.fspath(keyfile))) if keyfile else default_config_path()
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    if os.name == "posix":
        os.chmod(parent, 0o700)

    existing: dict[str, Any] = {}
    if os.path.exists(target):
        existing = _read_config_json(target)
    existing["api_key"] = validated_key
    if validated_url is not None:
        existing["server_url"] = validated_url
    elif "server_url" not in existing:
        existing["server_url"] = DEFAULT_SERVER_URL

    payload = (json.dumps(existing, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp_path = os.path.join(parent, f".{os.path.basename(target)}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            if os.name == "posix" and hasattr(os, "fchmod"):
                os.fchmod(fh.fileno(), 0o600)
        os.replace(tmp_path, target)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return target


__all__ = [
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
]
