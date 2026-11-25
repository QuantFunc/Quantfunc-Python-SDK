"""Pure-stdlib per-request runtime dial overrides and UNSET sentinel.

100% model-agnostic; stdlib + L1 config/sol/lora only (zero torch, zero model names).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .config import CacheConfig, normalize_attention_backend
from .lora import LoRAEntry, normalize_lora_entries
from .sol import SolConfig


class _UnsetType:
    """Immutable singleton sentinel distinguishing omitted fields from explicit ``None``."""

    __slots__ = ()
    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> _UnsetType:
        return self

    def __deepcopy__(self, memo: Any) -> _UnsetType:
        _ = memo
        return self

    def __reduce__(self) -> tuple[type[_UnsetType], tuple[()]]:
        return (type(self), ())


UNSET: Final[_UnsetType] = _UnsetType()

_CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "quality_enhance",
        "block_cache",
        "step_cache",
        "sol",
        "attention_backend",
        "loras",
        "default_steps",
    }
)
_ALIAS_MAP: dict[str, str] = {
    "fbcache": "block_cache",
    "stepcache": "step_cache",
}
_ALLOWED_KEYS: frozenset[str] = _CANONICAL_FIELDS | frozenset(_ALIAS_MAP.keys())


def _validate_cache_threshold(value: Any, field_name: str) -> float:
    if field_name == "block_cache":
        return CacheConfig(block_cache=value).block_cache
    return CacheConfig(step_cache=value).step_cache


@dataclass(frozen=True)
class RuntimeOverrides:
    """Strict, validated per-request runtime dial overrides."""

    quality_enhance: bool | _UnsetType = UNSET
    block_cache: float | _UnsetType = UNSET
    step_cache: float | _UnsetType = UNSET
    sol: SolConfig | None | _UnsetType = UNSET
    attention_backend: str | _UnsetType = UNSET
    loras: tuple[LoRAEntry, ...] | _UnsetType = UNSET
    default_steps: int | _UnsetType = UNSET

    def __post_init__(self) -> None:
        if self.quality_enhance is not UNSET:
            if not isinstance(self.quality_enhance, bool):
                raise ValueError(
                    f"RuntimeOverrides.quality_enhance must be bool, got {self.quality_enhance!r}"
                )

        if self.block_cache is not UNSET:
            object.__setattr__(self, "block_cache", _validate_cache_threshold(self.block_cache, "block_cache"))

        if self.step_cache is not UNSET:
            object.__setattr__(self, "step_cache", _validate_cache_threshold(self.step_cache, "step_cache"))

        if self.sol is not UNSET:
            if self.sol is None or isinstance(self.sol, SolConfig):
                pass
            elif isinstance(self.sol, Mapping):
                object.__setattr__(self, "sol", SolConfig.from_dict(self.sol))
            else:
                raise ValueError(
                    f"RuntimeOverrides.sol must be SolConfig, mapping, or None, got {type(self.sol).__name__}"
                )

        if self.attention_backend is not UNSET:
            if not isinstance(self.attention_backend, str) or not self.attention_backend.strip():
                raise ValueError(
                    f"RuntimeOverrides.attention_backend must be a non-empty string, got {self.attention_backend!r}"
                )
            if "\x00" in self.attention_backend:
                raise ValueError("RuntimeOverrides.attention_backend must not contain NUL bytes")
            object.__setattr__(
                self,
                "attention_backend",
                normalize_attention_backend(self.attention_backend),
            )

        if self.loras is not UNSET:
            if self.loras is None:
                raise ValueError("RuntimeOverrides.loras cannot be None; pass [] to clear active LoRAs")
            object.__setattr__(self, "loras", tuple(normalize_lora_entries(self.loras)))

        if self.default_steps is not UNSET:
            if (
                isinstance(self.default_steps, bool)
                or not isinstance(self.default_steps, int)
                or self.default_steps <= 0
            ):
                raise ValueError(
                    f"RuntimeOverrides.default_steps must be a positive integer, got {self.default_steps!r}"
                )
            object.__setattr__(self, "default_steps", int(self.default_steps))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeOverrides:
        """Parse a strict JSON mapping into ``RuntimeOverrides`` (supports ``fbcache`` / ``stepcache`` aliases)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"RuntimeOverrides.from_dict requires a mapping, got {type(data).__name__}")

        unknown: list[str] = []
        for k in data:
            if not isinstance(k, str):
                raise ValueError(f"RuntimeOverrides keys must be strings, got {type(k).__name__}")
            if k not in _ALLOWED_KEYS:
                unknown.append(k)
        if unknown:
            raise ValueError(f"Unknown RuntimeOverrides key(s): {', '.join(sorted(unknown))}")

        kwargs: dict[str, Any] = {}
        for alias, canonical in _ALIAS_MAP.items():
            has_alias = alias in data
            has_canon = canonical in data
            if has_alias and has_canon:
                v_alias = _validate_cache_threshold(data[alias], canonical)
                v_canon = _validate_cache_threshold(data[canonical], canonical)
                if v_alias != v_canon:
                    raise ValueError(
                        f"Conflicting RuntimeOverrides values for {canonical!r} ({v_canon}) "
                        f"and alias {alias!r} ({v_alias})"
                    )
                kwargs[canonical] = v_canon
            elif has_alias:
                kwargs[canonical] = data[alias]
            elif has_canon:
                kwargs[canonical] = data[canonical]

        for k, v in data.items():
            if k in _ALIAS_MAP or k in _ALIAS_MAP.values():
                continue
            kwargs[k] = v

        return cls(**kwargs)

    @classmethod
    def from_any(cls, value: RuntimeOverrides | Mapping[str, Any] | None) -> RuntimeOverrides:
        """Accept ``RuntimeOverrides``, a JSON mapping, ``None``, or ``UNSET``."""
        if value is None or value is UNSET:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise ValueError(
            f"RuntimeOverrides.from_any expects RuntimeOverrides, mapping, or None, got {type(value).__name__}"
        )

    def is_empty(self) -> bool:
        """Return ``True`` when every field is ``UNSET``."""
        return (
            self.quality_enhance is UNSET
            and self.block_cache is UNSET
            and self.step_cache is UNSET
            and self.sol is UNSET
            and self.attention_backend is UNSET
            and self.loras is UNSET
            and self.default_steps is UNSET
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all explicitly set (non-``UNSET``) fields to a JSON-compatible dictionary."""
        out: dict[str, Any] = {}
        if self.quality_enhance is not UNSET:
            out["quality_enhance"] = bool(self.quality_enhance)
        if self.block_cache is not UNSET:
            out["block_cache"] = float(self.block_cache)
        if self.step_cache is not UNSET:
            out["step_cache"] = float(self.step_cache)
        if self.sol is not UNSET:
            out["sol"] = None if self.sol is None else self.sol.to_dict()
        if self.attention_backend is not UNSET:
            out["attention_backend"] = str(self.attention_backend)
        if self.loras is not UNSET:
            out["loras"] = [{"path": entry.path, "scale": float(entry.scale)} for entry in self.loras]
        if self.default_steps is not UNSET:
            out["default_steps"] = int(self.default_steps)
        return out


__all__ = [
    "UNSET",
    "RuntimeOverrides",
]
