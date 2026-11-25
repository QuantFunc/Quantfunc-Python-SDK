"""Sparse attention (Sol) configuration, sigma window interpolation, and legacy v1 migration.

100% model-agnostic; stdlib only.
"""

from __future__ import annotations

import logging
import math
import struct
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

_log = logging.getLogger(__name__)

_VALID_SELECTORS = ("off", "meansim", "svg2")
_VALID_SINK_MODES = ("off", "exact_kv", "exact_kv_and_rows")
_MAX_MIN_SEQUENCE_LENGTH = 1 << 20

_NO_SIGMAS_WINDOW_WARNED = False


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Sol {name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Sol {name} must be finite, got {value!r}")
    return out


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Sol {name} must be an integer, got {value!r}")
    return int(value)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Sol {name} must be boolean, got {value!r}")
    return value


def legacy_tau(ratio: float) -> float:
    """Convert a legacy v1 Sol ratio into the engine's capped Acklam FP32 tau."""
    val = _require_finite_number(ratio, "legacy ratio")
    p = min(1.0 - 1e-4, max(1e-4, 1.0 - _f32(val)))
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.3577518672690,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        0.007784695709041462,
        0.3224671290700398,
        2.445134137142996,
        3.754408661907416,
    )
    if p < 0.02425 or p > 0.97575:
        q = math.sqrt(-2.0 * math.log(p if p < 0.02425 else 1.0 - p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
        if p > 0.97575:
            x = -x
    else:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return _f32(x)


def percent_to_sigma(sigmas: Sequence[float], percent: float) -> float:
    """Interpolate a normalized progress ``percent`` in ``[0, 1]`` along ``sigmas``."""
    pct = _require_finite_number(percent, "percent")
    if not 0.0 <= pct <= 1.0:
        raise ValueError(f"percent must be in [0, 1], got {percent!r}")
    vals = [_require_finite_number(s, "sigma") for s in sigmas]
    if not vals:
        raise ValueError("sigmas must be a non-empty sequence")
    if len(vals) == 1:
        return vals[0]
    pos = pct * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(vals) - 1, lo + 1)
    frac = pos - lo
    out = vals[lo] * (1.0 - frac) + vals[hi] * frac
    if not math.isfinite(out):
        raise ValueError("percent_to_sigma produced a non-finite sigma")
    return float(out)


def _reset_no_sigmas_warning() -> None:
    """Internal helper for resetting the one-shot no-sigmas warning gate in tests."""
    global _NO_SIGMAS_WINDOW_WARNED
    _NO_SIGMAS_WINDOW_WARNED = False


@dataclass(frozen=True)
class SolConfig:
    """Model-agnostic sparse attention configuration mapped to native ``options_json["sparse"]``."""

    enabled: bool = False
    tau: float = _f32(1.3)
    selector: str = "svg2"
    window_enabled: bool = True
    start_percent: float = 0.2
    end_percent: float = 0.9
    sigma_start: float | None = None
    sigma_end: float | None = None
    min_sequence_length: int = 12288
    sink_conditioning: str = "exact_kv_and_rows"
    first_layers_dense: int | None = None
    first_steps_dense: int | None = None
    first_steps_dense_frac: float | None = None
    plan_interval: int | None = None
    auto_dense_guard: bool | None = None
    require_permute: bool | None = None
    top_p: float | None = None
    cdf: float | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "enabled")
        _require_bool(self.window_enabled, "window_enabled")

        tau = _require_finite_number(self.tau, "tau")
        if not -4.0 <= tau <= 4.0:
            raise ValueError(f"Sol tau must be within [-4.0, 4.0], got {tau!r}")
        object.__setattr__(self, "tau", _f32(tau))

        if not isinstance(self.selector, str) or self.selector not in _VALID_SELECTORS:
            raise ValueError(f"Sol selector must be one of {_VALID_SELECTORS}, got {self.selector!r}")

        start_p = _require_finite_number(self.start_percent, "start_percent")
        end_p = _require_finite_number(self.end_percent, "end_percent")
        if not 0.0 <= start_p <= end_p <= 1.0:
            raise ValueError(
                f"Sol window percentages must satisfy 0 <= start_percent <= end_percent <= 1, "
                f"got start_percent={start_p}, end_percent={end_p}"
            )
        object.__setattr__(self, "start_percent", start_p)
        object.__setattr__(self, "end_percent", end_p)

        if (self.sigma_start is None) != (self.sigma_end is None):
            raise ValueError("Sol sigma_start and sigma_end must both be set or both be None")
        if self.sigma_start is not None and self.sigma_end is not None:
            ss = _require_finite_number(self.sigma_start, "sigma_start")
            se = _require_finite_number(self.sigma_end, "sigma_end")
            if ss < se:
                raise ValueError(f"Sol sigma_start ({ss}) must be >= sigma_end ({se})")
            object.__setattr__(self, "sigma_start", ss)
            object.__setattr__(self, "sigma_end", se)

        min_len = _require_int(self.min_sequence_length, "min_sequence_length")
        if not 0 <= min_len <= _MAX_MIN_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sol min_sequence_length must be in [0, {_MAX_MIN_SEQUENCE_LENGTH}], got {min_len!r}"
            )
        object.__setattr__(self, "min_sequence_length", min_len)

        if not isinstance(self.sink_conditioning, str) or self.sink_conditioning not in _VALID_SINK_MODES:
            raise ValueError(
                f"Sol sink_conditioning must be one of {_VALID_SINK_MODES}, got {self.sink_conditioning!r}"
            )

        if self.first_layers_dense is not None:
            fld = _require_int(self.first_layers_dense, "first_layers_dense")
            if fld < 0:
                raise ValueError(f"Sol first_layers_dense must be >= 0, got {fld!r}")
            object.__setattr__(self, "first_layers_dense", fld)

        if self.first_steps_dense is not None:
            fsd = _require_int(self.first_steps_dense, "first_steps_dense")
            if fsd < 0:
                raise ValueError(f"Sol first_steps_dense must be >= 0, got {fsd!r}")
            object.__setattr__(self, "first_steps_dense", fsd)

        if self.first_steps_dense_frac is not None:
            fsf = _require_finite_number(self.first_steps_dense_frac, "first_steps_dense_frac")
            if not 0.0 <= fsf <= 1.0:
                raise ValueError(f"Sol first_steps_dense_frac must be in [0, 1], got {fsf!r}")
            object.__setattr__(self, "first_steps_dense_frac", fsf)

        if self.plan_interval is not None:
            pi = _require_int(self.plan_interval, "plan_interval")
            if pi < 1:
                raise ValueError(f"Sol plan_interval must be >= 1, got {pi!r}")
            object.__setattr__(self, "plan_interval", pi)

        if self.auto_dense_guard is not None:
            _require_bool(self.auto_dense_guard, "auto_dense_guard")

        if self.require_permute is not None:
            _require_bool(self.require_permute, "require_permute")

        if self.top_p is not None:
            tp = _require_finite_number(self.top_p, "top_p")
            if not 0.0 < tp <= 1.0:
                raise ValueError(f"Sol top_p must be in (0, 1], got {tp!r}")
            object.__setattr__(self, "top_p", tp)

        if self.cdf is not None:
            cf = _require_finite_number(self.cdf, "cdf")
            if not 0.0 < cf <= 1.0:
                raise ValueError(f"Sol cdf must be in (0, 1], got {cf!r}")
            object.__setattr__(self, "cdf", cf)

    @classmethod
    def from_legacy(cls, sol_tau: float | None = None) -> SolConfig:
        """Build a ``SolConfig`` from a legacy v1 fraction (0 or ~1 disables Sol)."""
        return from_legacy(sol_tau)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolConfig:
        """Build a validated ``SolConfig`` from a JSON mapping (accepts wire alias ``min_tokens``)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"SolConfig.from_dict requires a mapping, got {type(data).__name__}")
        valid_fields = frozenset(f.name for f in fields(cls))
        allowed = valid_fields | {"min_tokens"}
        unknown: list[str] = []
        for k in data:
            if not isinstance(k, str):
                raise ValueError(f"SolConfig keys must be strings, got {type(k).__name__}")
            if k not in allowed:
                unknown.append(k)
        if unknown:
            raise ValueError(f"Unknown SolConfig key(s): {', '.join(sorted(unknown))}")

        kwargs: dict[str, Any] = {}
        has_min_len = "min_sequence_length" in data
        has_min_tok = "min_tokens" in data
        if has_min_len and has_min_tok:
            v_len = _require_int(data["min_sequence_length"], "min_sequence_length")
            v_tok = _require_int(data["min_tokens"], "min_tokens")
            if v_len != v_tok:
                raise ValueError(
                    f"Conflicting SolConfig values for 'min_sequence_length' ({v_len}) and alias 'min_tokens' ({v_tok})"
                )
            kwargs["min_sequence_length"] = v_len
        elif has_min_tok:
            kwargs["min_sequence_length"] = data["min_tokens"]

        for k, v in data.items():
            if k == "min_tokens" or (k == "min_sequence_length" and has_min_tok):
                continue
            kwargs[k] = v

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all 18 canonical ``SolConfig`` fields into a JSON-compatible dictionary."""
        return {
            "enabled": bool(self.enabled),
            "tau": float(self.tau),
            "selector": str(self.selector),
            "window_enabled": bool(self.window_enabled),
            "start_percent": float(self.start_percent),
            "end_percent": float(self.end_percent),
            "sigma_start": None if self.sigma_start is None else float(self.sigma_start),
            "sigma_end": None if self.sigma_end is None else float(self.sigma_end),
            "min_sequence_length": int(self.min_sequence_length),
            "sink_conditioning": str(self.sink_conditioning),
            "first_layers_dense": None if self.first_layers_dense is None else int(self.first_layers_dense),
            "first_steps_dense": None if self.first_steps_dense is None else int(self.first_steps_dense),
            "first_steps_dense_frac": None if self.first_steps_dense_frac is None else float(self.first_steps_dense_frac),
            "plan_interval": None if self.plan_interval is None else int(self.plan_interval),
            "auto_dense_guard": None if self.auto_dense_guard is None else bool(self.auto_dense_guard),
            "require_permute": None if self.require_permute is None else bool(self.require_permute),
            "top_p": None if self.top_p is None else float(self.top_p),
            "cdf": None if self.cdf is None else float(self.cdf),
        }

    def to_native(
        self,
        sigmas: Sequence[float] | None = None,
        percent_to_sigma_fn: Callable[[float], float] | None = None,
    ) -> dict[str, Any] | None:
        """Serialize into the native engine ``options_json["sparse"]`` dictionary."""
        global _NO_SIGMAS_WINDOW_WARNED
        if not self.enabled:
            return None

        window: dict[str, float] | None = None
        if self.window_enabled:
            if self.sigma_start is not None and self.sigma_end is not None:
                window = {"sigma_start": float(self.sigma_start), "sigma_end": float(self.sigma_end)}
            elif percent_to_sigma_fn is not None:
                start = _require_finite_number(percent_to_sigma_fn(self.start_percent), "sigma_start")
                end = _require_finite_number(percent_to_sigma_fn(self.end_percent), "sigma_end")
                if start < end:
                    raise ValueError(f"Sol percent_to_sigma_fn returned invalid window bounds: {start} < {end}")
                window = {"sigma_start": start, "sigma_end": end}
            elif sigmas is not None:
                start = percent_to_sigma(sigmas, self.start_percent)
                end = percent_to_sigma(sigmas, self.end_percent)
                if start < end:
                    raise ValueError(f"Sol sigma schedule produced invalid window bounds: {start} < {end}")
                window = {"sigma_start": start, "sigma_end": end}
            else:
                if not _NO_SIGMAS_WINDOW_WARNED:
                    _NO_SIGMAS_WINDOW_WARNED = True
                    msg = (
                        "SolConfig(window_enabled=True) invoked without sigmas or explicit sigma_start/sigma_end; "
                        "disabling sigma window (window=None) for this session. Call attach_session_hooks(pipe) "
                        "or pass sigmas for exact windowing."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    _log.warning(msg)
                window = None

        payload: dict[str, Any] = {
            "enabled": True,
            "selector": self.selector,
            "tau": float(self.tau),
            "min_tokens": int(self.min_sequence_length),
            "sink_conditioning": self.sink_conditioning,
            "window": window,
        }
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.cdf is not None:
            payload["cdf"] = float(self.cdf)
        if self.first_layers_dense is not None:
            payload["first_layers_dense"] = int(self.first_layers_dense)
        if self.first_steps_dense is not None:
            payload["first_steps_dense"] = int(self.first_steps_dense)
        if self.first_steps_dense_frac is not None:
            payload["first_steps_dense_frac"] = float(self.first_steps_dense_frac)
        if self.plan_interval is not None:
            payload["plan_interval"] = int(self.plan_interval)
        if self.auto_dense_guard is not None:
            payload["auto_dense_guard"] = bool(self.auto_dense_guard)
        if self.require_permute is not None:
            payload["require_permute"] = bool(self.require_permute)
        return payload


def from_legacy(sol_tau: float | None = None) -> SolConfig:
    """Convert a legacy v1 fraction into a v2 ``SolConfig``."""
    ratio = 1.0 if sol_tau is None else _require_finite_number(sol_tau, "legacy fraction")
    if ratio == 0.0 or abs(ratio - 1.0) <= 1e-6:
        return SolConfig(enabled=False)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"Legacy Sol fraction must be in (0, 1), got {ratio!r}")
    return SolConfig(
        enabled=True,
        tau=legacy_tau(ratio),
        window_enabled=False,
        start_percent=0.0,
        end_percent=1.0,
        min_sequence_length=0,
        sink_conditioning="off",
    )


__all__ = [
    "SolConfig",
    "legacy_tau",
    "from_legacy",
    "percent_to_sigma",
]
