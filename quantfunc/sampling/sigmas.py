"""Sigma schedule construction, splitting, and denoise truncation utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _coerce_sigmas(sigmas: Sequence[float] | Any) -> list[float]:
    if sigmas is None:
        raise ValueError("sigmas must not be None.")
    if hasattr(sigmas, "detach") and hasattr(sigmas, "cpu") and hasattr(sigmas, "tolist"):
        raw = sigmas.detach().cpu().tolist()
    elif hasattr(sigmas, "tolist"):
        raw = sigmas.tolist()
    else:
        raw = list(sigmas)
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("sigmas must be a 1D sequence with at least 2 values.")

    vals: list[float] = []
    for idx, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"sigmas[{idx}] must be a real number, got {item!r}.")
        val = float(item)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"sigmas[{idx}] must be finite and >= 0, got {val!r}.")
        vals.append(val)

    for idx in range(len(vals) - 1):
        if vals[idx] <= vals[idx + 1]:
            raise ValueError(
                f"sigmas must be strictly decreasing, but sigmas[{idx}]={vals[idx]} "
                f"<= sigmas[{idx + 1}]={vals[idx + 1]}."
            )
    return vals


def build_linear_sigmas(num_steps: int, shift: float = 1.0) -> list[float]:
    """Build a strictly decreasing flow-matching sigma schedule from 1.0 to 0.0.

    Applies optional time-shift warping ``(shift * t) / (1 + (shift - 1) * t)``.
    Returns ``num_steps + 1`` sigma endpoints terminating at ``0.0``.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")
    if isinstance(shift, bool) or not isinstance(shift, int | float):
        raise ValueError(f"shift must be a finite positive float, got {shift!r}.")
    shift_f = float(shift)
    if not math.isfinite(shift_f) or shift_f <= 0.0:
        raise ValueError(f"shift must be finite and > 0, got {shift!r}.")

    result: list[float] = []
    for i in range(num_steps + 1):
        t = 1.0 - (i / num_steps)
        if t <= 0.0:
            sigma = 0.0
        elif t >= 1.0:
            sigma = 1.0
        else:
            sigma = (shift_f * t) / (1.0 + (shift_f - 1.0) * t)
        result.append(float(sigma))
    return result


def split_sigmas(
    sigmas: Sequence[float] | Any,
    split_step: int,
) -> tuple[list[float], list[float]]:
    """Split a sigma schedule at interior step index ``split_step``.

    Stage 1 integrates ``sigmas[:split_step + 1]`` (``split_step`` steps) and
    Stage 2 integrates ``sigmas[split_step:]`` (``N - split_step`` steps),
    sharing the exact boundary sigma ``sigmas[split_step]``.
    """
    vals = _coerce_sigmas(sigmas)
    n_steps = len(vals) - 1
    if isinstance(split_step, bool) or not isinstance(split_step, int):
        raise ValueError(f"split_step must be an integer, got {split_step!r}.")
    if split_step <= 0 or split_step >= n_steps:
        raise ValueError(
            f"split_step must satisfy 0 < split_step < {n_steps}, got {split_step}."
        )
    return vals[: split_step + 1], vals[split_step:]


def denoise_sigmas(
    sigmas: Sequence[float] | Any,
    denoise: float,
) -> list[float]:
    """Truncate a sigma schedule according to ``denoise`` in ``(0, 1]``."""
    vals = _coerce_sigmas(sigmas)
    if isinstance(denoise, bool) or not isinstance(denoise, int | float):
        raise ValueError(f"denoise must be a float in (0, 1], got {denoise!r}.")
    denoise_f = float(denoise)
    if not math.isfinite(denoise_f) or denoise_f <= 0.0 or denoise_f > 1.0:
        raise ValueError(f"denoise must be finite and in (0, 1], got {denoise!r}.")

    if denoise_f == 1.0:
        return list(vals)

    n_steps = len(vals) - 1
    keep = max(1, int(round(n_steps * denoise_f)))
    start = n_steps - keep
    if start > 0:
        return vals[start:]

    end_val = vals[-1]
    return [end_val + (v - end_val) * denoise_f for v in vals]
