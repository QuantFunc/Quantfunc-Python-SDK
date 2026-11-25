"""Shared internal L4 helpers for model-specific pipeline loaders and presets."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from quantfunc.accelerate.hooks import attach_session_hooks, enable_offload
from quantfunc.sampling.res_multistep import (
    FlowMatchResMultistepScheduler,
    use_res_multistep_scheduler,
)

_VALID_PIPELINE_SCHEDULERS: tuple[str, ...] = ("res_multistep", "euler", "keep")


def import_diffusers_class(
    module_name: str,
    class_name: str,
    *,
    fallback_names: Sequence[str] = (),
    repo_hint: str | None = None,
) -> Any:
    """Import ``class_name`` (or fallback aliases) from ``module_name`` with actionable errors."""
    names = (class_name, *fallback_names)
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{module_name} is required to load {class_name}. "
            "Install with `pip install quantfunc[diffusers]`."
        ) from exc

    for name in names:
        cls = getattr(mod, name, None)
        if cls is not None:
            return cls

    version = getattr(mod, "__version__", "unknown")
    hint = f" or the official model repository ({repo_hint})" if repo_hint else ""
    if len(names) == 1:
        subject = f"{names[0]} is not available"
    else:
        subject = f"None of {names} are available"
    raise ImportError(
        f"{subject} in installed {module_name} ({version}). "
        f"Install diffusers@main{hint}, or pass an explicit `pipeline_cls=` argument."
    )


def align_size(value: int | float, multiple: int) -> int:
    """Round ``value`` to the nearest positive integer multiple of ``multiple``."""
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise ValueError(f"multiple must be a positive integer, got {multiple!r}")
    return max(multiple, int(round(value / multiple)) * multiple)


def finalize_pipeline_load(
    pipe: Any,
    *,
    attach_hooks: bool = True,
    offload: str | None = None,
    primary_scheduler: Any | None = None,
    scheduler: str | None = "res_multistep",
) -> Any:
    """Configure default scheduler, attach Tier-1 session hooks, and optional CPU offload."""
    eff_scheduler = "keep" if scheduler is None else scheduler
    if not isinstance(eff_scheduler, str) or eff_scheduler not in _VALID_PIPELINE_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}; expected one of {_VALID_PIPELINE_SCHEDULERS} or None."
        )

    eff_primary = primary_scheduler
    if eff_scheduler == "res_multistep":
        active = use_res_multistep_scheduler(pipe, primary_scheduler=primary_scheduler)
        if primary_scheduler is not None:
            eff_primary = active
    elif eff_scheduler == "euler":
        target = primary_scheduler if primary_scheduler is not None else getattr(pipe, "scheduler", None)
        if isinstance(target, FlowMatchResMultistepScheduler):
            target.sampler_mode = "euler"

    if attach_hooks:
        attach_session_hooks(pipe, primary_scheduler=eff_primary)
    if offload is not None:
        enable_offload(pipe, mode=offload)
    return pipe


__all__ = [
    "align_size",
    "finalize_pipeline_load",
    "import_diffusers_class",
]
