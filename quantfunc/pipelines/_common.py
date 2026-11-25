"""Shared internal L4 helpers for model-specific pipeline loaders and presets."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from quantfunc.accelerate.hooks import attach_session_hooks, enable_offload


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
) -> Any:
    """Attach Tier-1 session hooks and optional CPU offload to a loaded pipeline."""
    if attach_hooks:
        attach_session_hooks(pipe, primary_scheduler=primary_scheduler)
    if offload is not None:
        enable_offload(pipe, mode=offload)
    return pipe


__all__ = [
    "align_size",
    "finalize_pipeline_load",
    "import_diffusers_class",
]
