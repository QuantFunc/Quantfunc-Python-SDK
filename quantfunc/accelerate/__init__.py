"""Instance-level pipeline acceleration hooks and lifecycle utilities."""

from __future__ import annotations

from quantfunc.accelerate.hooks import (
    apply_cache_on_pipe,
    apply_sol_on_pipe,
    attach_session_hooks,
    detach_session_hooks,
    enable_offload,
    pipe_session,
)

__all__ = [
    "apply_cache_on_pipe",
    "apply_sol_on_pipe",
    "attach_session_hooks",
    "detach_session_hooks",
    "enable_offload",
    "pipe_session",
]
