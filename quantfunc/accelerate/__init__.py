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
from quantfunc.accelerate.serving import (
    apply_runtime_overrides,
    get_pipeline_lock,
    request_scope,
    restore_runtime_overrides,
)

__all__ = [
    "apply_cache_on_pipe",
    "apply_runtime_overrides",
    "apply_sol_on_pipe",
    "attach_session_hooks",
    "detach_session_hooks",
    "enable_offload",
    "get_pipeline_lock",
    "pipe_session",
    "request_scope",
    "restore_runtime_overrides",
]
