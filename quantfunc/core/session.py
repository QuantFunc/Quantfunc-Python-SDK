"""Model-agnostic 3-tier session lifecycle state machine, schedule tracking, and CFG key assignment."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantfunc.native.capi import QF_ERR_BUSY
from quantfunc.native.errors import EngineBusyError

DEFAULT_AUTO_SESSION_MAX_STEPS: int = 256
_DEFAULT_BUSY_RETRY_DELAY_S: float = 0.05
_SIGMA_EPS: float = 1e-6


class CtxKeyAssigner:
    """Assigns stable positive uint64 symbolic keys to conditioning groups within a session."""

    def __init__(self) -> None:
        self._id_to_key: dict[Any, int] = {}

    def reset(self) -> None:
        self._id_to_key.clear()

    def key(self, sig_id: Any) -> int:
        if sig_id is None:
            return 0
        existing = self._id_to_key.get(sig_id)
        if existing is not None:
            return existing
        assigned = len(self._id_to_key) + 1
        self._id_to_key[sig_id] = assigned
        return assigned

    @property
    def num_keys(self) -> int:
        return len(self._id_to_key)


@dataclass(frozen=True)
class StagedSchedule:
    """Exact denoising schedule pushed by Tier-1 pipeline hooks or Tier-3 explicit sessions."""

    sigmas: tuple[float, ...]
    num_steps: int
    guidance_scale: float = 1.0
    do_cfg: bool = False
    timestep_scale: float | None = None
    use_cursor_indexing: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps <= 0:
            raise ValueError(f"StagedSchedule.num_steps must be a positive integer, got {self.num_steps!r}")
        if isinstance(self.guidance_scale, bool) or not isinstance(self.guidance_scale, int | float):
            raise ValueError(f"StagedSchedule.guidance_scale must be numeric, got {self.guidance_scale!r}")
        gs = float(self.guidance_scale)
        if not math.isfinite(gs):
            raise ValueError(f"StagedSchedule.guidance_scale must be finite, got {gs!r}")
        object.__setattr__(self, "guidance_scale", gs)
        object.__setattr__(self, "do_cfg", bool(self.do_cfg))
        if self.timestep_scale is not None:
            if isinstance(self.timestep_scale, bool) or not isinstance(self.timestep_scale, int | float):
                raise ValueError(
                    f"StagedSchedule.timestep_scale must be a positive finite number or None, "
                    f"got {self.timestep_scale!r}"
                )
            ts = float(self.timestep_scale)
            if not math.isfinite(ts) or ts <= 0.0:
                raise ValueError(
                    f"StagedSchedule.timestep_scale must be a positive finite number or None, "
                    f"got {self.timestep_scale!r}"
                )
            object.__setattr__(self, "timestep_scale", ts)
        object.__setattr__(self, "use_cursor_indexing", bool(self.use_cursor_indexing))
        converted = tuple(float(s) for s in self.sigmas)
        if any(not math.isfinite(s) for s in converted):
            raise ValueError("StagedSchedule.sigmas must contain only finite floats")
        object.__setattr__(self, "sigmas", converted)

    def step_index_for_sigma(self, sigma: float) -> int:
        """Map a scalar sigma to the closest step index in ``[0, num_steps - 1]``."""
        if not self.sigmas:
            return 0
        limit = min(self.num_steps, len(self.sigmas))
        if limit <= 0:
            return 0
        target = float(sigma)
        return min(range(limit), key=lambda idx: abs(self.sigmas[idx] - target))


def _canonical_options_sig(options: Mapping[str, Any] | None) -> str:
    if not options:
        return "{}"
    return json.dumps(dict(options), sort_keys=True, separators=(",", ":"))


def _is_busy_exception(exc: BaseException) -> bool:
    if isinstance(exc, EngineBusyError):
        return True
    if getattr(exc, "status_code", None) == QF_ERR_BUSY:
        return True
    return "busy" in str(exc).lower()


class SamplingSession:
    """Tier-3 explicit session context manager guaranteeing session close on exit or error."""

    def __init__(
        self,
        handle: Any,
        *,
        tracker: AutoSessionTracker | None = None,
        width: int | None = None,
        height: int | None = None,
        num_steps: int | None = None,
        max_context_dims: Sequence[int] | None = None,
        cond_dtype: int | None = None,
        options: Mapping[str, Any] | None = None,
        sigmas: Sequence[float] | None = None,
        guidance_scale: float = 1.0,
        do_cfg: bool = False,
        producer_stream: int = 0,
    ) -> None:
        self.handle = handle
        self.tracker = tracker
        self.width = int(width) if width is not None else None
        self.height = int(height) if height is not None else None
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.max_context_dims = tuple(int(v) for v in max_context_dims) if max_context_dims is not None else None
        self.cond_dtype = int(cond_dtype) if cond_dtype is not None else None
        self.options = dict(options) if options is not None else None
        self.sigmas = tuple(float(s) for s in sigmas) if sigmas is not None else None
        self.guidance_scale = float(guidance_scale)
        self.do_cfg = bool(do_cfg)
        self.producer_stream = int(producer_stream)
        self.session_handle: Any = None
        self._active: bool = False

        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError(f"SamplingSession.num_steps must be positive, got {self.num_steps!r}")

    @property
    def schedule(self) -> StagedSchedule | None:
        if self.sigmas:
            steps = self.num_steps if self.num_steps is not None else max(1, len(self.sigmas) - 1)
            return StagedSchedule(
                sigmas=self.sigmas,
                num_steps=steps,
                guidance_scale=self.guidance_scale,
                do_cfg=self.do_cfg,
            )
        return None

    def __enter__(self) -> SamplingSession:
        self.handle.end_session_if_open()
        if (
            self.width is not None
            and self.height is not None
            and self.num_steps is not None
            and self.max_context_dims is not None
            and self.cond_dtype is not None
        ):
            self.session_handle = self.handle.session_begin(
                self.width,
                self.height,
                self.num_steps,
                self.max_context_dims,
                self.cond_dtype,
                self.options,
                producer_stream=self.producer_stream,
            )
        self._active = True
        if self.tracker is not None:
            self.tracker.enter_explicit_session(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.tracker is not None:
                self.tracker.exit_explicit_session()
        finally:
            self._active = False
            self.session_handle = None
            self.handle.end_session_if_open()


class AutoSessionTracker:
    """3-tier session tracker (Tier 3 explicit > Tier 1 staged hook > Tier 2 auto) with BUSY retry."""

    def __init__(
        self,
        handle: Any,
        *,
        default_steps: int = DEFAULT_AUTO_SESSION_MAX_STEPS,
        busy_retry_delay_s: float = _DEFAULT_BUSY_RETRY_DELAY_S,
    ) -> None:
        if isinstance(default_steps, bool) or not isinstance(default_steps, int) or default_steps <= 0:
            raise ValueError(f"default_steps must be a positive integer, got {default_steps!r}")
        self._handle = handle
        self._default_steps: int = int(default_steps)
        self._busy_retry_delay_s: float = float(max(0.0, busy_retry_delay_s))

        self._staged_schedule: StagedSchedule | None = None
        self._explicit_session: SamplingSession | None = None
        self._ctx_assigner = CtxKeyAssigner()

        self._session_open: bool = False
        self._bound_geom_key: Any = None
        self._bound_max_ctx_seq: int = 0
        self._bound_options_sig: str | None = None
        self._bound_total_steps: int = 0
        self._last_sigma: float | None = None
        self._step_index: int = -1
        self._step_cursor: int = 0
        self._same_sigma_branch_count: int = 0

    @property
    def default_steps(self) -> int:
        return self._default_steps

    @property
    def staged_schedule(self) -> StagedSchedule | None:
        return self._staged_schedule

    @property
    def explicit_session(self) -> SamplingSession | None:
        return self._explicit_session

    @property
    def tier(self) -> int:
        if self._explicit_session is not None:
            return 3
        if self._staged_schedule is not None:
            return 1
        return 2

    @property
    def current_step_index(self) -> int:
        return self._step_index

    @property
    def step_cursor(self) -> int:
        return self._step_cursor

    @property
    def bound_max_ctx_seq(self) -> int:
        return self._bound_max_ctx_seq

    @property
    def active_branch_count(self) -> int:
        return self._same_sigma_branch_count

    def set_step_cursor(self, idx: int) -> None:
        """Set the external 0-indexed scheduler step cursor (clamped >= 0)."""
        if isinstance(idx, bool):
            raise ValueError(f"step_cursor must be an integer, got {idx!r}")
        self._step_cursor = max(0, int(idx))

    def set_default_steps(self, n: int) -> None:
        """Override Tier-2 fallback step budget (must be a positive integer)."""
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"default_steps must be a positive integer, got {n!r}")
        self._default_steps = int(n)

    def set_staged_schedule(self, schedule: StagedSchedule | None) -> None:
        """Bind or clear a Tier-1 ``StagedSchedule``."""
        if schedule is not None and not isinstance(schedule, StagedSchedule):
            raise TypeError(f"schedule must be a StagedSchedule or None, got {type(schedule).__name__}")
        self._staged_schedule = schedule
        self._step_cursor = 0

    def set_schedule(self, schedule: StagedSchedule | None) -> None:
        """Alias for ``set_staged_schedule``."""
        self.set_staged_schedule(schedule)

    def _reset_session_tracking(self) -> None:
        self._session_open = False
        self._bound_geom_key = None
        self._bound_max_ctx_seq = 0
        self._bound_options_sig = None
        self._bound_total_steps = 0
        self._last_sigma = None
        self._step_index = -1
        self._same_sigma_branch_count = 0
        self._ctx_assigner.reset()

    def on_session_closed(self) -> None:
        """Reset cached tracking state after an external session close."""
        self._reset_session_tracking()
        self._step_cursor = 0

    def enter_explicit_session(self, session: SamplingSession) -> None:
        """Activate a Tier-3 explicit session (takes precedence over Tier-1 and Tier-2)."""
        self._reset_session_tracking()
        self._step_cursor = 0
        self._explicit_session = session
        if session.session_handle is not None or getattr(self._handle, "current_session", None) is not None:
            self._session_open = True
            self._bound_total_steps = int(
                session.num_steps
                if session.num_steps is not None
                else (len(session.sigmas) - 1 if session.sigmas and len(session.sigmas) > 1 else self._default_steps)
            )
            if session.max_context_dims is not None and len(session.max_context_dims) == 3:
                self._bound_max_ctx_seq = int(session.max_context_dims[1])
            if session.options is not None:
                self._bound_options_sig = _canonical_options_sig(session.options)

    def exit_explicit_session(self) -> None:
        """Deactivate the active Tier-3 explicit session and clear tracking state."""
        self._explicit_session = None
        self._reset_session_tracking()
        self._step_cursor = 0

    def _active_schedule_and_total_steps(self) -> tuple[StagedSchedule | None, int]:
        if self._explicit_session is not None:
            sched = self._explicit_session.schedule or self._staged_schedule
            if self._explicit_session.num_steps is not None:
                total = int(self._explicit_session.num_steps)
            elif sched is not None:
                total = int(sched.num_steps)
            else:
                total = int(self._default_steps)
            return sched, total
        if self._staged_schedule is not None:
            return self._staged_schedule, int(self._staged_schedule.num_steps)
        return None, int(self._default_steps)

    def _invoke_open_with_retry(
        self,
        open_fn: Callable[[int, dict[str, Any]], Any],
        total_steps: int,
        options: dict[str, Any],
        *,
        allow_busy_retry: bool = True,
    ) -> Any:
        try:
            return open_fn(total_steps, options)
        except BaseException as exc:
            if not allow_busy_retry or not _is_busy_exception(exc):
                raise
            if self._busy_retry_delay_s > 0.0:
                time.sleep(self._busy_retry_delay_s)
            self._handle.end_session_if_open()
            return open_fn(total_steps, options)

    def resolve_step_and_session(
        self,
        sigma: float,
        geom_key: Any,
        ctx_seq: int,
        options: Mapping[str, Any],
        open_fn: Callable[[int, dict[str, Any]], Any],
        *,
        cfg_group_idx: int | None = None,
        sig_id: Any = None,
        allow_busy_retry: bool = True,
    ) -> tuple[int, int, int]:
        """Ensure an active native session and return ``(step_index, total_steps, cfg_context_key)``."""
        cur_sigma = float(sigma)
        if not math.isfinite(cur_sigma):
            raise ValueError(f"sigma must be finite, got {sigma!r}")

        schedule, planned_steps = self._active_schedule_and_total_steps()
        opts_dict = dict(options)
        opts_sig = _canonical_options_sig(opts_dict)
        handle_has_session = getattr(self._handle, "current_session", None) is not None

        # Tier-3 pre-opened session: do not tear down the caller's explicit session unless not yet opened.
        explicit_preopened = self._explicit_session is not None and self._session_open and handle_has_session
        need_open = False
        if not explicit_preopened:
            if not self._session_open or not handle_has_session:
                need_open = True
            elif geom_key != self._bound_geom_key:
                need_open = True
            elif int(ctx_seq) > self._bound_max_ctx_seq:
                need_open = True
            elif opts_sig != self._bound_options_sig:
                need_open = True
            elif schedule is None:
                if self._last_sigma is not None and cur_sigma > self._last_sigma + _SIGMA_EPS:
                    need_open = True
                elif (
                    self._last_sigma is not None
                    and abs(cur_sigma - self._last_sigma) > _SIGMA_EPS
                    and self._step_index + 1 >= self._bound_total_steps
                ):
                    need_open = True
            elif planned_steps > self._bound_total_steps:
                need_open = True

        if need_open:
            self._handle.end_session_if_open()
            self._reset_session_tracking()
            self._invoke_open_with_retry(open_fn, planned_steps, opts_dict, allow_busy_retry=allow_busy_retry)
            self._session_open = True
            self._bound_geom_key = geom_key
            self._bound_max_ctx_seq = int(ctx_seq)
            self._bound_options_sig = opts_sig
            self._bound_total_steps = int(planned_steps)
        elif explicit_preopened and self._bound_geom_key is None:
            self._bound_geom_key = geom_key
            self._bound_max_ctx_seq = max(self._bound_max_ctx_seq, int(ctx_seq))

        eff_total = self._bound_total_steps if self._bound_total_steps > 0 else planned_steps

        same_sigma = self._last_sigma is not None and abs(cur_sigma - self._last_sigma) <= _SIGMA_EPS
        if schedule is not None:
            if schedule.use_cursor_indexing:
                step_index = min(self._step_cursor, max(0, eff_total - 1))
                if self._last_sigma is not None and step_index == self._step_index and same_sigma:
                    self._same_sigma_branch_count += 1
                else:
                    self._same_sigma_branch_count = 1
                self._step_index = step_index
            elif not schedule.sigmas:
                if self._step_cursor > 0:
                    step_index = min(self._step_cursor, max(0, eff_total - 1))
                    if self._last_sigma is not None and step_index == self._step_index and same_sigma:
                        self._same_sigma_branch_count += 1
                    else:
                        self._same_sigma_branch_count = 1
                    self._step_index = step_index
                elif self._last_sigma is None:
                    self._step_index = 0
                    self._same_sigma_branch_count = 1
                    step_index = 0
                elif same_sigma:
                    self._same_sigma_branch_count += 1
                    step_index = self._step_index
                else:
                    self._step_index = min(self._step_index + 1, max(0, eff_total - 1))
                    self._same_sigma_branch_count = 1
                    step_index = self._step_index
            else:
                step_index = schedule.step_index_for_sigma(cur_sigma)
                if step_index >= eff_total:
                    step_index = eff_total - 1
                if same_sigma:
                    self._same_sigma_branch_count += 1
                else:
                    self._same_sigma_branch_count = 1
                self._step_index = step_index
        else:
            if self._last_sigma is None:
                self._step_index = 0
                self._same_sigma_branch_count = 1
            elif same_sigma:
                self._same_sigma_branch_count += 1
            else:
                self._step_index = min(self._step_index + 1, max(0, eff_total - 1))
                self._same_sigma_branch_count = 1
            step_index = self._step_index

        self._last_sigma = cur_sigma

        if sig_id is not None:
            ctx_key = self._ctx_assigner.key(sig_id)
        elif cfg_group_idx is not None:
            ctx_key = self._ctx_assigner.key(("cfg_group", int(cfg_group_idx)))
        else:
            ctx_key = self._ctx_assigner.key(("branch", self._same_sigma_branch_count))

        return (int(step_index), int(eff_total), int(ctx_key))

    def prepare_step(
        self,
        *,
        width: int,
        height: int,
        sigma: float,
        max_context_dims: Sequence[int],
        cond_dtype: int,
        options: Mapping[str, Any],
        geometry_key: Any,
        ctx_identity: Any = None,
        producer_stream: int = 0,
        allow_busy_retry: bool = True,
        before_begin: Callable[[], None] | None = None,
        after_begin: Callable[[], None] | None = None,
    ) -> tuple[int, int, int]:
        """High-level wrapper over ``resolve_step_and_session`` driving ``handle.session_begin`` directly."""
        ctx_seq = int(max_context_dims[1]) if len(max_context_dims) >= 2 else 0

        def _open(num_steps: int, opts: dict[str, Any]) -> Any:
            if before_begin is not None:
                before_begin()
            sess = self._handle.session_begin(
                int(width),
                int(height),
                int(num_steps),
                tuple(int(d) for d in max_context_dims),
                int(cond_dtype),
                opts,
                producer_stream=int(producer_stream),
            )
            if after_begin is not None:
                after_begin()
            return sess

        return self.resolve_step_and_session(
            sigma=sigma,
            geom_key=geometry_key,
            ctx_seq=ctx_seq,
            options=options,
            open_fn=_open,
            sig_id=ctx_identity,
            allow_busy_retry=allow_busy_retry,
        )

    def end_session_if_open(self) -> tuple[bool, bool]:
        """Close any active native session via handle and reset tracker state."""
        was_open, ok = self._handle.end_session_if_open()
        if ok or not was_open:
            self._reset_session_tracking()
            self._step_cursor = 0
        return (bool(was_open), bool(ok))

    def close_if_open(self) -> bool:
        """Close any active native session and return whether a session was open."""
        was_open, _ok = self.end_session_if_open()
        return was_open


__all__ = [
    "DEFAULT_AUTO_SESSION_MAX_STEPS",
    "CtxKeyAssigner",
    "StagedSchedule",
    "SamplingSession",
    "AutoSessionTracker",
]
