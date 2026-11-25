"""Instance-level session, cache, Sol, and offload hooks for diffusion pipelines.

All hooks live strictly on object instance dictionaries (never mutating
``pipe.__class__``) and remain safe under ``copy.deepcopy(pipe.scheduler)``.
"""

from __future__ import annotations

import contextlib
import copy
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from quantfunc.core.session import StagedSchedule
from quantfunc.core.sol import SolConfig
from quantfunc.models.base import QuantFuncBaseTransformer

_MISSING = object()
_HOOK_STATE_ATTR = "_qf_hook_state"
_STEP_CALLS_ATTR = "_qf_step_calls"
_VALID_OFFLOAD_MODES = ("model", "low_vram")
_SIGMA_UNIT_TOL: float = 1e-4
_SIGMA_DUP_EPS: float = 1e-6
_HIGH_TIMESTEP_SCALE: float = 1000.0
_UNIT_TIMESTEP_SCALE: float = 1.0


class _DeepcopySafeMethodHook:
    """Instance-bound method wrapper that unwraps cleanly on ``copy.deepcopy``.

    When a pipeline duplicates its primary scheduler via
    ``copy.deepcopy(self.scheduler)``, Python invokes ``__deepcopy__`` on every
    entry in ``scheduler.__dict__``. Returning the unwrapped original method
    bound to the copied owner avoids copying hook closures or locks and ensures
    auxiliary scheduler copies never trigger primary session hooks.
    """

    def __init__(self, owner: Any, orig_method: Any, wrapper: Any) -> None:
        self._owner_id = id(owner)
        self._orig_method = orig_method
        self._orig_func = getattr(orig_method, "__func__", None)
        self._wrapper = wrapper

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        new_owner = memo.get(self._owner_id)
        if new_owner is not None and self._orig_func is not None:
            return types.MethodType(self._orig_func, new_owner)
        if new_owner is not None:
            return copy.deepcopy(self._orig_method, memo)
        return self._orig_method

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig_method, name)


@dataclass
class _PipeHookState:
    primary_scheduler: Any | None
    saved_sched_attrs: dict[str, Any]
    saved_pipe_attrs: dict[str, Any]
    total_step_calls: int | None = None
    step_calls: int = 0
    low_vram: bool = False


def _is_quantfunc_transformer(obj: Any) -> bool:
    if obj is None:
        return False
    return isinstance(obj, QuantFuncBaseTransformer) or bool(
        getattr(obj, "_is_quantfunc_transformer", False)
    )


def discover_transformers(pipe: Any) -> list[Any]:
    """Return all unique QuantFunc transformer instances attached to ``pipe``."""
    found: list[Any] = []
    seen: set[int] = set()

    def _consider(candidate: Any) -> None:
        if _is_quantfunc_transformer(candidate):
            cid = id(candidate)
            if cid not in seen:
                seen.add(cid)
                found.append(candidate)

    _consider(getattr(pipe, "transformer", None))
    _consider(getattr(pipe, "transformer_ref", None))

    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        for comp in components.values():
            _consider(comp)

    pipe_dict = getattr(pipe, "__dict__", None)
    if isinstance(pipe_dict, dict):
        for key, val in pipe_dict.items():
            if key == _HOOK_STATE_ATTR:
                continue
            _consider(val)

    return found


def _to_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if hasattr(values, "detach") and hasattr(values, "cpu") and hasattr(values, "tolist"):
        raw = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        try:
            raw = list(values)
        except TypeError:
            return None
    if not isinstance(raw, list):
        return [float(raw)]
    return [float(v) for v in raw]


def _set_transformer_hook_cursor(transformer: Any, cursor: int) -> None:
    setter = getattr(transformer, "set_hook_step_cursor", None)
    if callable(setter):
        setter(int(cursor))


def _infer_timestep_scale(ts_list: list[float] | None) -> float | None:
    if not ts_list:
        return None
    return _HIGH_TIMESTEP_SCALE if max(ts_list) > 1.0 else _UNIT_TIMESTEP_SCALE


def _should_use_cursor_indexing(sigmas: tuple[float, ...], total_calls: int | None) -> bool:
    if not sigmas:
        return True
    if any(s > 1.0 + _SIGMA_UNIT_TOL for s in sigmas):
        return True
    if total_calls is not None and len(sigmas) != total_calls + 1:
        return True
    return any(abs(a - b) <= _SIGMA_DUP_EPS for a, b in zip(sigmas, sigmas[1:], strict=False))


def _close_transformer_session(transformer: Any, *, unload_low_vram: bool = False) -> None:
    try:
        if callable(getattr(transformer, "end_session_if_open", None)):
            transformer.end_session_if_open()
        else:
            auto_session = getattr(transformer, "_auto_session", None)
            if auto_session is not None and callable(getattr(auto_session, "close", None)):
                auto_session.close()
            handle = getattr(transformer, "_handle", None)
            if handle is not None and callable(getattr(handle, "end_session_if_open", None)):
                handle.end_session_if_open()
    finally:
        _set_transformer_hook_cursor(transformer, 0)

    if unload_low_vram and callable(getattr(transformer, "unload_vram", None)):
        transformer.unload_vram()


def _close_all_sessions(pipe: Any, *, unload_low_vram: bool = False) -> None:
    first_error: BaseException | None = None
    for transformer in discover_transformers(pipe):
        try:
            _close_transformer_session(transformer, unload_low_vram=unload_low_vram)
        except BaseException as exc:  # pragma: no cover - preserve first cleanup error
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _push_schedule_to_transformer(
    transformer: Any,
    schedule: StagedSchedule,
    sigmas: list[float] | None,
) -> None:
    if callable(getattr(transformer, "set_staged_schedule", None)):
        transformer.set_staged_schedule(schedule)
    elif callable(getattr(transformer, "_set_staged_schedule", None)):
        transformer._set_staged_schedule(schedule)
    else:
        auto_session = getattr(transformer, "_auto_session", None)
        if auto_session is not None and callable(getattr(auto_session, "stage_schedule", None)):
            auto_session.stage_schedule(schedule)
        elif hasattr(transformer, "_staged_schedule"):
            transformer._staged_schedule = schedule

    if sigmas is not None:
        if callable(getattr(transformer, "set_sol_sigmas", None)):
            transformer.set_sol_sigmas(sigmas)
        elif hasattr(transformer, "_sol_sigmas"):
            transformer._sol_sigmas = list(sigmas)


def _infer_total_step_calls(sched: Any, sigmas: list[float] | None) -> int | None:
    timesteps = getattr(sched, "timesteps", None)
    if timesteps is not None:
        try:
            length = len(timesteps)
            if length > 0:
                return int(length)
        except TypeError:
            pass
    if sigmas is not None and len(sigmas) > 0:
        if len(sigmas) > 1:
            return len(sigmas) - 1
        return len(sigmas)
    return None


def _save_and_wrap(
    target: Any,
    attr: str,
    saved: dict[str, Any],
    wrapper_factory: Any,
) -> None:
    orig_attr = getattr(target, attr, None)
    if not callable(orig_attr):
        return
    if attr not in saved:
        target_dict = target.__dict__
        saved[attr] = target_dict[attr] if attr in target_dict else _MISSING
    hook = _DeepcopySafeMethodHook(target, orig_attr, wrapper_factory(orig_attr))
    target.__dict__[attr] = hook


def _restore_attrs(target: Any, saved: dict[str, Any]) -> None:
    if target is None or not hasattr(target, "__dict__"):
        return
    target_dict = target.__dict__
    for attr, prev in saved.items():
        if prev is _MISSING:
            target_dict.pop(attr, None)
        else:
            target_dict[attr] = prev


def _get_hook_state(pipe: Any) -> _PipeHookState | None:
    pipe_dict = getattr(pipe, "__dict__", None)
    if not isinstance(pipe_dict, dict):
        return None
    state = pipe_dict.get(_HOOK_STATE_ATTR)
    return state if isinstance(state, _PipeHookState) else None


def attach_session_hooks(pipe: Any, *, primary_scheduler: Any | None = None) -> Any:
    """Attach instance-level Tier-1 session lifecycle hooks to ``pipe``.

    Only the single primary scheduler (``primary_scheduler`` if supplied, else
    ``pipe.scheduler``) is wrapped. Auxiliary schedulers attached to the
    pipeline or created via ``copy.deepcopy(pipe.scheduler)`` are never hooked.
    ``pipe.__class__`` is never modified.
    """
    existing = _get_hook_state(pipe)
    prev_low_vram = existing.low_vram if existing is not None else False
    if existing is not None:
        detach_session_hooks(pipe)

    sched = primary_scheduler if primary_scheduler is not None else getattr(pipe, "scheduler", None)
    state = _PipeHookState(
        primary_scheduler=sched,
        saved_sched_attrs={},
        saved_pipe_attrs={},
        total_step_calls=None,
        step_calls=0,
        low_vram=prev_low_vram,
    )
    pipe.__dict__[_HOOK_STATE_ATTR] = state

    if sched is not None and hasattr(sched, "__dict__"):
        def _make_set_timesteps_wrapper(orig_set_timesteps: Any) -> Any:
            def _wrapped_set_timesteps(*args: Any, **kwargs: Any) -> Any:
                _close_all_sessions(pipe, unload_low_vram=False)
                result = orig_set_timesteps(*args, **kwargs)

                sigmas_list = _to_float_list(getattr(sched, "sigmas", None))
                ts_list = _to_float_list(getattr(sched, "timesteps", None))
                total_calls = _infer_total_step_calls(sched, sigmas_list)
                state.total_step_calls = total_calls
                state.step_calls = 0
                sched.__dict__[_STEP_CALLS_ATTR] = 0

                guidance_scale_raw = getattr(pipe, "guidance_scale", None)
                guidance_scale = 1.0
                if guidance_scale_raw is not None:
                    try:
                        guidance_scale = float(guidance_scale_raw)
                    except (TypeError, ValueError):
                        guidance_scale = 1.0
                do_cfg = bool(getattr(pipe, "do_classifier_free_guidance", False))

                transformers = discover_transformers(pipe)
                if total_calls is not None and total_calls > 0:
                    sigmas_tuple = tuple(sigmas_list) if sigmas_list is not None else ()
                    timestep_scale = _infer_timestep_scale(ts_list)
                    use_cursor_indexing = _should_use_cursor_indexing(sigmas_tuple, total_calls)
                    schedule = StagedSchedule(
                        sigmas=sigmas_tuple,
                        num_steps=total_calls,
                        guidance_scale=guidance_scale,
                        do_cfg=do_cfg,
                        timestep_scale=timestep_scale,
                        use_cursor_indexing=use_cursor_indexing,
                    )
                    for transformer in transformers:
                        _push_schedule_to_transformer(transformer, schedule, sigmas_list)
                        _set_transformer_hook_cursor(transformer, 0)
                else:
                    for transformer in transformers:
                        _set_transformer_hook_cursor(transformer, 0)
                return result

            return _wrapped_set_timesteps

        def _make_step_wrapper(orig_step: Any) -> Any:
            def _wrapped_step(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = orig_step(*args, **kwargs)
                except BaseException:
                    _close_all_sessions(pipe, unload_low_vram=state.low_vram)
                    raise

                state.step_calls += 1
                sched.__dict__[_STEP_CALLS_ATTR] = state.step_calls
                for transformer in discover_transformers(pipe):
                    _set_transformer_hook_cursor(transformer, state.step_calls)

                total = state.total_step_calls
                cur_idx = getattr(sched, "step_index", None)
                done = (
                    (total is not None and cur_idx is not None and cur_idx >= total)
                    or (total is not None and state.step_calls >= total)
                    or bool(getattr(pipe, "interrupt", False))
                )
                if done:
                    _close_all_sessions(pipe, unload_low_vram=state.low_vram)
                return result

            return _wrapped_step

        _save_and_wrap(sched, "set_timesteps", state.saved_sched_attrs, _make_set_timesteps_wrapper)
        _save_and_wrap(sched, "step", state.saved_sched_attrs, _make_step_wrapper)

    if hasattr(pipe, "__dict__") and callable(getattr(pipe, "maybe_free_model_hooks", None)):
        def _make_free_hooks_wrapper(orig_free: Any) -> Any:
            def _wrapped_free(*args: Any, **kwargs: Any) -> Any:
                try:
                    return orig_free(*args, **kwargs)
                finally:
                    _close_all_sessions(pipe, unload_low_vram=state.low_vram)

            return _wrapped_free

        _save_and_wrap(pipe, "maybe_free_model_hooks", state.saved_pipe_attrs, _make_free_hooks_wrapper)

    return pipe


def detach_session_hooks(pipe: Any) -> Any:
    """Restore primary scheduler and pipeline instance dictionaries."""
    pipe_dict = getattr(pipe, "__dict__", None)
    if not isinstance(pipe_dict, dict):
        return pipe
    state = pipe_dict.pop(_HOOK_STATE_ATTR, None)
    if not isinstance(state, _PipeHookState):
        return pipe

    if state.primary_scheduler is not None:
        _restore_attrs(state.primary_scheduler, state.saved_sched_attrs)
        if hasattr(state.primary_scheduler, "__dict__"):
            state.primary_scheduler.__dict__.pop(_STEP_CALLS_ATTR, None)
    _restore_attrs(pipe, state.saved_pipe_attrs)
    return pipe


@contextlib.contextmanager
def pipe_session(pipe: Any) -> Iterator[Any]:
    """Context manager guaranteeing session cleanup on exit or exception."""
    try:
        yield pipe
    finally:
        state = _get_hook_state(pipe)
        low_vram = bool(state.low_vram) if state is not None else False
        _close_all_sessions(pipe, unload_low_vram=low_vram)


def apply_cache_on_pipe(
    pipe: Any,
    block_cache: float | None = None,
    step_cache: float | None = None,
    residual_diff_threshold: float | None = None,
) -> Any:
    """Configure block/step caching on all QuantFunc transformers on ``pipe``."""
    if block_cache is not None and residual_diff_threshold is not None:
        if float(block_cache) != float(residual_diff_threshold):
            raise ValueError(
                "Conflicting values for block_cache and residual_diff_threshold; "
                "provide only one or equal values."
            )
    eff_block_cache = block_cache if block_cache is not None else residual_diff_threshold

    transformers = discover_transformers(pipe)
    if not transformers:
        raise ValueError("No QuantFunc transformer found on pipeline.")

    for transformer in transformers:
        kwargs: dict[str, Any] = {}
        if eff_block_cache is not None:
            kwargs["block_cache"] = eff_block_cache
        if step_cache is not None:
            kwargs["step_cache"] = step_cache
        if kwargs:
            transformer.set_cache(**kwargs)

    if _get_hook_state(pipe) is None:
        attach_session_hooks(pipe)
    return pipe


def apply_sol_on_pipe(pipe: Any, sol_config: SolConfig) -> Any:
    """Apply ``SolConfig`` to all QuantFunc transformers on ``pipe``."""
    if not isinstance(sol_config, SolConfig):
        raise TypeError(f"Expected SolConfig instance, got {type(sol_config).__name__}.")

    transformers = discover_transformers(pipe)
    if not transformers:
        raise ValueError("No QuantFunc transformer found on pipeline.")

    for transformer in transformers:
        transformer.set_sol(sol_config)

    if _get_hook_state(pipe) is None:
        attach_session_hooks(pipe)
    return pipe


def enable_offload(pipe: Any, mode: str = "model") -> Any:
    """Enable pipeline CPU offloading and optional low-VRAM end-of-loop unloading."""
    if mode not in _VALID_OFFLOAD_MODES:
        raise ValueError(
            f"Unsupported offload mode {mode!r}; expected one of {_VALID_OFFLOAD_MODES}."
        )

    if _get_hook_state(pipe) is None:
        attach_session_hooks(pipe)
    state = _get_hook_state(pipe)
    if state is not None:
        state.low_vram = mode == "low_vram"

    if mode == "model":
        if callable(getattr(pipe, "enable_model_cpu_offload", None)):
            pipe.enable_model_cpu_offload()
    else:
        if callable(getattr(pipe, "enable_sequential_cpu_offload", None)):
            pipe.enable_sequential_cpu_offload()
        elif callable(getattr(pipe, "enable_model_cpu_offload", None)):
            pipe.enable_model_cpu_offload()

    return pipe
