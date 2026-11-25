"""Model-agnostic per-request serving lock, runtime override engine, and request scope.

L3a utility shared by ``request_scope`` and ``MultiStageRunner``. Never imports
``quantfunc.sampling`` and contains zero model-specific logic.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from quantfunc.accelerate.hooks import (
    _close_all_sessions,
    discover_transformers,
    pipe_session,
)
from quantfunc.core.lora import normalize_lora_entries
from quantfunc.core.overrides import UNSET, RuntimeOverrides
from quantfunc.models.base import QuantFuncBaseTransformer
from quantfunc.native.errors import EngineBusyError, UnsupportedError

logger = logging.getLogger(__name__)

_MISSING = object()
_SERVING_LOCK_ATTR = "_qf_serving_lock"
_RLOCK_TYPE = type(threading.RLock())
_LOCK_INIT_GUARD = threading.Lock()
_FALLBACK_LOCKS: weakref.WeakKeyDictionary[Any, threading.RLock] = weakref.WeakKeyDictionary()

_KNOWN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "quality_enhance",
        "block_cache",
        "fbcache",
        "step_cache",
        "stepcache",
        "sol",
        "attention_backend",
        "loras",
        "default_steps",
    }
)
_VALID_AV_CONDITION_KEYS: frozenset[str] = frozenset({"keyframes", "refs", "text_token_tags"})


def get_pipeline_lock(pipe: Any) -> threading.RLock:
    """Return the per-pipeline reentrant execution lock without mutating ``pipe.__class__``."""
    pipe_dict = getattr(pipe, "__dict__", None)
    if isinstance(pipe_dict, dict):
        existing = pipe_dict.get(_SERVING_LOCK_ATTR)
        if isinstance(existing, _RLOCK_TYPE):
            return existing
        with _LOCK_INIT_GUARD:
            existing = pipe_dict.get(_SERVING_LOCK_ATTR)
            if isinstance(existing, _RLOCK_TYPE):
                return existing
            lock = threading.RLock()
            pipe_dict[_SERVING_LOCK_ATTR] = lock
            return lock

    with _LOCK_INIT_GUARD:
        existing = _FALLBACK_LOCKS.get(pipe)
        if existing is not None:
            return existing
        lock = threading.RLock()
        _FALLBACK_LOCKS[pipe] = lock
        return lock


def _validate_busy_timeout(busy_timeout: float | None) -> float | None:
    if busy_timeout is None:
        return None
    if isinstance(busy_timeout, bool) or not isinstance(busy_timeout, int | float):
        raise ValueError(f"busy_timeout must be a non-negative finite number or None, got {busy_timeout!r}")
    val = float(busy_timeout)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"busy_timeout must be a non-negative finite number or None, got {busy_timeout!r}")
    return val


def _acquire_lock(lock: Any, timeout: float | None) -> bool:
    if timeout is None:
        return bool(lock.acquire(blocking=True))
    if timeout <= 0.0:
        return bool(lock.acquire(blocking=False))
    return bool(lock.acquire(blocking=True, timeout=timeout))


def _validate_av_conditions(av_conditions: Any) -> dict[str, Any] | None:
    if av_conditions is UNSET or av_conditions is None:
        return None
    if not isinstance(av_conditions, Mapping):
        raise ValueError(f"av_conditions must be a mapping or None, got {type(av_conditions).__name__}")
    unknown = sorted(k for k in av_conditions.keys() if k not in _VALID_AV_CONDITION_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown av_conditions key(s): {unknown}; expected subset of {sorted(_VALID_AV_CONDITION_KEYS)}"
        )
    return dict(av_conditions)


def snapshot_cache(transformer: Any) -> tuple[float | None, float | None]:
    """Capture current ``(step_cache, block_cache)`` thresholds from ``transformer``."""
    cache_cfg = getattr(transformer, "_cache_config", None) or getattr(transformer, "cache_config", None)
    if cache_cfg is not None:
        return (
            getattr(cache_cfg, "step_cache", None),
            getattr(cache_cfg, "block_cache", None),
        )
    return (
        getattr(transformer, "step_cache", None),
        getattr(transformer, "block_cache", None),
    )


def snapshot_loras(transformer: Any) -> list[Any]:
    """Capture current active LoRA stack from ``transformer``."""
    lora_mgr = getattr(transformer, "_lora_manager", None)
    if lora_mgr is not None and hasattr(lora_mgr, "active_loras"):
        return list(lora_mgr.active_loras)
    if hasattr(transformer, "active_loras"):
        return list(transformer.active_loras)
    if hasattr(transformer, "_loras"):
        return list(transformer._loras)
    if hasattr(transformer, "loras"):
        return list(transformer.loras)
    return []


def _callable_accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _required_lora_policy_keys(
    *,
    lora_registry: Mapping[str, str | os.PathLike[str]] | None,
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None,
    strict_lora_registry: bool | None,
) -> tuple[str, ...]:
    keys: list[str] = []
    if lora_registry is not None:
        keys.append("lora_registry")
    if allowed_lora_roots is not None:
        keys.append("allowed_lora_roots")
    if strict_lora_registry is True:
        keys.append("strict_registry")
    return tuple(keys)


def _ensure_lora_policy_supported(transformer: Any, required_keys: Sequence[str]) -> None:
    if not required_keys or isinstance(transformer, QuantFuncBaseTransformer):
        return
    set_fn = getattr(transformer, "set_loras", None)
    if not callable(set_fn):
        raise UnsupportedError(
            f"Transformer {type(transformer).__name__} does not implement set_loras() "
            f"to enforce LoRA security policy ({', '.join(required_keys)})."
        )
    for key in required_keys:
        if not _callable_accepts_kwarg(set_fn, key):
            raise UnsupportedError(
                f"Transformer {type(transformer).__name__}.set_loras() does not accept "
                f"required LoRA security policy parameter {key!r}."
            )


def _apply_loras_to_transformer(
    transformer: Any,
    entries: Sequence[Any],
    *,
    lora_registry: Mapping[str, str | os.PathLike[str]] | None,
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None,
    strict_lora_registry: bool | None,
    force_loras: bool,
) -> None:
    is_qf_base = isinstance(transformer, QuantFuncBaseTransformer)
    required_keys = _required_lora_policy_keys(
        lora_registry=lora_registry,
        allowed_lora_roots=allowed_lora_roots,
        strict_lora_registry=strict_lora_registry,
    )
    _ensure_lora_policy_supported(transformer, required_keys)

    if not entries:
        unload_fn = getattr(transformer, "unload_loras", None)
        if callable(unload_fn):
            if force_loras and (is_qf_base or _callable_accepts_kwarg(unload_fn, "force")):
                unload_fn(force=True)
            else:
                unload_fn()
            return
        set_fn = getattr(transformer, "set_loras", None)
        if callable(set_fn):
            if force_loras and (is_qf_base or _callable_accepts_kwarg(set_fn, "force")):
                set_fn([], force=True)
            else:
                set_fn([])
            return
        transformer.loras = []
        return

    set_fn = getattr(transformer, "set_loras", None)
    if not callable(set_fn):
        transformer.loras = list(entries)
        return

    policy_kwargs: dict[str, Any] = {}
    if lora_registry is not None:
        policy_kwargs["lora_registry"] = lora_registry
    if allowed_lora_roots is not None:
        policy_kwargs["allowed_lora_roots"] = allowed_lora_roots
    if strict_lora_registry is not None:
        policy_kwargs["strict_registry"] = bool(strict_lora_registry)

    if is_qf_base:
        if force_loras:
            policy_kwargs["force"] = True
        set_fn(entries, **policy_kwargs)
        return

    call_kwargs: dict[str, Any] = {}
    for k, v in policy_kwargs.items():
        if _callable_accepts_kwarg(set_fn, k):
            call_kwargs[k] = v
    if force_loras and _callable_accepts_kwarg(set_fn, "force"):
        call_kwargs["force"] = True
    set_fn(entries, **call_kwargs)


def apply_runtime_overrides(
    pipe: Any,
    overrides: RuntimeOverrides | Mapping[str, Any] | None,
    *,
    sticky_loras: bool = False,
    default_loras: Sequence[Any] | Any = UNSET,
    lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
    strict_lora_registry: bool | None = None,
    allow_extra_attrs: bool = False,
    force_loras: bool = False,
) -> list[tuple[Any, list[tuple[str, Any]]]]:
    """Apply validated runtime overrides to all QuantFunc transformers on ``pipe``.

    Returns opaque restoration records for ``restore_runtime_overrides``. If any
    step raises mid-apply, all previously mutated dials are rolled back cleanly.
    """
    if strict_lora_registry is not None and not isinstance(strict_lora_registry, bool):
        raise ValueError(f"strict_lora_registry must be bool or None, got {strict_lora_registry!r}")

    raw_loras: Any = UNSET
    extra_attrs: dict[str, Any] = {}
    if overrides is None:
        parsed = RuntimeOverrides()
    elif isinstance(overrides, RuntimeOverrides):
        parsed = overrides
    elif isinstance(overrides, Mapping):
        if allow_extra_attrs:
            known_dict: dict[str, Any] = {}
            for key, value in overrides.items():
                if not isinstance(key, str):
                    raise ValueError(f"Override keys must be strings, got {key!r}")
                if key in _KNOWN_OVERRIDE_KEYS:
                    known_dict[key] = value
                else:
                    extra_attrs[key] = value
            parsed = RuntimeOverrides.from_dict(known_dict) if known_dict else RuntimeOverrides()
            if "loras" in known_dict and known_dict["loras"] is not UNSET:
                raw_loras = list(known_dict["loras"])
        else:
            parsed = RuntimeOverrides.from_dict(overrides)
    else:
        parsed = RuntimeOverrides.from_any(overrides)

    norm_default_loras: Any = UNSET
    if default_loras is not UNSET:
        if default_loras is None:
            raise ValueError(
                "default_loras cannot be None; pass () or [] to clear LoRAs, or UNSET to keep active LoRAs."
            )
        norm_default_loras = tuple(normalize_lora_entries(default_loras))

    effective_loras: Any = UNSET
    use_raw_loras = False
    if parsed.loras is not UNSET:
        effective_loras = parsed.loras
        use_raw_loras = raw_loras is not UNSET
    elif norm_default_loras is not UNSET:
        effective_loras = norm_default_loras

    required_policy_keys = _required_lora_policy_keys(
        lora_registry=lora_registry,
        allowed_lora_roots=allowed_lora_roots,
        strict_lora_registry=strict_lora_registry,
    )

    if parsed.is_empty() and effective_loras is UNSET and not extra_attrs and not required_policy_keys:
        return []

    transformers = discover_transformers(pipe)
    if not transformers:
        return []

    # Fail-closed check before mutating any transformer if explicit policy is unsupported.
    if required_policy_keys:
        for transformer in transformers:
            _ensure_lora_policy_supported(transformer, required_policy_keys)

    if parsed.is_empty() and effective_loras is UNSET and not extra_attrs:
        return []

    saved_records: list[tuple[Any, list[tuple[str, Any]]]] = []
    rollback_records: list[tuple[Any, list[tuple[str, Any]]]] = []

    for transformer in transformers:
        saved: list[tuple[str, Any]] = []
        rollback_saved: list[tuple[str, Any]] = []
        try:
            if parsed.quality_enhance is not UNSET:
                prev_qe = getattr(transformer, "quality_enhance", None)
                if prev_qe is None and hasattr(transformer, "_quality_config"):
                    prev_qe = getattr(transformer._quality_config, "quality_enhance", False)
                rec = ("quality_enhance", bool(prev_qe) if prev_qe is not None else False)
                saved.append(rec)
                rollback_saved.append(rec)
                if callable(getattr(transformer, "set_quality_enhance", None)):
                    transformer.set_quality_enhance(bool(parsed.quality_enhance))
                else:
                    transformer.quality_enhance = bool(parsed.quality_enhance)

            if parsed.block_cache is not UNSET or parsed.step_cache is not UNSET:
                rec = ("__cache__", snapshot_cache(transformer))
                saved.append(rec)
                rollback_saved.append(rec)
                cache_kwargs: dict[str, float] = {}
                if parsed.block_cache is not UNSET:
                    cache_kwargs["block_cache"] = parsed.block_cache
                if parsed.step_cache is not UNSET:
                    cache_kwargs["step_cache"] = parsed.step_cache
                if callable(getattr(transformer, "set_cache", None)):
                    transformer.set_cache(**cache_kwargs)
                else:
                    for ck, cv in cache_kwargs.items():
                        setattr(transformer, ck, cv)

            if parsed.sol is not UNSET:
                prev_sol = getattr(transformer, "sol_config", _MISSING)
                if prev_sol is _MISSING:
                    prev_sol = getattr(transformer, "_sol_config", None)
                rec = ("sol", prev_sol)
                saved.append(rec)
                rollback_saved.append(rec)
                if callable(getattr(transformer, "set_sol", None)):
                    transformer.set_sol(parsed.sol)
                else:
                    transformer.sol_config = parsed.sol

            if parsed.attention_backend is not UNSET:
                prev_ab = getattr(transformer, "attention_backend", None)
                if prev_ab is None and hasattr(transformer, "_engine_config"):
                    prev_ab = getattr(transformer._engine_config, "attention_backend", None)
                rec = ("attention_backend", prev_ab)
                saved.append(rec)
                rollback_saved.append(rec)
                if callable(getattr(transformer, "set_attention_backend", None)):
                    transformer.set_attention_backend(parsed.attention_backend)
                else:
                    transformer.attention_backend = parsed.attention_backend

            if parsed.default_steps is not UNSET:
                prev_ds = getattr(transformer, "_default_steps", None)
                rec = ("default_steps", prev_ds)
                saved.append(rec)
                rollback_saved.append(rec)
                if callable(getattr(transformer, "set_default_steps", None)):
                    transformer.set_default_steps(parsed.default_steps)
                else:
                    transformer._default_steps = parsed.default_steps

            if effective_loras is not UNSET:
                prev_loras = snapshot_loras(transformer)
                lora_entries = (
                    raw_loras
                    if (
                        allow_extra_attrs
                        and use_raw_loras
                        and not isinstance(transformer, QuantFuncBaseTransformer)
                    )
                    else effective_loras
                )
                _apply_loras_to_transformer(
                    transformer,
                    lora_entries,
                    lora_registry=lora_registry,
                    allowed_lora_roots=allowed_lora_roots,
                    strict_lora_registry=strict_lora_registry,
                    force_loras=bool(force_loras),
                )
                lora_rec = ("loras", prev_loras)
                rollback_saved.append(lora_rec)
                if not sticky_loras:
                    saved.append(lora_rec)

            for attr_key, attr_val in extra_attrs.items():
                prev_attr = getattr(transformer, attr_key, _MISSING)
                rec = (f"attr:{attr_key}", prev_attr)
                saved.append(rec)
                rollback_saved.append(rec)
                setattr(transformer, attr_key, attr_val)
        except BaseException:
            rollback_records.append((transformer, rollback_saved))
            restore_runtime_overrides(rollback_records)
            raise

        saved_records.append((transformer, saved))
        rollback_records.append((transformer, rollback_saved))

    return saved_records


def restore_runtime_overrides(
    saved_records: list[tuple[Any, list[tuple[str, Any]]]],
) -> None:
    """Restore runtime overrides in reverse order, preserving the first restoration error."""
    first_error: BaseException | None = None
    for transformer, saved in reversed(saved_records):
        for key, prev in reversed(saved):
            try:
                if key == "quality_enhance":
                    if callable(getattr(transformer, "set_quality_enhance", None)):
                        transformer.set_quality_enhance(bool(prev))
                    else:
                        transformer.quality_enhance = bool(prev)
                elif key == "__cache__":
                    prev_step, prev_block = prev
                    if callable(getattr(transformer, "set_cache", None)):
                        kwargs: dict[str, Any] = {}
                        if prev_step is not None:
                            kwargs["step_cache"] = prev_step
                        if prev_block is not None:
                            kwargs["block_cache"] = prev_block
                        if kwargs:
                            transformer.set_cache(**kwargs)
                    else:
                        if prev_step is not None:
                            transformer.step_cache = prev_step
                        if prev_block is not None:
                            transformer.block_cache = prev_block
                elif key == "sol":
                    if callable(getattr(transformer, "set_sol", None)):
                        transformer.set_sol(prev)
                    else:
                        transformer.sol_config = prev
                elif key == "attention_backend":
                    if callable(getattr(transformer, "set_attention_backend", None)) and prev is not None:
                        transformer.set_attention_backend(prev)
                    else:
                        transformer.attention_backend = prev
                elif key == "loras":
                    if not prev:
                        unload_fn = getattr(transformer, "unload_loras", None)
                        if callable(unload_fn):
                            unload_fn()
                        elif callable(getattr(transformer, "set_loras", None)):
                            transformer.set_loras([])
                        else:
                            transformer.loras = []
                    else:
                        set_fn = getattr(transformer, "set_loras", None)
                        if callable(set_fn):
                            set_fn(prev)
                        else:
                            transformer.loras = list(prev)
                elif key == "default_steps":
                    if callable(getattr(transformer, "set_default_steps", None)) and prev is not None:
                        transformer.set_default_steps(prev)
                    else:
                        transformer._default_steps = prev
                elif key.startswith("attr:"):
                    attr_name = key[5:]
                    if prev is _MISSING:
                        if hasattr(transformer, "__dict__"):
                            transformer.__dict__.pop(attr_name, None)
                    else:
                        setattr(transformer, attr_name, prev)
            except BaseException as exc:  # pragma: no cover - preserve first restore error
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error


def _apply_ref_audio(transformers: Sequence[Any], ref_audio: Any) -> None:
    targets = [t for t in transformers if callable(getattr(t, "set_reference_audio", None))]
    if not targets:
        if ref_audio is not None:
            raise UnsupportedError("Pipeline transformer does not support ref_audio conditioning.")
        return
    for transformer in targets:
        transformer.set_reference_audio(ref_audio)


def _apply_av_conditions(transformers: Sequence[Any], av_conditions: dict[str, Any] | None) -> None:
    targets = [t for t in transformers if callable(getattr(t, "set_av_conditions", None))]
    if not targets:
        if av_conditions is not None:
            raise UnsupportedError("Pipeline transformer does not support av_conditions conditioning.")
        return
    for transformer in targets:
        if av_conditions is None:
            clear_fn = getattr(transformer, "clear_av_conditions", None)
            if callable(clear_fn):
                clear_fn()
            else:
                transformer.set_av_conditions(keyframes=None, refs=None, text_token_tags=None)
        else:
            transformer.set_av_conditions(
                keyframes=av_conditions.get("keyframes"),
                refs=av_conditions.get("refs"),
                text_token_tags=av_conditions.get("text_token_tags"),
            )


def _reset_transformer_request_state(transformer: Any) -> None:
    reset_fn = getattr(transformer, "reset_request_state", None)
    if callable(reset_fn):
        reset_fn()
        return
    set_ref = getattr(transformer, "set_reference_audio", None)
    if callable(set_ref):
        set_ref(None)
    clear_av = getattr(transformer, "clear_av_conditions", None)
    if callable(clear_av):
        clear_av()


@contextlib.contextmanager
def request_scope(
    pipe: Any,
    overrides: RuntimeOverrides | Mapping[str, Any] | None = None,
    *,
    lock: Any | None = None,
    busy_timeout: float | None = None,
    sticky_loras: bool = True,
    default_loras: Sequence[Any] | Any = UNSET,
    lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
    strict_lora_registry: bool | None = None,
    ref_audio: Any = UNSET,
    av_conditions: Mapping[str, Any] | None | Any = UNSET,
    clear_conditions_on_exit: bool = True,
    force_loras: bool = False,
) -> Iterator[Any]:
    """Serialize one online request on ``pipe`` with validated overrides and guaranteed cleanup.

    Pure-CPU input validation happens strictly BEFORE acquiring the pipeline lock.
    On exit (normal or exceptional), active sessions are closed, per-request
    conditions/schedules are reset, non-sticky overrides are restored, and the lock
    is unconditionally released.
    """
    # 1. Pure-CPU validation strictly before acquiring lock
    eff_timeout = _validate_busy_timeout(busy_timeout)
    if strict_lora_registry is not None and not isinstance(strict_lora_registry, bool):
        raise ValueError(f"strict_lora_registry must be bool or None, got {strict_lora_registry!r}")
    parsed_overrides = RuntimeOverrides.from_any(overrides)

    norm_default_loras: Any = UNSET
    if default_loras is not UNSET:
        if default_loras is None:
            raise ValueError(
                "default_loras cannot be None; pass () or [] to clear LoRAs, or UNSET to keep active LoRAs."
            )
        norm_default_loras = tuple(normalize_lora_entries(default_loras))

    av_cond_dict = _validate_av_conditions(av_conditions)

    # 2. Acquire per-pipeline reentrant execution lock
    eff_lock = lock if lock is not None else get_pipeline_lock(pipe)
    if not _acquire_lock(eff_lock, eff_timeout):
        raise EngineBusyError(
            f"Pipeline is busy; failed to acquire execution lock within busy_timeout={eff_timeout!r}s."
        )

    saved: list[tuple[Any, list[tuple[str, Any]]]] = []
    body_exc: BaseException | None = None
    try:
        try:
            _close_all_sessions(pipe, unload_low_vram=False)
            saved = apply_runtime_overrides(
                pipe,
                parsed_overrides,
                sticky_loras=bool(sticky_loras),
                default_loras=norm_default_loras,
                lora_registry=lora_registry,
                allowed_lora_roots=allowed_lora_roots,
                strict_lora_registry=strict_lora_registry,
                allow_extra_attrs=False,
                force_loras=bool(force_loras),
            )

            transformers = discover_transformers(pipe)
            if ref_audio is not UNSET:
                _apply_ref_audio(transformers, ref_audio)
            if av_conditions is not UNSET:
                _apply_av_conditions(transformers, av_cond_dict)

            with pipe_session(pipe):
                yield pipe
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            # (i) Guarantee all active sessions are closed
            try:
                _close_all_sessions(pipe, unload_low_vram=False)
            except BaseException as exc:
                if body_exc is not None:
                    logger.warning("Secondary error closing sessions during request_scope exit: %s", exc)
                elif cleanup_error is None:
                    cleanup_error = exc

            # (ii) Clear per-request conditions & staged schedules
            if clear_conditions_on_exit:
                for transformer in discover_transformers(pipe):
                    try:
                        _reset_transformer_request_state(transformer)
                    except BaseException as exc:
                        if body_exc is not None:
                            logger.warning(
                                "Secondary error resetting transformer request state during exit: %s",
                                exc,
                            )
                        elif cleanup_error is None:
                            cleanup_error = exc

            # (iii) Restore runtime dials (and LoRAs when sticky_loras=False)
            if saved:
                try:
                    restore_runtime_overrides(saved)
                except BaseException as exc:
                    if body_exc is not None:
                        logger.warning(
                            "Secondary error restoring runtime overrides during request_scope exit: %s",
                            exc,
                        )
                    elif cleanup_error is None:
                        cleanup_error = exc

            if body_exc is None and cleanup_error is not None:
                raise cleanup_error
    finally:
        # (iv) Unconditionally release execution lock
        eff_lock.release()


__all__ = [
    "get_pipeline_lock",
    "snapshot_cache",
    "snapshot_loras",
    "apply_runtime_overrides",
    "restore_runtime_overrides",
    "request_scope",
]
