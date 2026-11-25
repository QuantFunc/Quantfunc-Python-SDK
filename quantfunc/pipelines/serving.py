"""Model-agnostic thread-safe serving pipeline wrapper for online per-request switching.

L4a facade composing L3a (``request_scope``, ``attach_session_hooks``) and L3b
(``MultiStageRunner``, ``StageSpec``) with zero model-specific assumptions.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from quantfunc.accelerate.hooks import (
    _get_hook_state,
    attach_session_hooks,
    detach_session_hooks,
    discover_transformers,
)
from quantfunc.accelerate.serving import (
    _acquire_lock,
    _validate_av_conditions,
    _validate_busy_timeout,
    get_pipeline_lock,
    request_scope,
    snapshot_cache,
    snapshot_loras,
)
from quantfunc.core.lora import LoRAEntry, normalize_lora_entries
from quantfunc.core.overrides import UNSET, RuntimeOverrides
from quantfunc.native.errors import EngineBusyError, UnsupportedError
from quantfunc.sampling.runner import MultiStageRunner, StageSpec


def _validate_lora_registry(
    lora_registry: Mapping[str, str | os.PathLike[str]] | None,
) -> dict[str, str] | None:
    if lora_registry is None:
        return None
    if not isinstance(lora_registry, Mapping):
        raise ValueError(f"lora_registry must be a mapping or None, got {type(lora_registry).__name__}")
    out: dict[str, str] = {}
    for alias, raw_target in lora_registry.items():
        if not isinstance(alias, str) or not alias.strip() or "\x00" in alias:
            raise ValueError(f"lora_registry alias must be a non-empty string without NUL, got {alias!r}")
        if not isinstance(raw_target, str | os.PathLike):
            raise ValueError(
                f"lora_registry[{alias!r}] must be a path string or PathLike, got {type(raw_target).__name__}"
            )
        target = os.fspath(raw_target)
        if not target.strip() or "\x00" in target:
            raise ValueError(f"lora_registry[{alias!r}] target path must be non-empty without NUL")
        out[alias] = target
    return out


def _validate_allowed_lora_roots(
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None,
) -> tuple[str, ...] | None:
    if allowed_lora_roots is None:
        return None
    if isinstance(allowed_lora_roots, str | bytes) or not isinstance(allowed_lora_roots, Sequence):
        raise ValueError("allowed_lora_roots must be a sequence of directory paths, not a bare string.")
    roots: list[str] = []
    for item in allowed_lora_roots:
        if not isinstance(item, str | os.PathLike):
            raise ValueError(
                f"allowed_lora_roots entries must be path strings or PathLike, got {type(item).__name__}"
            )
        root_str = os.fspath(item)
        if not root_str.strip() or "\x00" in root_str:
            raise ValueError("allowed_lora_roots entries must be non-empty paths without NUL")
        roots.append(root_str)
    return tuple(roots)


def _format_safe_lora_entry(
    entry: Any,
    lora_registry: Mapping[str, str] | None,
    realpath_to_alias: Mapping[str, str],
) -> dict[str, Any]:
    if isinstance(entry, LoRAEntry):
        raw_path, scale = entry.path, float(entry.scale)
    elif isinstance(entry, tuple | list) and len(entry) == 2:
        raw_path, scale = str(entry[0]), float(entry[1])
    elif isinstance(entry, Mapping):
        raw_path = str(entry.get("name") or entry.get("path") or "")
        scale = float(entry.get("scale", 1.0))
    else:
        raw_path, scale = str(entry), 1.0

    if lora_registry is not None and raw_path in lora_registry:
        safe_name = raw_path
    else:
        resolved = os.path.realpath(os.path.expanduser(raw_path)) if raw_path else ""
        if resolved and resolved in realpath_to_alias:
            safe_name = realpath_to_alias[resolved]
        else:
            safe_name = os.path.basename(raw_path)
    return {"name": safe_name, "scale": scale}


class ServingPipeline:
    """Thread-safe, model-agnostic serving wrapper for a single warm diffusion pipeline.

    Supports online per-request switching of prompts, multimodal conditioning,
    runtime acceleration dials (Sol, block/step cache, quality enhance, attention
    backend, LoRA stacks), and single-stage vs multi-stage execution without
    reloading base model weights.

    Deployment model: run one worker process per GPU with a single warm
    ``ServingPipeline`` instance (handling concurrent threads or ``asyncio``
    tasks via the built-in reentrant execution lock). Scale across multiple GPUs
    horizontally with one worker process per ``CUDA_VISIBLE_DEVICES`` device.
    """

    def __init__(
        self,
        pipe: Any,
        *,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_lora_registry: bool | None = None,
        default_loras: Sequence[Any] | None = None,
        sticky_loras: bool = True,
        busy_timeout: float | None = None,
        attach_hooks: bool = True,
        primary_scheduler: Any | None = None,
        lock: Any | None = None,
    ) -> None:
        transformers = discover_transformers(pipe)
        if not transformers:
            raise ValueError("ServingPipeline requires a pipeline with at least one QuantFunc transformer.")

        self._busy_timeout: float | None = _validate_busy_timeout(busy_timeout)
        if strict_lora_registry is not None and not isinstance(strict_lora_registry, bool):
            raise ValueError(f"strict_lora_registry must be bool or None, got {strict_lora_registry!r}")

        self._lora_registry: dict[str, str] | None = _validate_lora_registry(lora_registry)
        self._allowed_lora_roots: tuple[str, ...] | None = _validate_allowed_lora_roots(allowed_lora_roots)

        # Default to strict_lora_registry=True whenever lora_registry is configured
        # without an explicit allowed_lora_roots sandbox (prevents arbitrary host file reads).
        self._explicit_policy_provided: bool = (
            self._lora_registry is not None
            or self._allowed_lora_roots is not None
            or strict_lora_registry is not None
        )
        if strict_lora_registry is None:
            self._strict_lora_registry: bool | None = (
                True
                if (self._lora_registry is not None and self._allowed_lora_roots is None)
                else (False if self._explicit_policy_provided else None)
            )
        else:
            self._strict_lora_registry = bool(strict_lora_registry)

        self._realpath_to_alias: dict[str, str] = {}
        if self._lora_registry is not None:
            for alias, target in self._lora_registry.items():
                self._realpath_to_alias[os.path.realpath(os.path.expanduser(target))] = alias

        self._default_loras: tuple[LoRAEntry, ...] = (
            () if default_loras is None else tuple(normalize_lora_entries(default_loras))
        )

        # Push LoRA security policy down to every transformer so StageSpec overrides
        # inherit the exact same containment and registry rules automatically.
        has_security_policy = (
            self._lora_registry is not None
            or self._allowed_lora_roots is not None
            or self._strict_lora_registry is True
        )
        if self._explicit_policy_provided:
            for transformer in transformers:
                cfg_fn = getattr(transformer, "configure_lora_policy", None)
                if callable(cfg_fn):
                    cfg_fn(
                        lora_registry=self._lora_registry,
                        allowed_lora_roots=self._allowed_lora_roots,
                        strict_registry=bool(self._strict_lora_registry),
                    )
                elif has_security_policy:
                    raise UnsupportedError(
                        f"Transformer {type(transformer).__name__} does not implement configure_lora_policy() "
                        "to enforce ServingPipeline LoRA security policy."
                    )

        self._pipe = pipe
        self._lock = lock if lock is not None else get_pipeline_lock(pipe)
        self._sticky_loras: bool = bool(sticky_loras)
        self._closed: bool = False
        self._attached_hooks: bool = False

        if attach_hooks:
            had_hooks = _get_hook_state(pipe) is not None
            if not had_hooks or primary_scheduler is not None:
                attach_session_hooks(pipe, primary_scheduler=primary_scheduler)
                self._attached_hooks = True

        self._stats_lock = threading.Lock()
        self._in_flight: int = 0
        self._total_requests: int = 0
        self._failed_requests: int = 0
        self._cached_vram_bytes: dict[int, int] = {}
        self._refresh_cached_vram_locked()

    @property
    def pipe(self) -> Any:
        return self._pipe

    @property
    def lock(self) -> Any:
        return self._lock

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def default_loras(self) -> tuple[LoRAEntry, ...]:
        return self._default_loras

    @property
    def sticky_loras(self) -> bool:
        return self._sticky_loras

    @property
    def busy_timeout(self) -> float | None:
        return self._busy_timeout

    def _refresh_cached_vram_locked(self) -> None:
        snapshot: dict[int, int] = {}
        for transformer in discover_transformers(self._pipe):
            fn = getattr(transformer, "resident_vram_bytes", None)
            if callable(fn):
                try:
                    snapshot[id(transformer)] = int(fn())
                except Exception:
                    snapshot[id(transformer)] = self._cached_vram_bytes.get(id(transformer), 0)
            else:
                snapshot[id(transformer)] = 0
        with self._stats_lock:
            self._cached_vram_bytes.update(snapshot)

    def generate(
        self,
        prompt: Any = UNSET,
        *,
        overrides: RuntimeOverrides | Mapping[str, Any] | None = None,
        stages: MultiStageRunner | Sequence[StageSpec] | None = None,
        return_all_stages: bool = False,
        ref_audio: Any = UNSET,
        av_conditions: Mapping[str, Any] | None | Any = UNSET,
        busy_timeout: float | None | Any = UNSET,
        sticky_loras: bool | None = None,
        clear_conditions_on_exit: bool = True,
        **pipe_kwargs: Any,
    ) -> Any:
        """Execute one serialized single-stage or multi-stage request on the warm pipeline."""
        # 1. Fast pure-CPU validation strictly before acquiring execution lock
        if self._closed:
            raise RuntimeError("ServingPipeline is closed")

        eff_timeout = (
            self._busy_timeout if busy_timeout is UNSET else _validate_busy_timeout(busy_timeout)
        )
        eff_sticky = self._sticky_loras if sticky_loras is None else bool(sticky_loras)

        runner: MultiStageRunner | None
        if stages is None:
            runner = None
        elif isinstance(stages, MultiStageRunner):
            runner = stages
        elif isinstance(stages, Sequence) and not isinstance(stages, str | bytes):
            runner = MultiStageRunner(stages)
        else:
            raise TypeError(
                f"stages must be a MultiStageRunner, Sequence[StageSpec], or None, got {type(stages).__name__}"
            )

        parsed_overrides = RuntimeOverrides.from_any(overrides)
        _validate_av_conditions(av_conditions)

        call_kwargs: dict[str, Any] = dict(pipe_kwargs)
        if prompt is not UNSET:
            if "prompt" in call_kwargs:
                raise TypeError("ServingPipeline.generate() got multiple values for argument 'prompt'")
            call_kwargs["prompt"] = prompt

        # 2. Acquire reentrant lock and re-check _closed BEFORE entering request_scope
        if not _acquire_lock(self._lock, eff_timeout):
            raise EngineBusyError(
                f"ServingPipeline is busy; failed to acquire execution lock within busy_timeout={eff_timeout!r}s."
            )
        try:
            if self._closed:
                raise RuntimeError("ServingPipeline is closed")

            with self._stats_lock:
                self._in_flight += 1
                self._total_requests += 1

            try:
                with request_scope(
                    self._pipe,
                    parsed_overrides,
                    lock=self._lock,
                    busy_timeout=eff_timeout,
                    sticky_loras=eff_sticky,
                    default_loras=self._default_loras,
                    lora_registry=self._lora_registry,
                    allowed_lora_roots=self._allowed_lora_roots,
                    strict_lora_registry=self._strict_lora_registry,
                    ref_audio=ref_audio,
                    av_conditions=av_conditions,
                    clear_conditions_on_exit=bool(clear_conditions_on_exit),
                ):
                    if runner is None:
                        result = self._pipe(**call_kwargs)
                    else:
                        result = runner.run(
                            self._pipe,
                            return_all_stages=bool(return_all_stages),
                            **call_kwargs,
                        )
                    self._refresh_cached_vram_locked()
                    return result
            except BaseException:
                with self._stats_lock:
                    self._failed_requests += 1
                raise
            finally:
                with self._stats_lock:
                    self._in_flight = max(0, self._in_flight - 1)
        finally:
            self._lock.release()

    async def agenerate(
        self,
        prompt: Any = UNSET,
        *,
        overrides: RuntimeOverrides | Mapping[str, Any] | None = None,
        stages: MultiStageRunner | Sequence[StageSpec] | None = None,
        return_all_stages: bool = False,
        ref_audio: Any = UNSET,
        av_conditions: Mapping[str, Any] | None | Any = UNSET,
        busy_timeout: float | None | Any = UNSET,
        sticky_loras: bool | None = None,
        clear_conditions_on_exit: bool = True,
        **pipe_kwargs: Any,
    ) -> Any:
        """Async counterpart to ``generate()``, offloaded to a worker thread."""
        return await asyncio.to_thread(
            self.generate,
            prompt,
            overrides=overrides,
            stages=stages,
            return_all_stages=return_all_stages,
            ref_audio=ref_audio,
            av_conditions=av_conditions,
            busy_timeout=busy_timeout,
            sticky_loras=sticky_loras,
            clear_conditions_on_exit=clear_conditions_on_exit,
            **pipe_kwargs,
        )

    def status(self) -> dict[str, Any]:
        """Return a non-blocking, JSON-safe serving snapshot without leaking host paths."""
        lock_acquired = bool(self._lock.acquire(blocking=False))
        try:
            with self._stats_lock:
                in_flight = self._in_flight
                total_requests = self._total_requests
                failed_requests = self._failed_requests
            closed = self._closed
            busy = (not lock_acquired) or (in_flight > 0)

            if lock_acquired and not closed and in_flight == 0:
                self._refresh_cached_vram_locked()

            with self._stats_lock:
                cached_vram = dict(self._cached_vram_bytes)

            transformer_infos: list[dict[str, Any]] = []
            for transformer in discover_transformers(self._pipe):
                step_c, block_c = snapshot_cache(transformer)
                sol_cfg = getattr(transformer, "sol_config", None)
                if sol_cfg is None:
                    sol_cfg = getattr(transformer, "_sol_config", None)
                raw_loras = snapshot_loras(transformer)
                safe_loras = [
                    _format_safe_lora_entry(entry, self._lora_registry, self._realpath_to_alias)
                    for entry in raw_loras
                ]
                transformer_infos.append(
                    {
                        "type": type(transformer).__name__,
                        "model_arch": str(getattr(transformer, "model_arch", "") or ""),
                        "resident_vram_bytes": 0 if closed else int(cached_vram.get(id(transformer), 0)),
                        "quality_enhance": bool(getattr(transformer, "quality_enhance", False)),
                        "block_cache": float(block_c) if block_c is not None else 0.0,
                        "step_cache": float(step_c) if step_c is not None else 0.0,
                        "attention_backend": str(getattr(transformer, "attention_backend", "auto") or "auto"),
                        "sol_enabled": bool(sol_cfg is not None and getattr(sol_cfg, "enabled", False)),
                        "active_loras": safe_loras,
                    }
                )

            return {
                "ready": bool(not closed),
                "closed": bool(closed),
                "busy": bool(busy),
                "in_flight": int(in_flight),
                "total_requests": int(total_requests),
                "failed_requests": int(failed_requests),
                "sticky_loras": bool(self._sticky_loras),
                "busy_timeout": self._busy_timeout,
                "lora_aliases": sorted(self._lora_registry.keys()) if self._lora_registry is not None else [],
                "transformers": transformer_infos,
            }
        finally:
            if lock_acquired:
                self._lock.release()

    def close(self) -> None:
        """Wait for in-flight work, detach hooks if attached, reset state, and close transformers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            first_error: BaseException | None = None
            if self._attached_hooks:
                try:
                    detach_session_hooks(self._pipe)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                self._attached_hooks = False

            for transformer in discover_transformers(self._pipe):
                reset_fn = getattr(transformer, "reset_request_state", None)
                if callable(reset_fn):
                    try:
                        reset_fn()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                close_fn = getattr(transformer, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc

            with self._stats_lock:
                self._cached_vram_bytes.clear()

            if first_error is not None:
                raise first_error

    def __enter__(self) -> ServingPipeline:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["ServingPipeline"]
