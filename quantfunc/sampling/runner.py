"""Generic N-stage sampling orchestrator with per-stage overrides and adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from quantfunc.accelerate.hooks import discover_transformers, pipe_session
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.sigmas import build_linear_sigmas
from quantfunc.sampling.upscalers import LatentUpscaler

_MISSING = object()


@dataclass
class StageSpec:
    """Declarative specification for one stage in a multi-stage sampling workflow."""

    name: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    denoise: float = 1.0
    sigmas: Sequence[float] | None = None
    output_type: str = "pil"
    upscaler: LatentUpscaler | None = None
    renoise: Callable[..., Any] | None = None
    latent_extractor: Callable[[Any, StageSpec], LatentBundle] | None = None
    stage_call_adapter: Callable[[Any, StageSpec, LatentBundle | None, dict[str, Any]], Any] | None = None
    transformer_overrides: dict[str, Any] | None = None
    conditioning_callback: Callable[
        [Any, StageSpec, LatentBundle | None, dict[str, Any]], dict[str, Any] | None
    ] | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("StageSpec.name must be a non-empty string.")
        if isinstance(self.denoise, bool) or not isinstance(self.denoise, int | float):
            raise ValueError(f"StageSpec.denoise must be a float in (0, 1], got {self.denoise!r}.")
        denoise_f = float(self.denoise)
        if not math.isfinite(denoise_f) or denoise_f <= 0.0 or denoise_f > 1.0:
            raise ValueError(f"StageSpec.denoise must be finite and in (0, 1], got {self.denoise!r}.")
        self.denoise = denoise_f

        for field_name in ("width", "height", "num_frames", "num_inference_steps"):
            val = getattr(self, field_name)
            if val is not None:
                if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                    raise ValueError(f"StageSpec.{field_name} must be a positive int, got {val!r}.")


@dataclass
class StageResult:
    """Captured output and intermediate bundle from one executed stage."""

    name: str
    output: Any
    bundle: LatentBundle | None = None
    stage_index: int = 0


def _default_extract_bundle(output: Any) -> LatentBundle:
    if isinstance(output, LatentBundle):
        return output
    if isinstance(output, torch.Tensor):
        return LatentBundle(video_or_image=output)
    if isinstance(output, tuple | list) and len(output) >= 1:
        audio = output[1] if len(output) >= 2 else None
        return LatentBundle(video_or_image=output[0], audio=audio)

    audio = getattr(output, "audio", None)
    for attr in ("frames", "images", "latents", "videos", "sample"):
        val = getattr(output, attr, None)
        if val is not None:
            return LatentBundle(video_or_image=val, audio=audio)
    return LatentBundle(video_or_image=output, audio=audio)


def _snapshot_cache(transformer: Any) -> tuple[float | None, float | None]:
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


def _snapshot_loras(transformer: Any) -> list[Any]:
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


def _apply_transformer_overrides(
    pipe: Any,
    overrides: dict[str, Any] | None,
) -> list[tuple[Any, list[tuple[str, Any]]]]:
    if not overrides:
        return []
    transformers = discover_transformers(pipe)
    if not transformers:
        return []

    saved_records: list[tuple[Any, list[tuple[str, Any]]]] = []
    for transformer in transformers:
        saved: list[tuple[str, Any]] = []
        cache_saved = False
        try:
            for key, value in overrides.items():
                if key == "quality_enhance":
                    prev = getattr(transformer, "quality_enhance", None)
                    if prev is None and hasattr(transformer, "_quality_config"):
                        prev = getattr(transformer._quality_config, "quality_enhance", False)
                    saved.append(("quality_enhance", prev))
                    if callable(getattr(transformer, "set_quality_enhance", None)):
                        transformer.set_quality_enhance(bool(value))
                    else:
                        transformer.quality_enhance = bool(value)
                elif key in ("block_cache", "step_cache"):
                    if not cache_saved:
                        saved.append(("__cache__", _snapshot_cache(transformer)))
                        cache_saved = True
                    if callable(getattr(transformer, "set_cache", None)):
                        transformer.set_cache(**{key: value})
                    else:
                        setattr(transformer, key, value)
                elif key == "sol":
                    prev_sol = getattr(transformer, "sol_config", None)
                    if prev_sol is None:
                        prev_sol = getattr(transformer, "_sol_config", None)
                    saved.append(("sol", prev_sol))
                    if callable(getattr(transformer, "set_sol", None)):
                        transformer.set_sol(value)
                    else:
                        transformer.sol_config = value
                elif key == "attention_backend":
                    prev_ab = getattr(transformer, "attention_backend", None)
                    if prev_ab is None and hasattr(transformer, "_engine_config"):
                        prev_ab = getattr(transformer._engine_config, "attention_backend", None)
                    saved.append(("attention_backend", prev_ab))
                    if callable(getattr(transformer, "set_attention_backend", None)):
                        transformer.set_attention_backend(value)
                    else:
                        transformer.attention_backend = value
                elif key == "loras":
                    saved.append(("loras", _snapshot_loras(transformer)))
                    if not value and callable(getattr(transformer, "unload_loras", None)):
                        transformer.unload_loras()
                    elif callable(getattr(transformer, "set_loras", None)):
                        transformer.set_loras(value)
                    else:
                        transformer.loras = list(value)
                elif key == "default_steps":
                    prev_ds = getattr(transformer, "_default_steps", None)
                    saved.append(("default_steps", prev_ds))
                    if callable(getattr(transformer, "set_default_steps", None)):
                        transformer.set_default_steps(value)
                    else:
                        transformer._default_steps = value
                else:
                    prev_attr = getattr(transformer, key, _MISSING)
                    saved.append((f"attr:{key}", prev_attr))
                    setattr(transformer, key, value)
        except BaseException:
            saved_records.append((transformer, saved))
            _restore_transformer_overrides(saved_records)
            raise
        saved_records.append((transformer, saved))
    return saved_records


def _restore_transformer_overrides(
    saved_records: list[tuple[Any, list[tuple[str, Any]]]],
) -> None:
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
                    if callable(getattr(transformer, "set_sol", None)) and prev is not None:
                        transformer.set_sol(prev)
                    else:
                        transformer.sol_config = prev
                elif key == "attention_backend":
                    if callable(getattr(transformer, "set_attention_backend", None)) and prev is not None:
                        transformer.set_attention_backend(prev)
                    else:
                        transformer.attention_backend = prev
                elif key == "loras":
                    if not prev and callable(getattr(transformer, "unload_loras", None)):
                        transformer.unload_loras()
                    elif callable(getattr(transformer, "set_loras", None)):
                        transformer.set_loras(prev)
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


class MultiStageRunner:
    """Execute an ordered sequence of ``StageSpec`` definitions on any pipeline."""

    def __init__(self, stages: Sequence[StageSpec]) -> None:
        stage_tuple = tuple(stages)
        if not stage_tuple:
            raise ValueError("MultiStageRunner requires at least one StageSpec.")
        for item in stage_tuple:
            if not isinstance(item, StageSpec):
                raise TypeError(f"Expected StageSpec, got {type(item).__name__}.")
        self.stages: tuple[StageSpec, ...] = stage_tuple

    def run(
        self,
        pipe: Any,
        *,
        return_all_stages: bool = False,
        initial_bundle: LatentBundle | None = None,
        generator: Any = None,
        **common_kwargs: Any,
    ) -> Any:
        """Run every configured stage in sequence, passing latents forward.

        Each stage runs inside ``with pipe_session(pipe):`` and restores all
        ``transformer_overrides`` in a ``finally`` block even if an exception is
        raised.
        """
        bundle: LatentBundle | None = initial_bundle
        results: list[StageResult] = []
        last_index = len(self.stages) - 1

        for idx, stage in enumerate(self.stages):
            stage_sigmas: list[float] | None
            if stage.sigmas is not None:
                stage_sigmas = [float(s) for s in stage.sigmas]
            elif stage.denoise < 1.0 and stage.num_inference_steps is not None:
                full_steps = max(
                    stage.num_inference_steps,
                    int(round(stage.num_inference_steps / stage.denoise)),
                )
                full_sigmas = build_linear_sigmas(full_steps)
                stage_sigmas = full_sigmas[-(stage.num_inference_steps + 1) :]
            else:
                stage_sigmas = None

            if bundle is not None:
                if stage.upscaler is not None:
                    bundle = stage.upscaler(
                        bundle,
                        target_height=stage.height,
                        target_width=stage.width,
                    )
                if stage.renoise is not None:
                    sigma0 = float(stage_sigmas[0]) if stage_sigmas else float(stage.denoise)
                    bundle = stage.renoise(bundle, sigma0, generator=generator)

            call_kwargs: dict[str, Any] = dict(common_kwargs)
            call_kwargs.update(stage.extra_kwargs)

            if stage.width is not None:
                call_kwargs["width"] = stage.width
            if stage.height is not None:
                call_kwargs["height"] = stage.height
            if stage.num_frames is not None:
                call_kwargs["num_frames"] = stage.num_frames
            if stage.num_inference_steps is not None:
                call_kwargs["num_inference_steps"] = stage.num_inference_steps
            if stage.guidance_scale is not None:
                call_kwargs["guidance_scale"] = stage.guidance_scale
            if stage.output_type is not None:
                call_kwargs["output_type"] = stage.output_type
            if stage_sigmas is not None:
                call_kwargs["sigmas"] = stage_sigmas
            if generator is not None and "generator" not in call_kwargs:
                call_kwargs["generator"] = generator

            if bundle is not None:
                call_kwargs.setdefault("latents", bundle.video_or_image)
                if bundle.audio is not None:
                    call_kwargs.setdefault("audio_latents", bundle.audio)

            if stage.conditioning_callback is not None:
                updated = stage.conditioning_callback(pipe, stage, bundle, call_kwargs)
                if isinstance(updated, dict):
                    call_kwargs = updated

            saved_overrides = _apply_transformer_overrides(pipe, stage.transformer_overrides)
            try:
                with pipe_session(pipe):
                    if stage.stage_call_adapter is not None:
                        output = stage.stage_call_adapter(pipe, stage, bundle, call_kwargs)
                    else:
                        output = pipe(**call_kwargs)
            finally:
                _restore_transformer_overrides(saved_overrides)

            if idx < last_index or stage.output_type == "latent" or stage.latent_extractor is not None:
                if stage.latent_extractor is not None:
                    extracted = stage.latent_extractor(output, stage)
                    if not isinstance(extracted, LatentBundle):
                        raise TypeError(
                            "StageSpec.latent_extractor must return a LatentBundle instance."
                        )
                    bundle = extracted
                else:
                    bundle = _default_extract_bundle(output)
                if stage.width is not None:
                    bundle.meta.setdefault("prev_width", stage.width)
                if stage.height is not None:
                    bundle.meta.setdefault("prev_height", stage.height)

            results.append(
                StageResult(
                    name=stage.name,
                    output=output,
                    bundle=bundle if idx < last_index or stage.output_type == "latent" else None,
                    stage_index=idx,
                )
            )

        if return_all_stages:
            return results
        return results[-1].output


def run_stages(
    pipe: Any,
    stages: Sequence[StageSpec],
    *,
    return_all_stages: bool = False,
    initial_bundle: LatentBundle | None = None,
    generator: Any = None,
    **common_kwargs: Any,
) -> Any:
    """Convenience function to run ``stages`` through ``MultiStageRunner``."""
    return MultiStageRunner(stages).run(
        pipe,
        return_all_stages=return_all_stages,
        initial_bundle=initial_bundle,
        generator=generator,
        **common_kwargs,
    )
