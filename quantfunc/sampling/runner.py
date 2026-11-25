"""Generic N-stage sampling orchestrator with per-stage overrides and adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from quantfunc.accelerate.hooks import pipe_session
from quantfunc.accelerate.serving import (
    apply_runtime_overrides,
    restore_runtime_overrides,
    snapshot_cache,
    snapshot_loras,
)
from quantfunc.core.overrides import RuntimeOverrides
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.sigmas import build_linear_sigmas
from quantfunc.sampling.upscalers import LatentUpscaler


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
    transformer_overrides: RuntimeOverrides | Mapping[str, Any] | None = None
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


_snapshot_cache = snapshot_cache
_snapshot_loras = snapshot_loras


def _apply_transformer_overrides(
    pipe: Any,
    overrides: RuntimeOverrides | Mapping[str, Any] | None,
) -> list[tuple[Any, list[tuple[str, Any]]]]:
    return apply_runtime_overrides(
        pipe,
        overrides,
        sticky_loras=False,
        allow_extra_attrs=True,
        strict_lora_registry=None,
    )


def _restore_transformer_overrides(
    saved_records: list[tuple[Any, list[tuple[str, Any]]]],
) -> None:
    restore_runtime_overrides(saved_records)


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
