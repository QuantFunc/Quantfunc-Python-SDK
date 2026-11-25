"""MiniMax-H3 AV pipeline loader, conditioning reuse, and two-stage helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel
from quantfunc.pipelines._common import (
    align_size,
    finalize_pipeline_load,
    import_diffusers_class,
)
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.runner import StageSpec
from quantfunc.sampling.sigmas import build_linear_sigmas, split_sigmas
from quantfunc.sampling.upscalers import InterpolateLatentUpscaler, video_only_upscaler

_DEFAULT_STALE_SPATIAL_KEYS: frozenset[str] = frozenset(
    {
        "keyframes",
        "keyframe_latents",
        "first_frame",
        "last_frame",
        "refs",
        "ref_images",
        "ref_latents",
        "ref_videos",
        "image_cond",
        "video_cond",
        "spatial_cond",
        "av_conds",
        "cond_latents",
        "before_denoise_conds",
    }
)


def reuse_h3_stage2_conditioning(
    stage1_conditioning: Mapping[str, Any],
    *,
    drop_spatial_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reuse Stage-1 text embeddings/tags for Stage 2 while dropping stale spatial inputs.

    Returns a new dictionary containing text-side entries (embeddings, masks,
    token tags, prompt metadata) and excluding resolution-bound spatial
    keyframes and reference tensors from Stage 1.
    """
    if not isinstance(stage1_conditioning, Mapping):
        raise TypeError(
            f"stage1_conditioning must be a Mapping, got {type(stage1_conditioning).__name__}."
        )
    drop_set = set(_DEFAULT_STALE_SPATIAL_KEYS)
    if drop_spatial_keys is not None:
        drop_set.update(drop_spatial_keys)
    return {k: v for k, v in stage1_conditioning.items() if k not in drop_set}


def _invoke_modular_stage(
    pipe: Any,
    call_kwargs: dict[str, Any],
    *,
    want_latents_only: bool,
) -> Any:
    """Invoke either a modular denoise-only interface or a standard pipeline callable."""
    if want_latents_only and callable(getattr(pipe, "run_denoise", None)):
        denoise_kwargs = {k: v for k, v in call_kwargs.items() if k != "output_type"}
        return pipe.run_denoise(**denoise_kwargs)

    if want_latents_only and hasattr(pipe, "blocks") and callable(getattr(pipe, "get_sub_pipeline", None)):
        blocks = getattr(pipe, "blocks", None)
        block_names = list(blocks.keys()) if isinstance(blocks, Mapping) else []
        denoise_blocks = [name for name in block_names if "decode" not in name.lower()]
        if denoise_blocks and len(denoise_blocks) < len(block_names):
            sub_pipe = pipe.get_sub_pipeline(denoise_blocks)
            denoise_kwargs = {k: v for k, v in call_kwargs.items() if k != "output_type"}
            return sub_pipe(**denoise_kwargs)

    return pipe(**call_kwargs)


def make_minimax_h3_two_stage_stages(
    *,
    width: int = 768,
    height: int = 512,
    num_frames: int = 81,
    scale_by: float = 1.5,
    total_steps: int = 20,
    split_step: int = 12,
    shift: float = 1.0,
    sigmas: Sequence[float] | None = None,
    stage2_quality_enhance: bool = True,
    guidance_scale: float | None = None,
    output_type: str = "pil",
    mode: str = "bilinear",
    canvas_multiple: int = 16,
    stage2_conditioning_builder: Callable[
        [Any, StageSpec, LatentBundle | None, dict[str, Any]], dict[str, Any] | None
    ]
    | None = None,
) -> list[StageSpec]:
    """Create a 2-stage MiniMax-H3 AV schedule using split sigmas and video-only upscaling."""
    full_sigmas = list(sigmas) if sigmas is not None else build_linear_sigmas(total_steps, shift=shift)
    stage1_sigmas, stage2_sigmas = split_sigmas(full_sigmas, split_step)

    upscaler = video_only_upscaler(
        InterpolateLatentUpscaler(
            scale_by=scale_by,
            mode=mode,
            canvas_multiple=canvas_multiple,
        )
    )
    stage2_width = align_size(width * scale_by, canvas_multiple)
    stage2_height = align_size(height * scale_by, canvas_multiple)

    def _stage1_adapter(
        pipe: Any,
        stage: StageSpec,
        bundle: LatentBundle | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        del stage, bundle
        raw = _invoke_modular_stage(pipe, call_kwargs, want_latents_only=True)
        if isinstance(raw, Mapping) and "latents" in raw:
            return SimpleNamespace(frames=raw["latents"], audio=raw.get("audio_latents"))
        return raw

    def _stage2_adapter(
        pipe: Any,
        stage: StageSpec,
        bundle: LatentBundle | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        del stage, bundle
        return _invoke_modular_stage(
            pipe,
            call_kwargs,
            want_latents_only=(output_type == "latent"),
        )

    def _default_stage2_conditioning(
        pipe: Any,
        stage: StageSpec,
        bundle: LatentBundle | None,
        call_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        updated = reuse_h3_stage2_conditioning(call_kwargs)
        if stage2_conditioning_builder is not None:
            custom = stage2_conditioning_builder(pipe, stage, bundle, updated)
            if isinstance(custom, dict):
                return custom
        return updated

    stage1 = StageSpec(
        name="base",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=len(stage1_sigmas) - 1,
        sigmas=stage1_sigmas,
        guidance_scale=guidance_scale,
        stage_call_adapter=_stage1_adapter,
        output_type="latent",
    )
    stage2 = StageSpec(
        name="refine",
        width=stage2_width,
        height=stage2_height,
        num_frames=num_frames,
        num_inference_steps=len(stage2_sigmas) - 1,
        sigmas=stage2_sigmas,
        guidance_scale=guidance_scale,
        upscaler=upscaler,
        stage_call_adapter=_stage2_adapter,
        conditioning_callback=_default_stage2_conditioning,
        transformer_overrides={"quality_enhance": True} if stage2_quality_enhance else None,
        output_type=output_type,
    )
    return [stage1, stage2]


def load_minimax_h3_pipeline(
    model_id_or_path: str,
    *,
    transformer_path: str | None = None,
    torch_dtype: torch.dtype | None = None,
    attention_backend: str = "auto",
    offload_mode: str | None = None,
    attach_hooks: bool = True,
    pipeline_cls: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Load a MiniMax-H3 AV modular pipeline with a QuantFunc transformer."""
    cls = pipeline_cls or import_diffusers_class(
        "diffusers",
        "MiniMaxH3ModularPipeline",
        repo_hint="https://github.com/MiniMax-AI/MiniMax-H3",
    )

    transformer = kwargs.pop("transformer", None)
    if transformer is None:
        transformer = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(
            transformer_path or model_id_or_path,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
        )

    pipe_kwargs: dict[str, Any] = dict(kwargs)
    if torch_dtype is not None:
        pipe_kwargs["torch_dtype"] = torch_dtype
    pipe = cls.from_pretrained(model_id_or_path, transformer=transformer, **pipe_kwargs)

    return finalize_pipeline_load(pipe, attach_hooks=attach_hooks, offload=offload_mode)
