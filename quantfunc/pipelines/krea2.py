"""Krea-2 pipeline loader and two-stage high-resolution sampling helpers."""

from __future__ import annotations

import math
from typing import Any

import torch

from quantfunc.models.krea2 import QuantFuncKrea2Transformer2DModel
from quantfunc.pipelines._common import (
    align_size,
    finalize_pipeline_load,
    import_diffusers_class,
)
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.renoise import FlowMatchRenoise
from quantfunc.sampling.runner import StageSpec
from quantfunc.sampling.upscalers import InterpolateLatentUpscaler


def _infer_grid_hw(seq_len: int, meta: dict[str, Any]) -> tuple[int, int]:
    grid_h = meta.get("grid_h")
    grid_w = meta.get("grid_w")
    if isinstance(grid_h, int) and isinstance(grid_w, int) and grid_h * grid_w == seq_len:
        return grid_h, grid_w

    prev_h = meta.get("prev_height")
    prev_w = meta.get("prev_width")
    if isinstance(prev_h, int) and isinstance(prev_w, int) and prev_h > 0 and prev_w > 0:
        ratio = prev_h / prev_w
        est_h = max(1, int(round(math.sqrt(seq_len * ratio))))
        for delta in range(seq_len):
            for cand_h in (est_h + delta, est_h - delta):
                if cand_h >= 1 and seq_len % cand_h == 0:
                    cand_w = seq_len // cand_h
                    if abs((cand_h / cand_w) - ratio) <= 1e-3 or delta == 0:
                        return cand_h, cand_w

    side = math.isqrt(seq_len)
    if side * side == seq_len:
        return side, side

    raise ValueError(
        f"Cannot infer 2D latent grid for 3D sequence length {seq_len}; "
        "provide `grid_h`/`grid_w` or `prev_height`/`prev_width` in LatentBundle.meta."
    )


class _Krea2PackedLatentUpscaler:
    """Adapter around ``InterpolateLatentUpscaler`` supporting 3D packed and 4D latents."""

    def __init__(self, inner: InterpolateLatentUpscaler) -> None:
        self._inner = inner

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        tensor = bundle.video_or_image
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor latent, got {type(tensor).__name__}.")

        if tensor.ndim == 4:
            return self._inner(
                bundle,
                target_height=target_height,
                target_width=target_width,
                **kwargs,
            )

        if tensor.ndim != 3:
            raise ValueError(
                f"Krea-2 latent upscaler expects 3D [B, N, C] or 4D [B, C, H, W], got {tuple(tensor.shape)}."
            )

        bsz, seq_len, channels = tensor.shape
        grid_h, grid_w = _infer_grid_hw(int(seq_len), bundle.meta)
        lat_4d = tensor.reshape(bsz, grid_h, grid_w, channels).permute(0, 3, 1, 2).contiguous()

        up_4d_bundle = self._inner(
            LatentBundle(video_or_image=lat_4d, audio=bundle.audio, meta=dict(bundle.meta)),
            target_height=target_height,
            target_width=target_width,
            **kwargs,
        )
        up_4d = up_4d_bundle.video_or_image
        out_h, out_w = int(up_4d.shape[-2]), int(up_4d.shape[-1])
        up_3d = up_4d.permute(0, 2, 3, 1).reshape(bsz, out_h * out_w, channels).contiguous()

        new_meta = dict(up_4d_bundle.meta)
        new_meta["grid_h"] = out_h
        new_meta["grid_w"] = out_w
        return LatentBundle(video_or_image=up_3d, audio=bundle.audio, meta=new_meta)


def make_krea2_two_stage_stages(
    *,
    width: int = 1024,
    height: int = 1024,
    scale_by: float = 1.5,
    stage1_steps: int = 8,
    stage2_steps: int = 8,
    denoise: float = 0.55,
    stage2_quality_enhance: bool = True,
    guidance_scale: float | None = None,
    output_type: str = "pil",
    mode: str = "bilinear",
    canvas_multiple: int = 16,
) -> list[StageSpec]:
    """Create a 2-stage Krea-2 generation schedule (base -> upscaled refine)."""
    inner_upscaler = InterpolateLatentUpscaler(
        scale_by=scale_by,
        mode=mode,
        canvas_multiple=canvas_multiple,
    )
    adapter_upscaler = _Krea2PackedLatentUpscaler(inner_upscaler)
    stage2_width = align_size(width * scale_by, canvas_multiple)
    stage2_height = align_size(height * scale_by, canvas_multiple)

    stage1 = StageSpec(
        name="base",
        width=width,
        height=height,
        num_inference_steps=stage1_steps,
        guidance_scale=guidance_scale,
        output_type="latent",
    )
    stage2 = StageSpec(
        name="refine",
        width=stage2_width,
        height=stage2_height,
        num_inference_steps=stage2_steps,
        guidance_scale=guidance_scale,
        denoise=denoise,
        upscaler=adapter_upscaler,
        renoise=FlowMatchRenoise(),
        transformer_overrides={"quality_enhance": bool(stage2_quality_enhance)},
        output_type=output_type,
    )
    return [stage1, stage2]


def load_krea2_pipeline(
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
    """Load a Krea-2 pipeline with a QuantFunc-accelerated transformer."""
    cls = pipeline_cls or import_diffusers_class(
        "diffusers",
        "Krea2Pipeline",
        repo_hint="https://github.com/krea-ai/krea-2",
    )

    transformer = kwargs.pop("transformer", None)
    if transformer is None:
        transformer = QuantFuncKrea2Transformer2DModel.from_pretrained(
            transformer_path or model_id_or_path,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
        )

    pipe_kwargs: dict[str, Any] = dict(kwargs)
    if torch_dtype is not None:
        pipe_kwargs["torch_dtype"] = torch_dtype
    pipe = cls.from_pretrained(model_id_or_path, transformer=transformer, **pipe_kwargs)

    return finalize_pipeline_load(pipe, attach_hooks=attach_hooks, offload=offload_mode)
