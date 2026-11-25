"""Custom 3-stage sampling workflow (draft -> mid -> high-res refine) using L3 primitives.

Demonstrates the model-agnostic extensibility of ``quantfunc.sampling``:
- User-defined ``LatentUpscaler`` implementation conforming to the L3 protocol
- Three stages with progressive resolution and denoise schedules
- Per-stage ``transformer_overrides`` switching LoRAs, ``block_cache``, and
  ``quality_enhance`` with guaranteed automatic restoration on exit or error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from quantfunc import (
    AutoQuantFuncTransformer,
    FlowMatchRenoise,
    LatentBundle,
    LatentUpscaler,
    MultiStageRunner,
    StageSpec,
    attach_session_hooks,
)


class BicubicDetailUpscaler:
    """Custom ``LatentUpscaler`` applying bicubic spatial upscaling plus mild sharpening."""

    def __init__(self, scale_by: float = 1.5, sharpen: float = 0.08) -> None:
        self.scale_by = scale_by
        self.sharpen = sharpen

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        del target_height, target_width, kwargs
        lat = bundle.video_or_image
        orig_dtype = lat.dtype
        work = lat.float()
        out_h = max(1, int(round(work.shape[-2] * self.scale_by)))
        out_w = max(1, int(round(work.shape[-1] * self.scale_by)))
        up = F.interpolate(work, size=(out_h, out_w), mode="bicubic", align_corners=False)
        if self.sharpen > 0.0:
            blurred = F.avg_pool2d(up, kernel_size=3, stride=1, padding=1)
            up = up + self.sharpen * (up - blurred)
        return LatentBundle(
            video_or_image=up.to(dtype=orig_dtype),
            audio=bundle.audio,
            meta=dict(bundle.meta),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom 3-stage MultiStageRunner example.")
    parser.add_argument("--model-id", required=True, help="Base diffusers pipeline repo or path.")
    parser.add_argument("--transformer", required=True, help="QuantFunc checkpoint path.")
    parser.add_argument("--draft-lora", default=None, help="Optional fast draft LoRA path.")
    parser.add_argument("--detail-lora", default=None, help="Optional high-res detail LoRA path.")
    parser.add_argument("--prompt", default="an intricate clockwork observatory under a starry sky")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="custom_3stage.png")
    return parser.parse_args()


def build_three_stage_specs(
    *,
    draft_lora: str | None = None,
    detail_lora: str | None = None,
) -> list[StageSpec]:
    upscaler_mid: LatentUpscaler = BicubicDetailUpscaler(scale_by=1.5, sharpen=0.05)
    upscaler_hi: LatentUpscaler = BicubicDetailUpscaler(scale_by=1.3333, sharpen=0.10)

    draft_overrides: dict[str, Any] = {"block_cache": 0.08, "quality_enhance": False}
    if draft_lora:
        draft_overrides["loras"] = [(draft_lora, 1.0)]

    mid_overrides: dict[str, Any] = {"block_cache": 0.04, "quality_enhance": False, "loras": []}

    hi_overrides: dict[str, Any] = {"block_cache": 0.0, "quality_enhance": True}
    if detail_lora:
        hi_overrides["loras"] = [(detail_lora, 0.7)]

    return [
        StageSpec(
            name="draft",
            width=768,
            height=768,
            num_inference_steps=6,
            transformer_overrides=draft_overrides,
            output_type="latent",
        ),
        StageSpec(
            name="mid",
            width=1152,
            height=1152,
            num_inference_steps=6,
            denoise=0.60,
            upscaler=upscaler_mid,
            renoise=FlowMatchRenoise(tone_map_compression=0.1),
            transformer_overrides=mid_overrides,
            output_type="latent",
        ),
        StageSpec(
            name="high_res_refine",
            width=1536,
            height=1536,
            num_inference_steps=8,
            denoise=0.45,
            upscaler=upscaler_hi,
            renoise=FlowMatchRenoise(adain_factor=0.25),
            transformer_overrides=hi_overrides,
            output_type="pil",
        ),
    ]


def main() -> None:
    from diffusers import DiffusionPipeline

    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = AutoQuantFuncTransformer.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    attach_session_hooks(pipe)

    stages = build_three_stage_specs(draft_lora=args.draft_lora, detail_lora=args.detail_lora)
    runner = MultiStageRunner(stages)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = runner.run(pipe, prompt=args.prompt, generator=generator)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"Saved 3-stage output to {out_path}")


if __name__ == "__main__":
    main()
