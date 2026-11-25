"""Krea-2 two-stage 1.5x high-resolution text-to-image generation.

Stage 1 generates a base latent with default speed settings; Stage 2 upscales by 1.5x,
re-noises with FlowMatchRenoise, temporarily enables ``quality_enhance=True`` via
``transformer_overrides``, and restores transformer settings automatically on exit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    MultiStageRunner,
    QuantFuncKrea2Transformer2DModel,
    enable_offload,
    load_krea2_pipeline,
    make_krea2_two_stage_stages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Krea-2 1.5x two-stage T2I example.")
    parser.add_argument("--model-id", default="krea/krea-2")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc Krea-2 .safetensors.")
    parser.add_argument("--prompt", default="a sunlit botanical greenhouse filled with glass orchids")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--stage1-steps", type=int, default=8)
    parser.add_argument("--stage2-steps", type=int, default=8)
    parser.add_argument("--denoise", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="krea2_two_stage.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncKrea2Transformer2DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = load_krea2_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        attach_hooks=True,
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    stages = make_krea2_two_stage_stages(
        width=args.width,
        height=args.height,
        scale_by=args.scale,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        denoise=args.denoise,
        stage2_quality_enhance=True,
        output_type="pil",
    )
    runner = MultiStageRunner(stages)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = runner.run(pipe, prompt=args.prompt, generator=generator)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"Saved two-stage image to {out_path}")


if __name__ == "__main__":
    main()
