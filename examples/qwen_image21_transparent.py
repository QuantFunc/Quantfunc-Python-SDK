"""Qwen-Image-2.1 transparent RGBA image generation preserving the alpha channel."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    QuantFuncQwenImage21Transformer2DModel,
    enable_offload,
    load_qwenimage21_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 transparent RGBA generation example.")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc Qwen-Image-2.1 .safetensors.")
    parser.add_argument(
        "--prompt",
        default="a hand-painted ceramic teacup isolated on a transparent background, clean edges",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="qwen_image21_transparent.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncQwenImage21Transformer2DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = load_qwenimage21_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        attach_hooks=True,
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        generator=generator,
    )
    image = result.images[0]
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    print(f"Saved RGBA PNG ({image.mode}, {image.size}) to {out_path}")


if __name__ == "__main__":
    main()
