"""Qwen-Image-2.1 single- and multi-reference image editing example.

Reference editing automatically engages full quality mode on the transformer
and supports multiple reference images with optional ``image_slots`` positions.
"""

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
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 reference image editing example.")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc Qwen-Image-2.1 .safetensors.")
    parser.add_argument(
        "--ref-image",
        action="append",
        required=True,
        help="Path to a reference image (pass multiple times for multi-reference editing).",
    )
    parser.add_argument("--prompt", default="place the subject in a softly lit modern studio")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="qwen_image21_edit.png")
    return parser.parse_args()


def main() -> None:
    from PIL import Image

    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    ref_images = [Image.open(path).convert("RGB") for path in args.ref_image]
    transformer = QuantFuncQwenImage21Transformer2DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = load_qwenimage21_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        variant="edit",
        attach_hooks=True,
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image_arg = ref_images[0] if len(ref_images) == 1 else ref_images
    result = pipe(
        prompt=args.prompt,
        image=image_arg,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"Saved edited image to {out_path}")


if __name__ == "__main__":
    main()
