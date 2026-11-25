"""Krea-2 single-stage text-to-image generation with QuantFunc drop-in acceleration.

Uses ``FlowMatchResMultistepScheduler`` (aligned with ComfyUI ``res_multistep``)
by default via ``load_krea2_pipeline(..., scheduler="res_multistep")``; pass
``scheduler="euler"`` or ``scheduler="keep"`` to opt out.
Authentication is loaded from ``QUANTFUNC_API_KEY`` or ``~/.config/quantfunc/config.json``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    QuantFuncKrea2Transformer2DModel,
    SolConfig,
    apply_cache_on_pipe,
    apply_sol_on_pipe,
    attach_session_hooks,
    enable_offload,
    load_krea2_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Krea-2 single-stage T2I example.")
    parser.add_argument("--model-id", default="krea/krea-2", help="Base pipeline repo or local path.")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc Krea-2 .safetensors.")
    parser.add_argument("--prompt", default="a cinematic mountain sunrise over alpine mist")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--block-cache", type=float, default=0.0)
    parser.add_argument("--sol-tau", type=float, default=0.0, help="Enable Sol sparse attention if > 0.")
    parser.add_argument("--quality-enhance", action="store_true")
    parser.add_argument("--output", default="krea2_t2i.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncKrea2Transformer2DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        api_key=api_key,
    )
    transformer.set_quality_enhance(args.quality_enhance)

    pipe = load_krea2_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        attach_hooks=False,
    )
    attach_session_hooks(pipe)

    if args.block_cache > 0.0:
        apply_cache_on_pipe(pipe, block_cache=args.block_cache)
    if args.sol_tau > 0.0:
        apply_sol_on_pipe(
            pipe,
            SolConfig(enabled=True, tau=args.sol_tau, start_percent=0.0, end_percent=0.9),
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
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
