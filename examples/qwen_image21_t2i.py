"""Qwen-Image-2.1 6-step T2I with default ``res_multistep`` alignment, Lightning LoRA hot-loading, and pre-norm text encoding."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    QuantFuncQwenImage21Transformer2DModel,
    enable_offload,
    encode_qwen3_vl_pre_norm,
    load_qwenimage21_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image-2.1 T2I + Lightning LoRA example.")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc Qwen-Image-2.1 .safetensors.")
    parser.add_argument("--lora", default=None, help="Optional Lightning LoRA .safetensors path.")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--prompt", default="an architectural watercolor of a coastal lighthouse")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use-pre-norm", action="store_true", help="Extract Qwen3-VL pre-norm hidden states.")
    parser.add_argument("--quality-enhance", action="store_true", default=True)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="qwen_image21_t2i.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncQwenImage21Transformer2DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    transformer.set_quality_enhance(args.quality_enhance)
    if args.lora:
        transformer.load_lora(args.lora, scale=args.lora_scale)

    pipe = load_qwenimage21_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        attach_hooks=True,
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    call_kwargs: dict[str, object] = {
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }

    if args.use_pre_norm and hasattr(pipe, "tokenizer") and hasattr(pipe, "text_encoder"):
        tokens = pipe.tokenizer(args.prompt, return_tensors="pt")
        tokens = {k: v.to(pipe.text_encoder.device) for k, v in tokens.items()}
        with torch.no_grad():
            prompt_embeds = encode_qwen3_vl_pre_norm(pipe.text_encoder, **tokens)
        call_kwargs["prompt_embeds"] = prompt_embeds
        call_kwargs["prompt_embeds_mask"] = tokens["attention_mask"]
    else:
        call_kwargs["prompt"] = args.prompt

    result = pipe(**call_kwargs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    print(f"Saved image to {out_path}")


if __name__ == "__main__":
    main()
