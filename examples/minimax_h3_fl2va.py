"""MiniMax-H3 first/last keyframe audio-video generation (FL2VA).

Uses ``FlowMatchResMultistepScheduler`` by default via ``load_minimax_h3_pipeline``,
demonstrates ``set_av_conditions(keyframes=...)`` with clean VAE-encoded keyframe
latents, and automatic data-direction adaptation with ``MiniMaxH3ModularPipeline``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    QuantFuncMiniMaxH3Transformer3DModel,
    enable_offload,
    load_minimax_h3_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 FL2VA example.")
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc MiniMax-H3 .safetensors.")
    parser.add_argument("--first-frame", required=True, help="Path to opening keyframe image.")
    parser.add_argument("--last-frame", default=None, help="Optional path to closing keyframe image.")
    parser.add_argument("--prompt", default="a vintage train gliding across a misty stone viaduct, rhythmic rail sound")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="minimax_h3_fl2va.mp4")
    return parser.parse_args()


def _encode_frame_latent(pipe: object, image_path: str, width: int, height: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((width, height))
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(height, width, 3)
    tensor = (arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 127.5) - 1.0
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return tensor
    tensor = tensor.to(device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        encoded = vae.encode(tensor)
    latent_dist = getattr(encoded, "latent_dist", None)
    return latent_dist.mode() if latent_dist is not None else encoded[0]


def main() -> None:
    from diffusers.utils import export_to_video

    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(
        args.transformer,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = load_minimax_h3_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        attach_hooks=True,
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "latent": _encode_frame_latent(pipe, args.first_frame, args.width, args.height)}
    ]
    if args.last_frame:
        keyframes.append(
            {
                "frame_index": args.num_frames - 1,
                "latent": _encode_frame_latent(pipe, args.last_frame, args.width, args.height),
            }
        )
    transformer.set_av_conditions(keyframes=keyframes)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(out_path), fps=24)
    print(f"Saved MiniMax-H3 FL2VA video to {out_path}")


if __name__ == "__main__":
    main()
