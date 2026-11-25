"""MiniMax-H3 multimodal reference audio-video generation (Ref2VA).

Demonstrates binding image, video, audio, or video-audio reference conditions
together with ``text_token_tags`` so the engine derives ``text_vision_row_ranges``.
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
    parser = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA multimodal reference example.")
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc MiniMax-H3 .safetensors.")
    parser.add_argument("--ref-image", action="append", default=[], help="Reference image path(s).")
    parser.add_argument("--ref-audio", default=None, help="Optional reference audio latent .pt path.")
    parser.add_argument("--ref-video", default=None, help="Optional reference video latent .pt path.")
    parser.add_argument("--prompt", default="the character plays acoustic guitar in a warm wooden studio")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="minimax_h3_ref2va.mp4")
    return parser.parse_args()


def _encode_image_latent(pipe: object, path: str, width: int, height: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((width, height))
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(height, width, 3)
    tensor = (arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 127.5) - 1.0
    vae = getattr(pipe, "vae", None)
    if vae is None or not callable(getattr(vae, "encode", None)):
        raise RuntimeError("Pipeline must expose a VAE with .encode() to encode reference images into latents.")
    tensor = tensor.to(device=getattr(vae, "device", tensor.device), dtype=getattr(vae, "dtype", tensor.dtype))
    with torch.no_grad():
        encoded = vae.encode(tensor)
    latent_dist = getattr(encoded, "latent_dist", None)
    latent = latent_dist.mode() if latent_dist is not None else encoded[0]
    vae_cfg = getattr(vae, "config", None)
    shift_factor = getattr(vae_cfg, "shift_factor", None) if vae_cfg is not None else None
    if isinstance(shift_factor, int | float) and not isinstance(shift_factor, bool) and shift_factor != 0.0:
        latent = latent - float(shift_factor)
    scaling_factor = getattr(vae_cfg, "scaling_factor", None) if vae_cfg is not None else None
    if isinstance(scaling_factor, int | float) and not isinstance(scaling_factor, bool) and scaling_factor != 1.0:
        latent = latent * float(scaling_factor)
    return latent


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

    refs: list[dict[str, object]] = []
    for img_path in args.ref_image:
        refs.append({"kind": "image", "latent": _encode_image_latent(pipe, img_path, args.width, args.height)})
    if args.ref_video:
        refs.append({"kind": "video", "latent": torch.load(args.ref_video, map_location="cpu", weights_only=True)})
    if args.ref_audio:
        refs.append({"kind": "audio", "audio_latent": torch.load(args.ref_audio, map_location="cpu", weights_only=True)})

    # Token tags == 0 mark visual-reference token spans for text_vision_row_ranges
    text_token_tags = [0, 0, 1, 1, 1, 1] if refs else None
    transformer.set_av_conditions(refs=refs, text_token_tags=text_token_tags)

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
    print(f"Saved MiniMax-H3 Ref2VA video to {out_path}")


if __name__ == "__main__":
    main()
