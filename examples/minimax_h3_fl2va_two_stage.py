"""MiniMax-H3 FL2VA two-stage 1.5x audio-video sampling.

Encodes first/last keyframes through ``pipe.vae.encode(...)`` and binds them via
``transformer.set_av_conditions(keyframes=...)``, then runs
``make_minimax_h3_two_stage_stages`` (which internally splits sigmas, upscales
only the visual stream via ``video_only_upscaler``, and drops stale Stage-1
spatial inputs via ``reuse_h3_stage2_conditioning``) while re-encoding Stage-2
high-resolution keyframes in ``stage2_conditioning_builder``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from quantfunc import (
    LatentBundle,
    MultiStageRunner,
    QuantFuncMiniMaxH3Transformer3DModel,
    StageSpec,
    enable_offload,
    load_minimax_h3_pipeline,
    make_minimax_h3_two_stage_stages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 FL2VA two-stage example.")
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc MiniMax-H3 .safetensors.")
    parser.add_argument("--first-frame", required=True, help="Opening keyframe image path.")
    parser.add_argument("--last-frame", default=None, help="Optional closing keyframe image path.")
    parser.add_argument("--prompt", default="raindrops rippling across a neon-lit harbor with distant thunder")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--split-step", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="minimax_h3_fl2va_two_stage.mp4")
    return parser.parse_args()


def _encode_frame_latent(pipe: object, image_path: str, width: int, height: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((width, height))
    arr = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(height, width, 3)
    tensor = (arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2) / 127.5) - 1.0
    vae = getattr(pipe, "vae", None)
    if vae is None or not callable(getattr(vae, "encode", None)):
        raise RuntimeError("Pipeline must expose a VAE with .encode() to encode keyframe images into latents.")
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


def _build_keyframes(
    pipe: object,
    first_frame: str,
    last_frame: str | None,
    width: int,
    height: int,
    num_frames: int,
) -> list[dict[str, object]]:
    keyframes: list[dict[str, object]] = [
        {"frame_index": 0, "latent": _encode_frame_latent(pipe, first_frame, width, height)}
    ]
    if last_frame:
        keyframes.append(
            {
                "frame_index": num_frames - 1,
                "latent": _encode_frame_latent(pipe, last_frame, width, height),
            }
        )
    return keyframes


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

    stage1_kfs = _build_keyframes(pipe, args.first_frame, args.last_frame, args.width, args.height, args.num_frames)
    transformer.set_av_conditions(keyframes=stage1_kfs)

    def _rebuild_stage2_keyframes(
        active_pipe: Any,
        stage: StageSpec,
        bundle: LatentBundle | None,
        call_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        del bundle
        w = stage.width or int(round(args.width * args.scale))
        h = stage.height or int(round(args.height * args.scale))
        stage2_kfs = _build_keyframes(active_pipe, args.first_frame, args.last_frame, w, h, args.num_frames)
        active_pipe.transformer.set_av_conditions(keyframes=stage2_kfs)
        return call_kwargs

    stages = make_minimax_h3_two_stage_stages(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        scale_by=args.scale,
        total_steps=args.total_steps,
        split_step=args.split_step,
        mode="bilinear",
        canvas_multiple=16,
        stage2_quality_enhance=True,
        stage2_conditioning_builder=_rebuild_stage2_keyframes,
        output_type="pil",
    )

    runner = MultiStageRunner(stages)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = runner.run(pipe, prompt=args.prompt, generator=generator)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(out_path), fps=24)
    print(f"Saved two-stage MiniMax-H3 FL2VA video to {out_path}")


if __name__ == "__main__":
    main()
