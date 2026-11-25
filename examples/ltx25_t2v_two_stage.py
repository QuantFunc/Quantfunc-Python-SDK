"""LTX-2.5 audio-video two-stage generation with 2x neural latent upscaling.

Demonstrates ``ensure_deterministic_scheduler``, default ``res_multistep``
scheduler alignment (with ``copy.deepcopy`` history isolation across dual
video/audio schedulers), block/step caching, Sol sparse attention,
``LTX2NeuralLatentUpscaler``, and Stage-2 latent-level frozen audio restoration
executed before ``audio_vae.decode`` and ``vocoder``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from quantfunc import (
    LTX2NeuralLatentUpscaler,
    MultiStageRunner,
    QuantFuncLTX25Transformer3DModel,
    SolConfig,
    apply_cache_on_pipe,
    apply_sol_on_pipe,
    enable_offload,
    load_ltx25_pipeline,
    make_ltx25_two_stage_stages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2.5 AV two-stage generation example.")
    parser.add_argument("--model-id", default="Lightricks/LTX-2.5")
    parser.add_argument("--transformer", required=True, help="Path to QuantFunc LTX-2.5 .safetensors.")
    parser.add_argument("--upscaler-id", required=True, help="Repo or path for LTX-2 latent upsampler.")
    parser.add_argument("--connectors", default=None, help="Optional connectors .safetensors path.")
    parser.add_argument("--prompt", default="ocean waves crashing against rocky cliffs at dusk")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--stage1-steps", type=int, default=20)
    parser.add_argument("--stage2-steps", type=int, default=3)
    parser.add_argument("--block-cache", type=float, default=0.05)
    parser.add_argument("--step-cache", type=float, default=0.12)
    parser.add_argument("--sol-tau", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=("none", "model", "low_vram"), default="model")
    parser.add_argument("--output", default="ltx25_two_stage.mp4")
    return parser.parse_args()


def main() -> None:
    from diffusers import LTX2LatentUpsamplePipeline
    from diffusers.utils import export_to_video

    args = parse_args()
    api_key = os.environ.get("QUANTFUNC_API_KEY")

    transformer = QuantFuncLTX25Transformer3DModel.from_pretrained(
        args.transformer,
        connectors_path=args.connectors,
        torch_dtype=torch.bfloat16,
        api_key=api_key,
    )
    pipe = load_ltx25_pipeline(
        args.model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        # deterministic_scheduler=True invokes ensure_deterministic_scheduler(pipe) once automatically.
        deterministic_scheduler=True,
        attach_hooks=True,
    )
    apply_cache_on_pipe(pipe, block_cache=args.block_cache, step_cache=args.step_cache)
    apply_sol_on_pipe(
        pipe,
        SolConfig(enabled=True, tau=args.sol_tau, start_percent=0.0, end_percent=0.9),
    )
    if args.offload != "none":
        enable_offload(pipe, mode=args.offload)

    upsample_pipe = LTX2LatentUpsamplePipeline.from_pretrained(
        args.upscaler_id,
        vae=pipe.vae,
        torch_dtype=torch.bfloat16,
    )
    upscaler = LTX2NeuralLatentUpscaler(upsample_pipe, vae=pipe.vae, latents_normalized=False)

    stages = make_ltx25_two_stage_stages(
        upscaler=upscaler,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        freeze_audio_in_stage2=True,
        output_type="pil",
    )
    runner = MultiStageRunner(stages)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = runner.run(pipe, prompt=args.prompt, generator=generator)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(out_path), fps=24)
    print(f"Saved LTX-2.5 video to {out_path} (audio tensor present: {result.audio is not None})")


if __name__ == "__main__":
    main()
