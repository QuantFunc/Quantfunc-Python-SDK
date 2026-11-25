"""Generic, model-agnostic multi-stage sampling primitives."""

from __future__ import annotations

from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.renoise import FlowMatchRenoise
from quantfunc.sampling.runner import MultiStageRunner, StageResult, StageSpec, run_stages
from quantfunc.sampling.sigmas import build_linear_sigmas, denoise_sigmas, split_sigmas
from quantfunc.sampling.upscalers import (
    InterpolateLatentUpscaler,
    LatentUpscaler,
    video_only_upscaler,
)

__all__ = [
    "FlowMatchRenoise",
    "InterpolateLatentUpscaler",
    "LatentBundle",
    "LatentUpscaler",
    "MultiStageRunner",
    "StageResult",
    "StageSpec",
    "build_linear_sigmas",
    "denoise_sigmas",
    "run_stages",
    "split_sigmas",
    "video_only_upscaler",
]
