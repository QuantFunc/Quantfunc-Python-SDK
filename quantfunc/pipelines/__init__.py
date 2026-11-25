"""Model-specific pipeline loaders, two-stage presets, and ergonomic helpers."""

from __future__ import annotations

from quantfunc.pipelines.krea2 import load_krea2_pipeline, make_krea2_two_stage_stages
from quantfunc.pipelines.ltx25 import (
    LTX2NeuralLatentUpscaler,
    ensure_deterministic_scheduler,
    load_ltx25_pipeline,
    make_ltx25_two_stage_stages,
)
from quantfunc.pipelines.minimax_h3 import (
    load_minimax_h3_pipeline,
    make_minimax_h3_two_stage_stages,
    reuse_h3_stage2_conditioning,
)
from quantfunc.pipelines.qwenimage21 import encode_qwen3_vl_pre_norm, load_qwenimage21_pipeline

__all__ = [
    "LTX2NeuralLatentUpscaler",
    "encode_qwen3_vl_pre_norm",
    "ensure_deterministic_scheduler",
    "load_krea2_pipeline",
    "load_ltx25_pipeline",
    "load_minimax_h3_pipeline",
    "load_qwenimage21_pipeline",
    "make_krea2_two_stage_stages",
    "make_ltx25_two_stage_stages",
    "make_minimax_h3_two_stage_stages",
    "reuse_h3_stage2_conditioning",
]
