"""Drop-in Qwen-Image-2.1 2D transformer (T2I + Edit refs + split-CFG QualityLaw guard)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from quantfunc.core.config import WIRE_QUALITY_RATIO_KEY
from quantfunc.core.registry import register_model
from quantfunc.models.base import (
    QuantFuncBaseTransformer,
    Transformer2DModelOutput,
    _torch_dtype_to_qf,
)
from quantfunc.native.capi import QF_MAX_STEP_REF_AREA, QF_MAX_STEP_REFS

_RANK_3D = 3
_RANK_4D = 4
_QUALITY_FULL_EPS = 0.9999


def _normalize_img_shapes(img_shapes: Any) -> list[tuple[int, int, int]]:
    """Normalize diffusers img_shapes into a flat list of (frames, h, w) tuples."""
    if not isinstance(img_shapes, list | tuple) or len(img_shapes) == 0:
        raise ValueError(f"Qwen-Image-2.1 img_shapes must be a non-empty sequence, got {img_shapes!r}")
    first = img_shapes[0]
    raw_list: Sequence[Any]
    if isinstance(first, list | tuple) and len(first) > 0 and isinstance(first[0], list | tuple):
        raw_list = first
    else:
        raw_list = img_shapes

    parsed: list[tuple[int, int, int]] = []
    for item in raw_list:
        if not isinstance(item, list | tuple) or len(item) != _RANK_3D:
            raise ValueError(f"Qwen-Image-2.1 img_shapes entries must be (f, h, w) 3-tuples, got {item!r}")
        f_dim, h_dim, w_dim = (int(item[0]), int(item[1]), int(item[2]))
        if f_dim <= 0 or h_dim <= 0 or w_dim <= 0:
            raise ValueError(f"Qwen-Image-2.1 img_shapes dimensions must be positive, got {(f_dim, h_dim, w_dim)}")
        parsed.append((f_dim, h_dim, w_dim))
    return parsed


@register_model(
    arch="qwenimage21",
    pipeline_classes=("QwenImage21Pipeline", "QwenImageEditPipeline", "QwenImagePipeline"),
    transformer_classes=("QwenImage21Transformer2DModel", "QwenImageTransformer2DModel"),
    engine_arch="qwenimage21",
)
class QuantFuncQwenImage21Transformer2DModel(QuantFuncBaseTransformer):
    """Drop-in Qwen-Image-2.1 transformer supporting T2I, multi-reference Edit, and QualityLaw."""

    model_arch: str = "qwenimage21"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_opened_ratio: float | None = None
        self._split_cfg_forced_full: bool = False
        self._last_sigma: float | None = None

    def _end_session_if_open(self) -> tuple[bool, bool]:
        self._session_opened_ratio = None
        self._split_cfg_forced_full = False
        self._last_sigma = None
        return super()._end_session_if_open()

    def set_staged_schedule(self, schedule: Any) -> None:
        self._session_opened_ratio = None
        self._split_cfg_forced_full = False
        self._last_sigma = None
        super().set_staged_schedule(schedule)

    def _in_channels(self) -> int:
        val = self.config.get("in_channels")
        if val is None or int(val) <= 0:
            raise ValueError(f"Qwen-Image-2.1 config missing positive in_channels: {val!r}")
        return int(val)

    def _context_in_dim(self) -> int:
        val = self.config.get("context_in_dim")
        if val is None:
            val = self.config.get("joint_attention_dim")
        if val is None or int(val) <= 0:
            raise ValueError(f"Qwen-Image-2.1 config missing positive context_in_dim: {val!r}")
        return int(val)

    def _effective_quality_ratio(
        self,
        has_refs: bool = False,
        batch: int = 1,
        single_cond_group: bool = True,
    ) -> float:
        """Dual QualityLaw: force 1.0 on quality_enhance, edit refs, batch > 1, or CFG > 1."""
        if self._quality_enhance or has_refs or batch > 1 or not single_cond_group:
            return 1.0
        return super()._effective_quality_ratio(
            has_refs=has_refs,
            batch=batch,
            single_cond_group=single_cond_group,
        )

    def _unpack_target_and_refs(
        self,
        hidden_states: torch.Tensor,
        *,
        ref_latents: Sequence[torch.Tensor] | None,
        img_shapes: Any,
        height: int | None,
        width: int | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], bool, int]:
        """Extract 4D target [B, C, H, W] and 4D ref list from explicit or concatenated inputs."""
        in_ch = self._in_channels()
        explicit_refs = list(ref_latents) if ref_latents is not None else []

        if hidden_states.ndim == _RANK_4D:
            if int(hidden_states.shape[1]) != in_ch:
                raise ValueError(
                    f"Qwen-Image-2.1 4D hidden_states channels={int(hidden_states.shape[1])} "
                    f"must match config.in_channels={in_ch}"
                )
            return hidden_states.contiguous(), explicit_refs, False, 0

        if hidden_states.ndim == _RANK_3D:
            batch, n_total, ch = (int(d) for d in hidden_states.shape)
            if ch != in_ch:
                raise ValueError(
                    f"Qwen-Image-2.1 3D hidden_states channels={ch} must match config.in_channels={in_ch}"
                )
            if img_shapes is not None:
                shapes = _normalize_img_shapes(img_shapes)
                f0, th, tw = shapes[0]
                if f0 != 1:
                    raise ValueError(f"Qwen-Image-2.1 target img_shapes frame count must be 1, got {f0}")
                n_target = th * tw
                ref_shapes = shapes[1:]
                expected_total = n_target + sum(rf * rh * rw for rf, rh, rw in ref_shapes)
                if expected_total != n_total:
                    raise ValueError(
                        f"Qwen-Image-2.1 img_shapes total tokens {expected_total} != hidden_states length {n_total}"
                    )
                if ref_shapes and explicit_refs:
                    raise ValueError(
                        "Qwen-Image-2.1 received both concatenated img_shapes references and explicit ref_latents"
                    )
                target4d = (
                    hidden_states[:, :n_target, :]
                    .permute(0, 2, 1)
                    .reshape(batch, in_ch, th, tw)
                    .contiguous()
                )
                extracted_refs: list[torch.Tensor] = list(explicit_refs)
                offset = n_target
                for rf, rh, rw in ref_shapes:
                    if rf != 1:
                        raise ValueError(f"Qwen-Image-2.1 ref img_shapes frame count must be 1, got {rf}")
                    seg = rh * rw
                    ref_slice = (
                        hidden_states[:, offset : offset + seg, :]
                        .permute(0, 2, 1)
                        .reshape(batch, in_ch, rh, rw)
                        .contiguous()
                    )
                    extracted_refs.append(ref_slice)
                    offset += seg
                tail_pad = n_total - n_target
                return target4d, extracted_refs, True, tail_pad

            if height is not None and width is not None:
                h_val, w_val = int(height), int(width)
                scale = self._spatial_scale()
                if h_val * w_val == n_total:
                    th, tw = h_val, w_val
                elif h_val % scale == 0 and w_val % scale == 0 and (h_val // scale) * (w_val // scale) == n_total:
                    th, tw = h_val // scale, w_val // scale
                else:
                    raise ValueError(
                        f"Qwen-Image-2.1 cannot map height={h_val}, width={w_val} to {n_total} tokens"
                    )
                target4d = (
                    hidden_states.permute(0, 2, 1)
                    .reshape(batch, in_ch, th, tw)
                    .contiguous()
                )
                return target4d, explicit_refs, True, 0

            raise ValueError(
                "Qwen-Image-2.1 3D hidden_states requires img_shapes or (height, width)"
            )

        raise ValueError(f"Qwen-Image-2.1 hidden_states must be 3D or 4D, got {hidden_states.ndim}D")

    def _validate_refs_and_slots(
        self,
        refs: list[torch.Tensor],
        image_slots: Sequence[int] | None,
        txt_len: int,
        batch: int,
    ) -> list[int]:
        in_ch = self._in_channels()
        num_refs = len(refs)
        if num_refs == 0:
            if image_slots is not None and len(image_slots) != 0:
                raise ValueError(
                    f"Qwen-Image-2.1 image_slots length ({len(image_slots)}) must match ref_latents count (0)"
                )
            return []

        if num_refs > QF_MAX_STEP_REFS:
            raise ValueError(
                f"Qwen-Image-2.1 supports at most {QF_MAX_STEP_REFS} reference latents (got {num_refs})"
            )

        total_area = 0
        for idx, ref in enumerate(refs):
            if not torch.is_tensor(ref) or ref.ndim != _RANK_4D:
                raise ValueError(
                    f"Qwen-Image-2.1 ref_latents[{idx}] must be a 4D tensor [B, C, h, w]"
                )
            rb, rc, rh, rw = (int(d) for d in ref.shape)
            if rb not in (1, batch) or rc != in_ch or rh <= 0 or rw <= 0:
                raise ValueError(
                    f"Qwen-Image-2.1 ref_latents[{idx}] has invalid shape {tuple(ref.shape)} "
                    f"(expected B in {{1, {batch}}}, C={in_ch}, positive h/w)"
                )
            total_area += rh * rw
            if total_area > QF_MAX_STEP_REF_AREA:
                raise ValueError(
                    f"Qwen-Image-2.1 total reference area {total_area} exceeds limit {QF_MAX_STEP_REF_AREA}"
                )

        if image_slots is None:
            return [txt_len] * num_refs

        if len(image_slots) != num_refs:
            raise ValueError(
                f"Qwen-Image-2.1 image_slots length ({len(image_slots)}) must equal ref_latents count ({num_refs})"
            )

        slots: list[int] = []
        prev = 0
        for idx, raw_slot in enumerate(image_slots):
            if isinstance(raw_slot, bool) or not isinstance(raw_slot, int):
                raise ValueError(f"Qwen-Image-2.1 image_slots[{idx}] must be int, got {raw_slot!r}")
            if raw_slot < prev or raw_slot > txt_len:
                raise ValueError(
                    f"Qwen-Image-2.1 image_slots must be non-decreasing in [0, {txt_len}], "
                    f"got {raw_slot} at index {idx} after {prev}"
                )
            slots.append(raw_slot)
            prev = raw_slot
        return slots

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | float | int | None = None,
        ref_latents: Sequence[torch.Tensor] | None = None,
        image_slots: Sequence[int] | None = None,
        img_shapes: Any = None,
        height: int | None = None,
        width: int | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Run one Qwen-Image-2.1 denoising step (T2I or multi-reference Edit)."""
        for forbidden in (
            "controlnet_block_samples",
            "controlnet_single_block_samples",
            "control",
            "audio_hidden_states",
        ):
            if kwargs.get(forbidden) is not None:
                raise ValueError(f"Qwen-Image-2.1 does not support '{forbidden}' conditioning")

        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image-2.1 requires encoder_hidden_states")
        if timestep is None:
            raise ValueError("Qwen-Image-2.1 forward requires timestep")

        orig_dtype = hidden_states.dtype
        dev = hidden_states.device
        target4d, refs, was_3d, tail_pad = self._unpack_target_and_refs(
            hidden_states,
            ref_latents=ref_latents,
            img_shapes=img_shapes,
            height=height,
            width=width,
        )
        batch, in_ch, th, tw = (int(d) for d in target4d.shape)

        ctx = encoder_hidden_states
        if ctx.ndim != _RANK_3D or int(ctx.shape[0]) != batch or int(ctx.shape[2]) != self._context_in_dim():
            raise ValueError(
                f"Qwen-Image-2.1 encoder_hidden_states shape {tuple(ctx.shape)} must be "
                f"[{batch}, S_txt, {self._context_in_dim()}]"
            )
        txt_len = int(ctx.shape[1])
        slots = self._validate_refs_and_slots(refs, image_slots, txt_len, batch)
        has_refs = len(refs) > 0

        sigmas = self._extract_batch_sigmas(timestep, batch)
        act_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16, torch.float32) else orig_dtype
        x_in = target4d if target4d.dtype == act_dtype else target4d.to(dtype=act_dtype)
        ctx_in = ctx if (ctx.dtype == act_dtype and ctx.device == dev) else ctx.to(device=dev, dtype=act_dtype)

        spatial_scale = self._spatial_scale()
        pixel_h = th * spatial_scale
        pixel_w = tw * spatial_scale

        staged_do_cfg = bool(
            self._staged_schedule is not None and bool(self._staged_schedule.do_cfg)
        )
        if not self._session_is_open():
            self._session_opened_ratio = None
            self._split_cfg_forced_full = False
        elif self._last_sigma is not None and sigmas[0] > self._last_sigma + 1e-6:
            # New trajectory starting at a higher sigma resets split-CFG escalation.
            self._session_opened_ratio = None
            self._split_cfg_forced_full = False

        out4d = torch.empty_like(x_in)
        geom_key = (in_ch, th, tw, has_refs, int(ctx_in.shape[2]))
        stream_ptr = self._producer_stream_ptr(x_in)
        cond_qf_dtype = _torch_dtype_to_qf(ctx_in.dtype)

        for i in range(batch):
            xi = x_in[i : i + 1].contiguous()
            oi = self._ensure_scratch("qi21_out", tuple(xi.shape), xi.dtype, dev)
            ci = ctx_in[i : i + 1].contiguous()
            sig_i = sigmas[i]
            ctx_id = self._context_fingerprint(ci, i)

            was_open_before = self._session_is_open()
            single_cond = (batch == 1) and (not staged_do_cfg) and (not self._split_cfg_forced_full)
            opts = self._build_base_session_options(
                has_refs=has_refs,
                batch=batch,
                single_cond_group=single_cond,
            )
            step_idx, total_steps, ctx_key = self._ensure_session_step(
                width=pixel_w,
                height=pixel_h,
                sigma=sig_i,
                max_context_dims=self._max_context_dims_for(ci),
                cond_dtype=cond_qf_dtype,
                options=opts,
                geometry_key=geom_key,
                ctx_identity=ctx_id,
                producer_stream=stream_ptr,
            )
            if not was_open_before or self._session_opened_ratio is None:
                self._session_opened_ratio = float(opts[WIRE_QUALITY_RATIO_KEY])

            # Tier-2 sequential split-CFG guard: reopen at ratio 1.0 upon encountering a second CFG branch.
            if self._session_opened_ratio < _QUALITY_FULL_EPS and (
                ctx_key > 1
                or getattr(self._session_tracker, "active_branch_count", 1) > 1
                or has_refs
                or batch > 1
            ):
                self._handle.end_session_if_open()
                if hasattr(self._session_tracker, "on_session_closed"):
                    self._session_tracker.on_session_closed()
                self._split_cfg_forced_full = True
                full_opts = self._build_base_session_options(
                    has_refs=has_refs,
                    batch=batch,
                    single_cond_group=False,
                )
                step_idx, total_steps, ctx_key = self._ensure_session_step(
                    width=pixel_w,
                    height=pixel_h,
                    sigma=sig_i,
                    max_context_dims=self._max_context_dims_for(ci),
                    cond_dtype=cond_qf_dtype,
                    options=full_opts,
                    geometry_key=geom_key,
                    ctx_identity=ctx_id,
                    producer_stream=stream_ptr,
                )
                self._session_opened_ratio = float(full_opts[WIRE_QUALITY_RATIO_KEY])

            if has_refs:
                ref_i = [
                    (r[i : i + 1] if int(r.shape[0]) > 1 else r)
                    .to(device=dev, dtype=xi.dtype)
                    .contiguous()
                    for r in refs
                ]
                self._run_step_refs(
                    latent_in=xi,
                    velocity_out=oi,
                    sigma=sig_i,
                    step_index=step_idx,
                    total_steps=total_steps,
                    context=ci,
                    ref_latents=ref_i,
                    image_slots=slots,
                    cfg_context_key=ctx_key,
                    producer_stream=stream_ptr,
                )
            else:
                self._run_step(
                    latent_in=xi,
                    velocity_out=oi,
                    sigma=sig_i,
                    step_index=step_idx,
                    total_steps=total_steps,
                    context=ci,
                    cfg_context_key=ctx_key,
                    producer_stream=stream_ptr,
                )
            out4d[i : i + 1] = oi
            self._last_sigma = sig_i

        if was_3d:
            out3d = out4d.permute(0, 2, 3, 1).reshape(batch, th * tw, in_ch)
            if tail_pad > 0:
                pad = torch.zeros((batch, tail_pad, in_ch), dtype=out3d.dtype, device=out3d.device)
                out3d = torch.cat([out3d, pad], dim=1)
            sample = out3d.to(dtype=orig_dtype)
        else:
            sample = out4d.to(dtype=orig_dtype)

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)
