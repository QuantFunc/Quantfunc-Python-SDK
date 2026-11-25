"""Drop-in Krea-2 2D transformer backed by the QuantFunc PC Engine."""

from __future__ import annotations

from typing import Any

import torch

from quantfunc.core.registry import register_model
from quantfunc.models.base import (
    QuantFuncBaseTransformer,
    Transformer2DModelOutput,
    _torch_dtype_to_qf,
)

_RANK_3D = 3
_RANK_4D = 4
_POS_AXES_2D = 2
_POS_AXES_3D = 3


def _infer_grid_from_position_ids(
    pos_ids: torch.Tensor,
    num_tokens: int,
) -> tuple[int, int, torch.Tensor | None]:
    """Derive (grid_h, grid_w, perm) from 2D/3D position_ids / img_ids."""
    if not torch.is_tensor(pos_ids):
        raise ValueError(f"position_ids/img_ids must be a Tensor, got {type(pos_ids).__name__}")
    coords = pos_ids.detach()
    if coords.ndim == _RANK_3D:
        coords = coords[0]
    if coords.ndim != _POS_AXES_2D or int(coords.shape[0]) != num_tokens:
        raise ValueError(
            f"Krea-2 position_ids shape {tuple(pos_ids.shape)} does not match token count {num_tokens}"
        )
    last_dim = int(coords.shape[-1])
    if last_dim == _POS_AXES_3D:
        ys = coords[:, 1].long()
        xs = coords[:, 2].long()
    elif last_dim == _POS_AXES_2D:
        ys = coords[:, 0].long()
        xs = coords[:, 1].long()
    else:
        raise ValueError(
            f"Krea-2 position_ids last dim must be 2 or 3, got {last_dim}"
        )

    y_min = int(ys.min().item())
    y_max = int(ys.max().item())
    x_min = int(xs.min().item())
    x_max = int(xs.max().item())
    grid_h = y_max - y_min + 1
    grid_w = x_max - x_min + 1
    if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != num_tokens:
        raise ValueError(
            f"Krea-2 position_ids imply grid ({grid_h}, {grid_w}) whose product != {num_tokens}"
        )

    linear = (ys - y_min) * grid_w + (xs - x_min)
    expected = torch.arange(num_tokens, device=linear.device, dtype=linear.dtype)
    if torch.equal(linear, expected):
        return grid_h, grid_w, None
    sorted_vals = torch.sort(linear).values
    if not torch.equal(sorted_vals, expected):
        raise ValueError("Krea-2 position_ids do not form a bijective 2D grid")
    return grid_h, grid_w, linear


def _infer_grid_from_hw(
    height: int | None,
    width: int | None,
    num_tokens: int,
    patch_size: int,
    vae_scale: int,
) -> tuple[int, int]:
    """Derive (grid_h, grid_w) from explicit height/width kwargs."""
    if height is None or width is None:
        raise ValueError(
            "Krea-2 3D hidden_states requires img_ids/position_ids or explicit (height, width) "
            "to reconstruct a 4D spatial grid for 2D RoPE"
        )
    h = int(height)
    w = int(width)
    if h <= 0 or w <= 0:
        raise ValueError(f"Krea-2 height and width must be positive, got ({h}, {w})")
    if h * w == num_tokens:
        return h, w
    if h % patch_size == 0 and w % patch_size == 0 and (h // patch_size) * (w // patch_size) == num_tokens:
        return h // patch_size, w // patch_size
    full_scale = vae_scale * patch_size
    if h % full_scale == 0 and w % full_scale == 0 and (h // full_scale) * (w // full_scale) == num_tokens:
        return h // full_scale, w // full_scale
    raise ValueError(
        f"Krea-2 cannot map height={h}, width={w} to {num_tokens} tokens "
        f"(patch_size={patch_size}, vae_scale={vae_scale})"
    )


def _compress_middle_padded_context(
    ctx: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compress middle-padded text sequences along bool(mask), failing closed on empty/mismatched batches."""
    if encoder_attention_mask is None:
        return ctx
    if not torch.is_tensor(encoder_attention_mask):
        raise ValueError(
            f"Krea-2 encoder_attention_mask must be a Tensor, got {type(encoder_attention_mask).__name__}"
        )
    batch = int(ctx.shape[0])
    seq_len = int(ctx.shape[1])
    mask = encoder_attention_mask.to(dtype=torch.bool)
    if mask.ndim == 1 and batch == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != _POS_AXES_2D or tuple(mask.shape) != (batch, seq_len):
        raise ValueError(
            f"Krea-2 encoder_attention_mask shape {tuple(encoder_attention_mask.shape)} "
            f"must match (batch={batch}, seq_len={seq_len})"
        )

    counts = [int(mask[b].sum().item()) for b in range(batch)]
    if any(c == 0 for c in counts):
        raise ValueError("Krea-2 encoder_attention_mask has zero valid tokens (all-zero mask)")
    if batch > 1 and len(set(counts)) > 1:
        raise ValueError(
            f"Krea-2 encoder_attention_mask has unequal valid token counts across batch: {counts}"
        )

    valid_len = counts[0]
    if valid_len == seq_len:
        return ctx

    mask_dev = mask.to(device=ctx.device)
    compressed = [ctx[b][mask_dev[b]] for b in range(batch)]
    return torch.stack(compressed, dim=0).contiguous()


@register_model(
    arch="krea2",
    pipeline_classes=("Krea2Pipeline",),
    transformer_classes=("Krea2Transformer2DModel",),
    engine_arch="krea2",
)
class QuantFuncKrea2Transformer2DModel(QuantFuncBaseTransformer):
    """Drop-in Krea-2 transformer with 4D RoPE reconstruction and middle-padded mask support."""

    model_arch: str = "krea2"

    def _in_channels(self) -> int:
        val = self.config.get("in_channels")
        if val is None or int(val) <= 0:
            raise ValueError(f"Krea-2 config missing positive in_channels: {val!r}")
        return int(val)

    def _patch_size(self) -> int:
        val = self.config.get("patch_size")
        if val is None or int(val) <= 0:
            raise ValueError(f"Krea-2 config missing positive patch_size: {val!r}")
        return int(val)

    def _text_geometry(self) -> tuple[int, int]:
        layers = self.config.get("num_text_layers")
        hidden = self.config.get("text_hidden_dim")
        if layers is None or int(layers) <= 0 or hidden is None or int(hidden) <= 0:
            raise ValueError(
                f"Krea-2 config missing positive num_text_layers/text_hidden_dim: ({layers!r}, {hidden!r})"
            )
        return int(layers), int(hidden)

    def _prepare_latent(
        self,
        hidden_states: torch.Tensor,
        *,
        img_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        height: int | None,
        width: int | None,
    ) -> tuple[torch.Tensor, bool, torch.Tensor | None, int, int, bool]:
        """Convert caller latent into 4D [B, C, H, W] for the engine and record restore info."""
        in_ch = self._in_channels()
        patch = self._patch_size()
        vae_scale = self._spatial_scale()
        if patch * patch <= 0 or in_ch % (patch * patch) != 0:
            raise ValueError(
                f"Krea-2 config in_channels={in_ch} is not divisible by patch_size^2={patch * patch}"
            )
        unpatched_ch = in_ch // (patch * patch)

        if hidden_states.ndim == _RANK_3D:
            batch, num_tokens, ch = (int(d) for d in hidden_states.shape)
            if ch != in_ch:
                raise ValueError(
                    f"Krea-2 3D hidden_states channels={ch} must match config.in_channels={in_ch}"
                )
            pos = img_ids if img_ids is not None else position_ids
            perm: torch.Tensor | None = None
            if pos is not None:
                grid_h, grid_w, perm = _infer_grid_from_position_ids(pos, num_tokens)
            else:
                grid_h, grid_w = _infer_grid_from_hw(height, width, num_tokens, patch, vae_scale)

            x_ordered = hidden_states
            if perm is not None:
                buf = torch.empty_like(hidden_states)
                buf[:, perm.to(device=hidden_states.device)] = hidden_states
                x_ordered = buf

            x4d = (
                x_ordered.reshape(batch, grid_h, grid_w, ch)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            return x4d, True, perm, grid_h, grid_w, False

        if hidden_states.ndim == _RANK_4D:
            _, ch, h_in, w_in = (int(d) for d in hidden_states.shape)
            if ch == in_ch:
                return hidden_states.contiguous(), False, None, h_in, w_in, False
            if ch == unpatched_ch:
                if h_in % patch != 0 or w_in % patch != 0:
                    raise ValueError(
                        f"Krea-2 unpatched 4D latent spatial dims ({h_in}, {w_in}) "
                        f"must be divisible by patch_size={patch}"
                    )
                return hidden_states.contiguous(), False, None, h_in // patch, w_in // patch, True
            raise ValueError(
                f"Krea-2 4D hidden_states channels={ch} must be {in_ch} (pre-patched) "
                f"or {unpatched_ch} (unpatched for patch_size={patch})"
            )

        raise ValueError(f"Krea-2 hidden_states must be 3D or 4D, got {hidden_states.ndim}D")

    def _prepare_context(
        self,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        batch: int,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("Krea-2 requires encoder_hidden_states")
        num_layers, text_dim = self._text_geometry()
        expected_flat = num_layers * text_dim

        ctx = encoder_hidden_states
        if ctx.ndim == _RANK_4D:
            b_ctx, seq_len, taps, dim = (int(d) for d in ctx.shape)
            if taps != num_layers or dim != text_dim:
                raise ValueError(
                    f"Krea-2 4D context shape {tuple(ctx.shape)} does not match "
                    f"num_text_layers={num_layers}, text_hidden_dim={text_dim}"
                )
            ctx = ctx.reshape(b_ctx, seq_len, expected_flat)
        elif ctx.ndim == _RANK_3D:
            dim = int(ctx.shape[2])
            if dim not in (expected_flat, text_dim):
                raise ValueError(
                    f"Krea-2 3D context last dim={dim} must be {expected_flat} or {text_dim}"
                )
        else:
            raise ValueError(f"Krea-2 encoder_hidden_states must be 3D or 4D, got {ctx.ndim}D")

        if int(ctx.shape[0]) != batch:
            raise ValueError(
                f"Krea-2 context batch {int(ctx.shape[0])} does not match latent batch {batch}"
            )
        return _compress_middle_padded_context(ctx, encoder_attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | float | int | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Run one Krea-2 denoising step via qf_xfm_step."""
        for forbidden in (
            "audio_hidden_states",
            "ref_latents",
            "reference_latents",
            "controlnet_block_samples",
            "controlnet_single_block_samples",
            "control",
        ):
            if kwargs.get(forbidden) is not None:
                raise ValueError(f"Krea-2 does not support '{forbidden}' conditioning")

        if timestep is None:
            raise ValueError("Krea-2 forward requires timestep")

        orig_dtype = hidden_states.dtype
        dev = hidden_states.device
        x4d, was_3d, perm, grid_h, grid_w, is_unpatched = self._prepare_latent(
            hidden_states,
            img_ids=img_ids,
            position_ids=position_ids,
            height=height,
            width=width,
        )
        batch = int(x4d.shape[0])
        ctx = self._prepare_context(encoder_hidden_states, encoder_attention_mask, batch)
        sigmas = self._extract_batch_sigmas(timestep, batch)

        act_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16, torch.float32) else orig_dtype
        x_in = x4d if x4d.dtype == act_dtype else x4d.to(dtype=act_dtype)
        ctx_in = ctx if (ctx.dtype == act_dtype and ctx.device == dev) else ctx.to(device=dev, dtype=act_dtype)

        patch = self._patch_size()
        vae_scale = self._spatial_scale()
        pixel_h = (int(x_in.shape[2]) * vae_scale) if is_unpatched else (grid_h * patch * vae_scale)
        pixel_w = (int(x_in.shape[3]) * vae_scale) if is_unpatched else (grid_w * patch * vae_scale)

        out4d = torch.empty_like(x_in)
        geom_key = (tuple(int(d) for d in x_in.shape[1:]), int(ctx_in.shape[2]))
        stream_ptr = self._producer_stream_ptr(x_in)
        cond_qf_dtype = _torch_dtype_to_qf(ctx_in.dtype)

        for i in range(batch):
            xi = x_in[i : i + 1].contiguous()
            oi = self._ensure_scratch("krea2_out", tuple(xi.shape), xi.dtype, dev)
            ci = ctx_in[i : i + 1].contiguous()
            sig_i = sigmas[i]
            ctx_id = self._context_fingerprint(ci, i)
            opts = self._build_base_session_options(
                has_refs=False,
                batch=batch,
                single_cond_group=(batch == 1),
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

        if was_3d:
            out3d = out4d.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, int(out4d.shape[1]))
            if perm is not None:
                out3d = out3d[:, perm.to(device=out3d.device)]
            sample = out3d.to(dtype=orig_dtype)
        else:
            sample = out4d.to(dtype=orig_dtype)

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)
