"""Drop-in LTX-2.5 joint audio-video 3D transformer backed by the QuantFunc PC Engine."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from quantfunc.core.checkpoint import PresetMetadata, sniff_safetensors_header
from quantfunc.core.registry import register_model
from quantfunc.models.base import (
    QuantFuncBaseTransformer,
    Transformer2DModelOutput,
    _torch_dtype_to_qf,
)

logger = logging.getLogger(__name__)

_RANK_1D = 1
_RANK_2D = 2
_RANK_3D = 3
_RANK_4D = 4
_RANK_5D = 5
_FRAME_SCALE_VARIANCE_TOL = 1e-4
_FULL_DENOISE_EPS = 0.9999

_VIDEO_CONNECTOR_MARKERS = (
    "video_embeddings_connector.",
    "video_connector.",
)
_AUDIO_CONNECTOR_MARKERS = (
    "audio_embeddings_connector.",
    "audio_connector.",
)


def _header_has_any_marker(header: Mapping[str, Any], markers: tuple[str, ...]) -> bool:
    return any(any(m in key for m in markers) for key in header if key != "__metadata__")


def _file_has_both_connectors(path: str | os.PathLike[str]) -> tuple[bool, bool, bool]:
    """Return (has_both, has_video, has_audio) by reading only the safetensors JSON header."""
    try:
        header = sniff_safetensors_header(path)
    except Exception:
        return (False, False, False)
    has_vid = _header_has_any_marker(header, _VIDEO_CONNECTOR_MARKERS)
    has_aud = _header_has_any_marker(header, _AUDIO_CONNECTOR_MARKERS)
    return (has_vid and has_aud, has_vid, has_aud)


def _discover_connectors_path(
    weights_path: str | None,
    explicit_connectors_path: str | os.PathLike[str] | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Locate LTX-2.5 dual-modality connectors (explicit -> weights self -> sibling *connector*.safetensors)."""
    if explicit_connectors_path is not None:
        exp_str = str(explicit_connectors_path)
        if not os.path.isfile(exp_str):
            raise FileNotFoundError(f"LTX-2.5 connectors_path does not exist: {exp_str}")
        has_both, has_vid, has_aud = _file_has_both_connectors(exp_str)
        if not has_both:
            raise ValueError(
                f"LTX-2.5 connectors_path {exp_str!r} must contain BOTH video and audio connectors "
                f"(video={has_vid}, audio={has_aud})"
            )
        return str(Path(exp_str).resolve())

    if weights_path and os.path.isfile(weights_path):
        has_both, has_vid, has_aud = _file_has_both_connectors(weights_path)
        if has_both:
            return str(Path(weights_path).resolve())

        parent = Path(weights_path).resolve().parent
        try:
            siblings = sorted(
                p
                for p in parent.iterdir()
                if p.is_file()
                and p.name.lower().endswith(".safetensors")
                and "connector" in p.name.lower()
                and str(p.resolve()) != str(Path(weights_path).resolve())
            )
        except OSError:
            siblings = []

        for sib in siblings:
            sib_both, _, _ = _file_has_both_connectors(sib)
            if sib_both:
                return str(sib.resolve())

        if required:
            detail = (
                "packs video connector but NOT audio connector"
                if (has_vid and not has_aud)
                else "contains no dual-modality connectors"
            )
            raise RuntimeError(
                f"LTX-2.5 av_unprocessed_ctx=True requires connectors_path, but "
                f"{os.path.basename(weights_path)!r} {detail} and no qualifying "
                f"*connector*.safetensors sibling was found in {str(parent)!r}"
            )
    elif required:
        raise RuntimeError(
            "LTX-2.5 av_unprocessed_ctx=True requires connectors_path containing BOTH video and audio connectors"
        )
    return None


class _LTX25RopeStub(nn.Module):
    """Zero-weight RoPE attribute stub for upstream diffusers LTX-2.5 pipeline introspection."""

    def __init__(self, config: Mapping[str, Any], *, is_audio: bool = False) -> None:
        super().__init__()
        self.is_audio = bool(is_audio)
        self.theta = float(config.get("rope_theta", config.get("positional_embedding_theta", 0.0)) or 0.0)
        self.use_middle_indices_grid = bool(config.get("use_middle_indices_grid", True))
        self.scale_factors = tuple(config.get("vae_scale_factors", ()))

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (args, kwargs)
        raise RuntimeError(
            "QuantFunc LTX-2.5 computes 3D/1D RoPE internally inside the engine; "
            "do not call rope stub directly."
        )


def _normalize_stg_blocks(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str | bytes):
        raise ValueError(f"spatio_temporal_guidance_blocks must be a sequence of ints, got {raw!r}")
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(
                f"spatio_temporal_guidance_blocks elements must be non-negative ints, got {item!r}"
            )
        out.append(int(item))
    return tuple(sorted(set(out)))


def _normalize_ref_audio_payload(payload: Any) -> torch.Tensor | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        if set(payload.keys()) != {"tokens"}:
            raise ValueError("LTX-2.5 ref_audio dict payload must contain exactly {'tokens'}")
        return payload["tokens"]
    if torch.is_tensor(payload):
        return payload
    raise ValueError(f"LTX-2.5 ref_audio must be a Tensor or {{'tokens': Tensor}}, got {type(payload).__name__}")


def restore_frozen_audio(
    stage2_audio_latents: torch.Tensor,
    stage1_audio_latents: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Restore Stage-1 frozen audio latents where mask == 0, keeping Stage-2 where mask > 0."""
    if not (
        torch.is_tensor(stage2_audio_latents)
        and torch.is_tensor(stage1_audio_latents)
        and torch.is_tensor(mask)
    ):
        raise TypeError("restore_frozen_audio requires Tensor arguments for stage2, stage1, and mask")
    if tuple(stage2_audio_latents.shape) != tuple(stage1_audio_latents.shape):
        raise ValueError(
            f"restore_frozen_audio shape mismatch: stage2={tuple(stage2_audio_latents.shape)} "
            f"vs stage1={tuple(stage1_audio_latents.shape)}"
        )
    if not bool(torch.isfinite(mask).all().item()):
        raise ValueError("restore_frozen_audio mask must be finite")

    m = mask
    target_shape = tuple(int(d) for d in stage2_audio_latents.shape)
    if tuple(m.shape) != target_shape:
        if stage2_audio_latents.ndim == _RANK_4D:
            b, _, la, _ = target_shape
            if m.ndim == _RANK_1D and int(m.shape[0]) == la:
                m = m.reshape(1, 1, la, 1)
            elif m.ndim == _RANK_2D and tuple(int(d) for d in m.shape) == (b, la):
                m = m.reshape(b, 1, la, 1)
        elif stage2_audio_latents.ndim == _RANK_3D:
            b, la, _ = target_shape
            if m.ndim == _RANK_1D and int(m.shape[0]) == la:
                m = m.reshape(1, la, 1)
            elif m.ndim == _RANK_2D and tuple(int(d) for d in m.shape) == (b, la):
                m = m.reshape(b, la, 1)

    try:
        frozen = torch.broadcast_to(m == 0, target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"restore_frozen_audio mask shape {tuple(mask.shape)} is not broadcastable to {target_shape}"
        ) from exc

    if not bool(frozen.any().item()):
        return stage2_audio_latents
    clean = stage1_audio_latents.to(
        device=stage2_audio_latents.device,
        dtype=stage2_audio_latents.dtype,
    )
    return torch.where(frozen.to(device=stage2_audio_latents.device), clean, stage2_audio_latents)


@register_model(
    arch="ltx25",
    pipeline_classes=("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTXPipeline"),
    transformer_classes=("AVTransformer3DModel", "LTX2VideoTransformer3DModel", "LTXVideoTransformer3DModel"),
    engine_arch="ltx25",
)
class QuantFuncLTX25Transformer3DModel(QuantFuncBaseTransformer):
    """Drop-in LTX-2.5 joint audio-video transformer with dual ctx, I2V scales, ref_audio, and STG guard."""

    model_arch: str = "ltx25"

    def __init__(
        self,
        *,
        connectors_path: str | os.PathLike[str] | None = None,
        av_unprocessed_ctx: bool = False,
        allow_stg_session_reset: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rope = _LTX25RopeStub(self.config, is_audio=False)
        self.audio_rope = _LTX25RopeStub(self.config, is_audio=True)

        extra_conn = self._extra_create_config.get("connectors_path")
        self._connectors_path: str | None = (
            str(connectors_path)
            if connectors_path is not None
            else (str(extra_conn) if extra_conn is not None else None)
        )
        self._av_unprocessed_ctx: bool = bool(av_unprocessed_ctx)
        self.allow_stg_session_reset: bool = bool(allow_stg_session_reset)

        self._ref_audio_state: tuple[torch.Tensor, tuple[int, ...], torch.dtype, torch.Tensor, torch.Tensor] | None = (
            None
        )
        self._ref_audio_bound: bool = False
        self._bound_stg_blocks: tuple[int, ...] = ()
        self._bound_disable_av_cross_attn: bool = False

    @classmethod
    def _build_extra_create_config(
        cls,
        *,
        weights_path: str | None,
        config: Mapping[str, Any],
        config_path: str | None,
        preset: PresetMetadata | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        _ = (config, config_path, preset)
        explicit_conn = kwargs.get("connectors_path")
        av_unproc = bool(kwargs.get("av_unprocessed_ctx", False))
        require_conn = bool(kwargs.pop("require_connectors", False)) or av_unproc
        resolved_conn = _discover_connectors_path(
            weights_path,
            explicit_conn,
            required=require_conn,
        )
        extra: dict[str, Any] = {}
        if resolved_conn is not None:
            extra["connectors_path"] = resolved_conn
            kwargs["connectors_path"] = resolved_conn
        return extra

    # ── Dynamic geometry helpers (zero hardcoded constants) ─────────────────

    def _video_channels(self) -> int:
        val = self.config.get("in_channels")
        if val is None or int(val) <= 0:
            raise ValueError(f"LTX-2.5 config missing positive in_channels: {val!r}")
        return int(val)

    def _audio_packed_channels(self) -> int:
        val = self.config.get("audio_in_channels")
        if val is None:
            val = self.config.get("audio_out_channels")
        if val is not None and int(val) > 0:
            return int(val)
        ch, mel = self._audio_unpacked_geometry()
        return ch * mel

    def _audio_unpacked_geometry(self) -> tuple[int, int]:
        ch = getattr(self._preset, "audio_latent_channels", None) if self._preset is not None else None
        mel = getattr(self._preset, "audio_mel_bins", None) if self._preset is not None else None
        if ch is None:
            ch = self.config.get("audio_latent_channels")
        if mel is None:
            mel = self.config.get("audio_mel_bins")
        if ch is None or mel is None or int(ch) <= 0 or int(mel) <= 0:
            raise ValueError(
                f"LTX-2.5 preset/config missing positive audio_latent_channels/audio_mel_bins: ({ch!r}, {mel!r})"
            )
        return int(ch), int(mel)

    def _context_dims(self) -> tuple[int, int]:
        vc = self.config.get("cross_attention_dim")
        ac = self.config.get("audio_cross_attention_dim")
        if vc is None or int(vc) <= 0 or ac is None or int(ac) <= 0:
            raise ValueError(
                f"LTX-2.5 config missing positive cross_attention_dim/audio_cross_attention_dim: ({vc!r}, {ac!r})"
            )
        return int(vc), int(ac)

    def _default_fps(self) -> float:
        fps = getattr(self._preset, "default_fps", None) if self._preset is not None else None
        if fps is None:
            fps = self.config.get("default_fps")
        if fps is None or float(fps) <= 0.0:
            raise ValueError(f"LTX-2.5 preset/config missing positive default_fps: {fps!r}")
        return float(fps)

    # ── Reference audio lifecycle ───────────────────────────────────────────

    def _validate_ref_audio_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        packed_ch = self._audio_packed_channels()
        aud_ch, mel_bins = self._audio_unpacked_geometry()
        if not torch.is_tensor(tokens):
            raise ValueError("LTX-2.5 ref_audio tokens must be a Tensor")
        if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"LTX-2.5 ref_audio needs float16/bfloat16/float32, got {tokens.dtype}")
        if not bool(torch.isfinite(tokens).all().item()):
            raise ValueError("LTX-2.5 ref_audio tokens must be finite")

        if tokens.ndim == _RANK_4D:
            if (
                int(tokens.shape[0]) != 1
                or int(tokens.shape[1]) != aud_ch
                or int(tokens.shape[2]) <= 0
                or int(tokens.shape[3]) != mel_bins
            ):
                raise ValueError(
                    f"LTX-2.5 4D ref_audio must be [1, {aud_ch}, La, {mel_bins}], got {tuple(tokens.shape)}"
                )
            return tokens.permute(0, 2, 1, 3).reshape(1, int(tokens.shape[2]), packed_ch).contiguous()

        if tokens.ndim == _RANK_3D:
            if int(tokens.shape[0]) != 1 or int(tokens.shape[1]) <= 0 or int(tokens.shape[2]) != packed_ch:
                raise ValueError(
                    f"LTX-2.5 3D ref_audio must be [1, La, {packed_ch}] with La > 0, got {tuple(tokens.shape)}"
                )
            return tokens

        raise ValueError(f"LTX-2.5 ref_audio must be 3D or 4D, got {tokens.ndim}D")

    def set_reference_audio(self, ref_audio: torch.Tensor | Mapping[str, Any] | None) -> None:
        """Bind or clear reference audio tokens (fail-closed if called mid-session)."""
        if self._session_is_open():
            raise RuntimeError(
                "LTX-2.5 ref_audio cannot be modified while a session is open; close the session first"
            )
        if self._handle.reference_audio_pending():
            raise RuntimeError("LTX-2.5 previous reference audio begin has pending copy completion")
        raw = _normalize_ref_audio_payload(ref_audio)
        if raw is None:
            self._handle.retain_reference_audio(None)
            self._ref_audio_state = None
            self._ref_audio_bound = False
            return
        validated = self._validate_ref_audio_tensor(raw)
        snap = validated.detach().to(device="cpu").clone()
        upload = snap.to(dtype=self.dtype).contiguous()
        if not bool(torch.isfinite(upload).all().item()):
            raise ValueError("LTX-2.5 normalized ref_audio overflowed target dtype")
        self._ref_audio_state = (raw, tuple(int(d) for d in validated.shape), validated.dtype, snap, upload)
        self._ref_audio_bound = False

    def _reconcile_ref_audio_for_call(
        self,
        ref_audio_arg: Any,
        device: torch.device,
        opening: bool,
    ) -> bool:
        raw = _normalize_ref_audio_payload(ref_audio_arg)
        if raw is None:
            if self._ref_audio_state is None:
                return False
            if opening:
                # Re-upload bound ref audio onto active execution device.
                src, shape, dtype, snap, _ = self._ref_audio_state
                upload = snap.to(device=device, dtype=self.dtype).contiguous()
                self._ref_audio_state = (src, shape, dtype, snap, upload)
                self._ref_audio_bound = False
            return True

        validated = self._validate_ref_audio_tensor(raw)
        if not opening:
            state = self._ref_audio_state
            if state is None or not self._ref_audio_bound:
                raise RuntimeError("LTX-2.5 ref_audio binding requires starting a new session")
            src, shape, dtype, snap, _ = state
            if (
                raw is not src
                or tuple(int(d) for d in validated.shape) != shape
                or validated.dtype != dtype
                or not torch.equal(validated.detach().to(device="cpu"), snap)
            ):
                raise RuntimeError("LTX-2.5 ref_audio changed mid-session; start a new session")
            return True

        if self._handle.reference_audio_pending():
            raise RuntimeError("LTX-2.5 previous reference audio copy is still pending")
        snap = validated.detach().to(device="cpu").clone()
        upload = snap.to(device=device, dtype=self.dtype).contiguous()
        if not bool(torch.isfinite(upload).all().item()):
            raise ValueError("LTX-2.5 normalized ref_audio overflowed target dtype")
        self._ref_audio_state = (raw, tuple(int(d) for d in validated.shape), validated.dtype, snap, upload)
        self._ref_audio_bound = False
        return True

    def _end_session_if_open(self) -> tuple[bool, bool]:
        was_open, ok = super()._end_session_if_open()
        if ok and not self._handle.reference_audio_pending():
            self._ref_audio_bound = False
        self._bound_stg_blocks = ()
        self._bound_disable_av_cross_attn = False
        return was_open, ok

    # ── Latent-level frozen audio restoration ───────────────────────────────

    @staticmethod
    def restore_frozen_audio(
        stage2_audio_latents: torch.Tensor,
        stage1_audio_latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Restore Stage-1 frozen audio latents where mask == 0, keeping Stage-2 where mask > 0."""
        return restore_frozen_audio(stage2_audio_latents, stage1_audio_latents, mask)

    # ── Geometry, context & timestep scale extraction ───────────────────────

    def _unpack_video_latent(
        self,
        hidden_states: torch.Tensor,
        *,
        num_frames: int | None,
        height: int | None,
        width: int | None,
    ) -> tuple[torch.Tensor, bool, int, int, int, int, int, int]:
        """Return (vid_tokens_3d, was_5d, batch, ppf, pph, ppw, pixel_h, pixel_w)."""
        vid_ch = self._video_channels()
        spatial_scale = self._spatial_scale()
        temporal_scale = self._temporal_scale()

        if hidden_states.ndim == _RANK_5D:
            batch, ch, ppf, pph, ppw = (int(d) for d in hidden_states.shape)
            if ch != vid_ch:
                raise ValueError(
                    f"LTX-2.5 5D hidden_states channels={ch} must match config.in_channels={vid_ch}"
                )
            n_tok = ppf * pph * ppw
            tokens = hidden_states.reshape(batch, ch, n_tok).transpose(1, 2).contiguous()
            return tokens, True, batch, ppf, pph, ppw, pph * spatial_scale, ppw * spatial_scale

        if hidden_states.ndim == _RANK_3D:
            batch, n_tok, ch = (int(d) for d in hidden_states.shape)
            if ch != vid_ch:
                raise ValueError(
                    f"LTX-2.5 3D hidden_states channels={ch} must match config.in_channels={vid_ch}"
                )
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "LTX-2.5 3D hidden_states requires num_frames, height, and width"
                )
            nf = int(num_frames)
            h_arg = int(height)
            w_arg = int(width)
            if nf <= 0 or h_arg <= 0 or w_arg <= 0:
                raise ValueError(
                    f"LTX-2.5 num_frames/height/width must be positive, got ({nf}, {h_arg}, {w_arg})"
                )

            if nf * h_arg * w_arg == n_tok:
                ppf, pph, ppw = nf, h_arg, w_arg
            else:
                cand_f = (nf - 1) // temporal_scale + 1 if (nf - 1) % temporal_scale == 0 else nf
                cand_h = h_arg // spatial_scale if h_arg % spatial_scale == 0 else h_arg
                cand_w = w_arg // spatial_scale if w_arg % spatial_scale == 0 else w_arg
                if cand_f * cand_h * cand_w == n_tok:
                    ppf, pph, ppw = cand_f, cand_h, cand_w
                elif nf * cand_h * cand_w == n_tok:
                    ppf, pph, ppw = nf, cand_h, cand_w
                else:
                    raise ValueError(
                        f"LTX-2.5 cannot reconcile 3D token count {n_tok} with "
                        f"num_frames={nf}, height={h_arg}, width={w_arg}"
                    )
            return hidden_states.contiguous(), False, batch, ppf, pph, ppw, pph * spatial_scale, ppw * spatial_scale

        raise ValueError(f"LTX-2.5 hidden_states must be 3D or 5D, got {hidden_states.ndim}D")

    def _unpack_audio_latent(
        self,
        audio_hidden_states: torch.Tensor | None,
        batch: int,
    ) -> tuple[torch.Tensor | None, bool, int]:
        """Return (aud_3d, was_4d, audio_rows)."""
        if audio_hidden_states is None:
            return None, False, 0
        packed_ch = self._audio_packed_channels()
        aud_ch, mel_bins = self._audio_unpacked_geometry()

        if audio_hidden_states.ndim == _RANK_4D:
            ab, ca, la, mel = (int(d) for d in audio_hidden_states.shape)
            if ab != batch or ca != aud_ch or mel != mel_bins or la <= 0:
                raise ValueError(
                    f"LTX-2.5 4D audio_hidden_states shape {tuple(audio_hidden_states.shape)} "
                    f"must be [{batch}, {aud_ch}, La>0, {mel_bins}]"
                )
            packed = audio_hidden_states.permute(0, 2, 1, 3).reshape(batch, la, packed_ch).contiguous()
            return packed, True, la

        if audio_hidden_states.ndim == _RANK_3D:
            ab, la, ch = (int(d) for d in audio_hidden_states.shape)
            if ab != batch or la <= 0 or ch != packed_ch:
                raise ValueError(
                    f"LTX-2.5 3D audio_hidden_states shape {tuple(audio_hidden_states.shape)} "
                    f"must be [{batch}, La>0, {packed_ch}]"
                )
            return audio_hidden_states.contiguous(), False, la

        raise ValueError(f"LTX-2.5 audio_hidden_states must be 3D or 4D, got {audio_hidden_states.ndim}D")

    def _prepare_context(
        self,
        encoder_hidden_states: torch.Tensor | None,
        audio_encoder_hidden_states: torch.Tensor | None,
        batch: int,
        unprocessed: bool,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("LTX-2.5 requires encoder_hidden_states")
        if encoder_hidden_states.ndim != _RANK_3D or int(encoder_hidden_states.shape[0]) != batch:
            raise ValueError(
                f"LTX-2.5 encoder_hidden_states must be 3D with batch={batch}, "
                f"got {tuple(encoder_hidden_states.shape)}"
            )

        if unprocessed:
            if self._connectors_path is None:
                raise RuntimeError(
                    "LTX-2.5 av_unprocessed_ctx=True requires a discovered or configured connectors_path"
                )
            return encoder_hidden_states.contiguous()

        vc_dim, ac_dim = self._context_dims()
        if audio_encoder_hidden_states is not None:
            if audio_encoder_hidden_states.ndim != _RANK_3D:
                raise ValueError(
                    f"LTX-2.5 audio_encoder_hidden_states must be 3D, got {audio_encoder_hidden_states.ndim}D"
                )
            if (
                int(audio_encoder_hidden_states.shape[0]) != batch
                or int(audio_encoder_hidden_states.shape[1]) != int(encoder_hidden_states.shape[1])
            ):
                raise ValueError(
                    f"LTX-2.5 audio_encoder_hidden_states shape {tuple(audio_encoder_hidden_states.shape)} "
                    f"must match batch and seq_len of encoder_hidden_states {tuple(encoder_hidden_states.shape)}"
                )
            if int(encoder_hidden_states.shape[2]) != vc_dim or int(audio_encoder_hidden_states.shape[2]) != ac_dim:
                raise ValueError(
                    f"LTX-2.5 processed context dims ({int(encoder_hidden_states.shape[2])}, "
                    f"{int(audio_encoder_hidden_states.shape[2])}) must match ({vc_dim}, {ac_dim})"
                )
            aud_aligned = audio_encoder_hidden_states.to(
                device=encoder_hidden_states.device,
                dtype=encoder_hidden_states.dtype,
            )
            return torch.cat([encoder_hidden_states, aud_aligned], dim=-1).contiguous()

        ctx_dim = int(encoder_hidden_states.shape[2])
        if ctx_dim not in (vc_dim, vc_dim + ac_dim):
            raise ValueError(
                f"LTX-2.5 processed encoder_hidden_states last dim={ctx_dim} "
                f"must be {vc_dim} or {vc_dim + ac_dim}"
            )
        return encoder_hidden_states.contiguous()

    def _extract_video_sigma_and_frame_scales(
        self,
        timestep: Any,
        sigma_override: Any,
        batch: int,
        ppf: int,
        pph: int,
        ppw: int,
    ) -> tuple[list[float], list[float] | None]:
        """Extract normalized per-batch sigmas and optional per-frame I2V frame_t_scales."""
        if sigma_override is not None and timestep is None:
            return self._extract_batch_sigmas(sigma_override, batch), None

        if not torch.is_tensor(timestep):
            base = self._normalize_scalar_sigma(
                float(sigma_override) if sigma_override is not None else float(timestep)
            )
            return [base] * batch, None

        t_det = timestep.detach().float()
        if not bool(torch.isfinite(t_det).all().item()):
            raise ValueError("LTX-2.5 video timestep must be finite")
        if bool((t_det < 0.0).any().item()):
            raise ValueError("LTX-2.5 video timestep must be non-negative")

        if t_det.ndim <= _RANK_1D:
            raw_sigmas = self._extract_batch_sigmas(
                sigma_override if sigma_override is not None else t_det,
                batch,
            )
            return raw_sigmas, None

        # Multi-dimensional video_timestep: [B, N_vid], [B, F], or [B, F, H, W] / [B, C, F, H, W]
        if int(t_det.shape[0]) != batch:
            raise ValueError(
                f"LTX-2.5 multi-dim video_timestep batch {int(t_det.shape[0])} != {batch}"
            )
        sigmas_out: list[float] = []
        per_batch_scales: list[list[float]] = []
        frame_tokens = pph * ppw

        for b in range(batch):
            tb = t_det[b]
            if tb.ndim == _RANK_4D:
                # [C, F, H, W] -> [F, C*H*W]
                tb_frames = tb.movedim(1, 0).reshape(ppf, -1)
            elif tb.numel() == ppf * frame_tokens:
                tb_frames = tb.reshape(ppf, frame_tokens)
            elif tb.numel() == ppf:
                tb_frames = tb.reshape(ppf, 1)
            else:
                raise ValueError(
                    f"LTX-2.5 video_timestep per-item shape {tuple(tb.shape)} does not match "
                    f"ppf={ppf} or ppf*pph*ppw={ppf * frame_tokens}"
                )

            raw_max = float(tb_frames.max().item())
            if sigma_override is not None:
                ov = float(sigma_override[b].item()) if torch.is_tensor(sigma_override) and sigma_override.numel() > 1 else float(sigma_override)
                sig_norm = self._normalize_scalar_sigma(ov)
                scale_denom = ov if ov > 0.0 else raw_max
            else:
                sig_norm = self._normalize_scalar_sigma(raw_max)
                scale_denom = raw_max
            sigmas_out.append(sig_norm)

            scales_b: list[float] = []
            for f in range(ppf):
                row = tb_frames[f]
                if scale_denom > 0.0:
                    sc_row = row / scale_denom
                else:
                    sc_row = torch.ones_like(row)
                lo = float(sc_row.min().item())
                hi = float(sc_row.max().item())
                if hi - lo > _FRAME_SCALE_VARIANCE_TOL:
                    raise ValueError(
                        f"LTX-2.5 video_timestep varies within latent frame {f} "
                        f"(spread={hi - lo:.6g} > {_FRAME_SCALE_VARIANCE_TOL}); per-frame scales only"
                    )
                scales_b.append(max(0.0, min(1.0, hi)))
            per_batch_scales.append(scales_b)

        if batch > 1 and any(s != per_batch_scales[0] for s in per_batch_scales[1:]):
            raise ValueError("LTX-2.5 video_timestep per-frame scales must match across batch items")

        final_scales = per_batch_scales[0]
        if all(v >= _FULL_DENOISE_EPS for v in final_scales):
            return sigmas_out, None
        return sigmas_out, final_scales

    def _extract_audio_frame_scales(
        self,
        audio_timestep: Any,
        video_sigmas: list[float],
        batch: int,
        audio_rows: int,
    ) -> list[float] | None:
        if audio_timestep is None or audio_rows <= 0:
            return None
        if not torch.is_tensor(audio_timestep):
            _ = self._normalize_scalar_sigma(float(audio_timestep))
            return None

        at = audio_timestep.detach().float()
        if not bool(torch.isfinite(at).all().item()):
            raise ValueError("LTX-2.5 audio_timestep must be finite")
        if bool((at < 0.0).any().item()):
            raise ValueError("LTX-2.5 audio_timestep must be non-negative")
        if at.ndim <= _RANK_1D:
            for val in at.reshape(-1).tolist():
                _ = self._normalize_scalar_sigma(float(val))
            return None

        if int(at.shape[0]) != batch:
            raise ValueError(
                f"LTX-2.5 audio_timestep batch {int(at.shape[0])} != {batch}"
            )
        aud_ch, mel_bins = self._audio_unpacked_geometry()
        per_batch_scales: list[list[float]] = []

        for b in range(batch):
            ab = at[b]
            if ab.ndim == _RANK_3D and tuple(int(d) for d in ab.shape) == (aud_ch, audio_rows, mel_bins):
                ab_rows = ab.movedim(1, 0).reshape(audio_rows, aud_ch * mel_bins)
            elif ab.ndim == _RANK_2D and int(ab.shape[0]) == audio_rows:
                ab_rows = ab
            elif ab.numel() == audio_rows:
                ab_rows = ab.reshape(audio_rows, 1)
            elif ab.numel() % audio_rows == 0:
                ab_rows = ab.reshape(audio_rows, ab.numel() // audio_rows)
            else:
                raise ValueError(
                    f"LTX-2.5 audio_timestep per-item shape {tuple(ab.shape)} does not align with audio_rows={audio_rows}"
                )

            raw_max = float(ab_rows.max().item())
            denom = raw_max if raw_max > 0.0 else video_sigmas[b]
            scales_b: list[float] = []
            for r in range(audio_rows):
                row = ab_rows[r]
                sc_row = (row / denom) if denom > 0.0 else torch.ones_like(row)
                lo = float(sc_row.min().item())
                hi = float(sc_row.max().item())
                if hi - lo > _FRAME_SCALE_VARIANCE_TOL:
                    raise ValueError(
                        f"LTX-2.5 audio_timestep varies within audio row {r} "
                        f"(spread={hi - lo:.6g} > {_FRAME_SCALE_VARIANCE_TOL}); per-audio-frame scales only"
                    )
                scales_b.append(max(0.0, min(1.0, hi)))
            per_batch_scales.append(scales_b)

        if batch > 1 and any(s != per_batch_scales[0] for s in per_batch_scales[1:]):
            raise ValueError("LTX-2.5 audio_timestep per-row scales must match across batch items")

        final_scales = per_batch_scales[0]
        if all(v >= _FULL_DENOISE_EPS for v in final_scales):
            return None
        return final_scales

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        audio_encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | float | int | None = None,
        audio_timestep: torch.Tensor | float | int | None = None,
        sigma: torch.Tensor | float | int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        av_unprocessed_ctx: bool | None = None,
        ref_audio: torch.Tensor | Mapping[str, Any] | None = None,
        spatio_temporal_guidance_blocks: Sequence[int] | None = None,
        stg_blocks: Sequence[int] | None = None,
        isolate_modalities: bool | None = None,
        disable_av_cross_attn: bool | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor, ...]:
        """Run one LTX-2.5 AV denoising step via qf_xfm_step_av."""
        if "video_timestep" in kwargs and timestep is None:
            timestep = kwargs.pop("video_timestep")
        if timestep is None and sigma is None:
            raise ValueError("LTX-2.5 forward requires timestep or sigma")

        eff_unprocessed = self._av_unprocessed_ctx if av_unprocessed_ctx is None else bool(av_unprocessed_ctx)
        raw_stg = (
            spatio_temporal_guidance_blocks
            if spatio_temporal_guidance_blocks is not None
            else stg_blocks
        )
        cur_stg = _normalize_stg_blocks(raw_stg)
        if isolate_modalities is not None and disable_av_cross_attn is not None:
            if bool(isolate_modalities) != bool(disable_av_cross_attn):
                raise ValueError("Conflicting values for isolate_modalities and disable_av_cross_attn")
        cur_disable_av = bool(
            isolate_modalities if isolate_modalities is not None else (disable_av_cross_attn or False)
        )

        orig_vid_dtype = hidden_states.dtype
        orig_aud_dtype = audio_hidden_states.dtype if audio_hidden_states is not None else orig_vid_dtype
        dev = hidden_states.device

        vid_3d, vid_was_5d, batch, ppf, pph, ppw, pixel_h, pixel_w = self._unpack_video_latent(
            hidden_states,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        aud_3d, aud_was_4d, audio_rows = self._unpack_audio_latent(audio_hidden_states, batch)
        ctx = self._prepare_context(
            encoder_hidden_states,
            audio_encoder_hidden_states,
            batch,
            eff_unprocessed,
        )

        sigmas, frame_t_scales = self._extract_video_sigma_and_frame_scales(
            timestep, sigma, batch, ppf, pph, ppw
        )
        audio_frame_t_scales = self._extract_audio_frame_scales(
            audio_timestep, sigmas, batch, audio_rows
        )

        temporal_scale = self._temporal_scale()
        eff_num_frames = (
            int(num_frames)
            if (num_frames is not None and not vid_was_5d and int(num_frames) == (ppf - 1) * temporal_scale + 1)
            else ((ppf - 1) * temporal_scale + 1)
        )
        eff_fps = self._default_fps() if fps is None else float(fps)
        if not math.isfinite(eff_fps) or eff_fps <= 0.0:
            raise ValueError(f"LTX-2.5 fps must be finite and positive, got {eff_fps!r}")

        # Mid-session STG / isolate_modalities guard.
        if self._session_is_open() and (
            cur_stg != self._bound_stg_blocks or cur_disable_av != self._bound_disable_av_cross_attn
        ):
            if not self.allow_stg_session_reset:
                raise RuntimeError(
                    "LTX-2.5 spatio_temporal_guidance_blocks / isolate_modalities cannot change mid-session "
                    "(set allow_stg_session_reset=True to safely close and reopen the session)"
                )
            self._end_session_if_open()

        opening = not self._session_is_open()
        ref_active = self._reconcile_ref_audio_for_call(ref_audio, dev, opening)

        act_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16, torch.float32) else orig_vid_dtype
        vid_in = vid_3d if vid_3d.dtype == act_dtype else vid_3d.to(dtype=act_dtype)
        ctx_in = ctx if (ctx.dtype == act_dtype and ctx.device == dev) else ctx.to(device=dev, dtype=act_dtype)

        vid_ch = self._video_channels()
        aud_pack = self._audio_packed_channels()
        aud_ch, mel_bins = self._audio_unpacked_geometry()
        out_vid_3d = torch.empty_like(vid_in)
        out_aud_3d = (
            torch.empty((batch, audio_rows, aud_pack), dtype=torch.float32, device=dev)
            if aud_3d is not None
            else None
        )

        geom_key = (
            ppf,
            pph,
            ppw,
            audio_rows,
            eff_unprocessed,
            tuple(frame_t_scales) if frame_t_scales is not None else None,
            tuple(audio_frame_t_scales) if audio_frame_t_scales is not None else None,
            cur_stg,
            cur_disable_av,
            ref_active,
        )
        stream_ptr = self._producer_stream_ptr(vid_in)
        cond_qf_dtype = _torch_dtype_to_qf(ctx_in.dtype)

        for i in range(batch):
            vi = vid_in[i : i + 1].contiguous()
            vo = self._ensure_scratch("ltx25_vid_out", tuple(vi.shape), vi.dtype, dev)
            ci = ctx_in[i : i + 1].contiguous()
            sig_i = sigmas[i]
            ctx_id = self._context_fingerprint(ci, i)

            opts = self._build_base_session_options(
                has_refs=False,
                batch=batch,
                single_cond_group=(batch == 1),
            )
            opts["num_frames"] = int(eff_num_frames)
            opts["fps"] = float(eff_fps)
            opts["av_unprocessed_ctx"] = bool(eff_unprocessed)
            if aud_3d is not None:
                opts["audio_dims"] = [1, int(audio_rows), int(aud_pack)]
            if frame_t_scales is not None:
                opts["frame_t_scales"] = list(frame_t_scales)
            if audio_frame_t_scales is not None:
                opts["audio_frame_t_scales"] = list(audio_frame_t_scales)
            if cur_stg:
                opts["stg_blocks"] = list(cur_stg)
            if cur_disable_av:
                opts["disable_av_cross_attn"] = True

            before_cb = None
            after_cb = None
            if ref_active and self._ref_audio_state is not None:
                upload = self._ref_audio_state[-1]
                opts["av_conds"] = [
                    {
                        "kind": "audio",
                        "audio_latent_ptr": hex(upload.data_ptr()),
                        "audio_latent_dims": [int(d) for d in upload.shape],
                        "audio_latent_dtype": int(_torch_dtype_to_qf(upload.dtype)),
                    }
                ]

                def _before_ref(u: torch.Tensor = upload) -> None:
                    self._handle.retain_reference_audio(u)

                def _after_ref() -> None:
                    self._handle.reference_audio_copy_completed()
                    self._ref_audio_bound = True

                before_cb = _before_ref
                after_cb = _after_ref

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
                allow_busy_retry=not ref_active,
                before_begin=before_cb,
                after_begin=after_cb,
            )
            self._bound_stg_blocks = cur_stg
            self._bound_disable_av_cross_attn = cur_disable_av

            if aud_3d is not None and out_aud_3d is not None:
                ai = aud_3d[i : i + 1].to(device=dev, dtype=torch.float32).contiguous()
                ao = self._ensure_scratch("ltx25_aud_out", (1, audio_rows, aud_pack), torch.float32, dev)
                self._run_step_av(
                    latent_in=vi,
                    velocity_out=vo,
                    audio_latent_in=ai,
                    audio_velocity_out=ao,
                    sigma=sig_i,
                    step_index=step_idx,
                    total_steps=total_steps,
                    context=ci,
                    cfg_context_key=ctx_key,
                    audio_scale=1.0,
                    audio_reference_active=1 if (ref_active and self._ref_audio_bound) else 0,
                    producer_stream=stream_ptr,
                )
                out_aud_3d[i : i + 1] = ao
            else:
                self._run_step(
                    latent_in=vi,
                    velocity_out=vo,
                    sigma=sig_i,
                    step_index=step_idx,
                    total_steps=total_steps,
                    context=ci,
                    cfg_context_key=ctx_key,
                    producer_stream=stream_ptr,
                )
            out_vid_3d[i : i + 1] = vo

        if vid_was_5d:
            vid_out = out_vid_3d.transpose(1, 2).reshape(batch, vid_ch, ppf, pph, ppw).to(dtype=orig_vid_dtype)
        else:
            vid_out = out_vid_3d.to(dtype=orig_vid_dtype)

        if out_aud_3d is not None:
            if aud_was_4d:
                aud_out = (
                    out_aud_3d.reshape(batch, audio_rows, aud_ch, mel_bins)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .to(dtype=orig_aud_dtype)
                )
            else:
                aud_out = out_aud_3d.to(dtype=orig_aud_dtype)
            return (vid_out, aud_out)

        if return_dict:
            return Transformer2DModelOutput(sample=vid_out)
        return (vid_out,)
