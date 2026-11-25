"""Drop-in MiniMax-H3 joint audio-video 3D transformer backed by the QuantFunc PC Engine."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from quantfunc.core.registry import register_model
from quantfunc.models.base import (
    QuantFuncBaseTransformer,
    Transformer2DModelOutput,
    _parse_to_args,
    _torch_dtype_to_qf,
)

logger = logging.getLogger(__name__)

_RANK_1D = 1
_RANK_3D = 3
_RANK_4D = 4
_RANK_5D = 5
_VALID_REF_KINDS = frozenset({"image", "audio", "video", "video_audio"})


def _derive_text_vision_row_ranges(text_token_tags: Any) -> list[list[int]]:
    """Derive contiguous [start, length] runs where token_tags == 0."""
    if text_token_tags is None:
        return []
    if torch.is_tensor(text_token_tags):
        tags = text_token_tags.detach().reshape(-1).tolist()
    else:
        tags = list(text_token_tags)
    ranges: list[list[int]] = []
    start: int | None = None
    for idx, val in enumerate(tags):
        if int(val) == 0 and start is None:
            start = idx
        elif int(val) != 0 and start is not None:
            ranges.append([start, idx - start])
            start = None
    if start is not None:
        ranges.append([start, len(tags) - start])
    return ranges


def _index_at(seq: Any, idx: int) -> int:
    if torch.is_tensor(seq):
        return int(seq.detach().reshape(-1)[idx].item())
    return int(seq[idx])


def _seq_len(seq: Any) -> int:
    if torch.is_tensor(seq):
        return int(seq.numel())
    return len(seq)


@register_model(
    arch="minimax_h3",
    pipeline_classes=("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"),
    transformer_classes=("MiniMaxH3Transformer3DModel",),
    engine_arch="minimax_h3",
)
class QuantFuncMiniMaxH3Transformer3DModel(QuantFuncBaseTransformer):
    """Drop-in MiniMax-H3 joint AV transformer with packed-row stripping, sign inversion, and av_conds."""

    model_arch: str = "minimax_h3"

    def __init__(
        self,
        *,
        data_direction: bool = True,
        av_sigma_shift_video: float | None = None,
        av_sigma_shift_audio: float | None = None,
        audio_enhance: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._data_direction: bool = bool(data_direction)
        self._av_sigma_shift_video: float | None = (
            float(av_sigma_shift_video) if av_sigma_shift_video is not None else None
        )
        self._av_sigma_shift_audio: float | None = (
            float(av_sigma_shift_audio) if av_sigma_shift_audio is not None else None
        )
        self._audio_enhance: bool = False
        self._audio_enhance_warned: bool = False
        self._av_conditions: dict[str, Any] | None = None
        self._av_cond_version: int = 0
        self._av_cond_device_cache: dict[
            tuple[str, int], tuple[list[dict[str, Any]], list[torch.Tensor]]
        ] = {}

        if audio_enhance:
            self.set_audio_enhance(True)

    # ── Dynamic geometry helpers (zero hardcoded constants) ─────────────────

    def _video_in_channels(self) -> int:
        val = self.config.get("in_channels")
        if val is None:
            val = self.config.get("latents_dim")
        if val is None or int(val) <= 0:
            raise ValueError(f"MiniMax-H3 config missing positive in_channels/latents_dim: {val!r}")
        return int(val)

    def _audio_in_channels(self) -> int:
        val = self.config.get("audio_in_channels")
        if val is None:
            val = self.config.get("audio_latents_dim")
        if val is None and self._preset is not None:
            val = getattr(self._preset, "audio_latent_channels", None)
        if val is None or int(val) <= 0:
            raise ValueError(
                f"MiniMax-H3 config/preset missing positive audio_in_channels/audio_latents_dim: {val!r}"
            )
        return int(val)

    def _audio_stereo_channels(self) -> int:
        val = self.config.get("audio_channels")
        if val is None and self._preset is not None:
            val = getattr(self._preset, "audio_stereo_channels", None)
            if val is None:
                val = getattr(self._preset, "audio_mel_bins", None)
        if val is None or int(val) <= 0:
            raise ValueError(
                f"MiniMax-H3 config/preset missing positive audio_channels/audio_stereo_channels: {val!r}"
            )
        return int(val)

    def _patch_sizes(self) -> tuple[int, int, int]:
        raw = self.config.get("patch_size")
        if not isinstance(raw, list | tuple) or len(raw) != _RANK_3D:
            raise ValueError(f"MiniMax-H3 config patch_size must be [pt, ph, pw], got {raw!r}")
        pt, ph, pw = (int(raw[0]), int(raw[1]), int(raw[2]))
        if pt <= 0 or ph <= 0 or pw <= 0:
            raise ValueError(f"MiniMax-H3 config patch_size dims must be positive, got {(pt, ph, pw)}")
        return pt, ph, pw

    def _text_dim(self) -> int:
        val = self.config.get("text_dim")
        if val is None or int(val) <= 0:
            raise ValueError(f"MiniMax-H3 config missing positive text_dim: {val!r}")
        return int(val)

    def _default_fps(self) -> float:
        fps = getattr(self._preset, "default_fps", None) if self._preset is not None else None
        if fps is None:
            fps = self.config.get("default_fps")
        if fps is None or float(fps) <= 0.0:
            raise ValueError(f"MiniMax-H3 preset/config missing positive default_fps: {fps!r}")
        return float(fps)

    def _frame_grid(self) -> dict[str, int]:
        fg = getattr(self._preset, "frame_grid", None) if self._preset is not None else None
        if fg is None:
            fg = self.config.get("frame_grid")
        if not isinstance(fg, Mapping):
            raise ValueError(f"MiniMax-H3 preset/config missing frame_grid mapping: {fg!r}")
        required = ("base_frames", "frame_stride", "base_latent_t", "latent_t_stride")
        parsed: dict[str, int] = {}
        for k in required:
            v = fg.get(k)
            if v is None or isinstance(v, bool) or int(v) <= 0:
                raise ValueError(f"MiniMax-H3 frame_grid missing positive '{k}': {fg!r}")
            parsed[k] = int(v)
        return parsed

    def _frames_from_latent_t(self, latent_t: int) -> int:
        fg = self._frame_grid()
        base_f = fg["base_frames"]
        stride_f = fg["frame_stride"]
        base_t = fg["base_latent_t"]
        stride_t = fg["latent_t_stride"]
        if latent_t < base_t or (latent_t - base_t) % stride_t != 0:
            raise ValueError(
                f"MiniMax-H3 latent_t={latent_t} does not lie on frame_grid "
                f"(base_latent_t={base_t}, latent_t_stride={stride_t})"
            )
        steps = (latent_t - base_t) // stride_t
        return base_f + steps * stride_f

    def _latent_t_from_frames(self, num_frames: int) -> int:
        fg = self._frame_grid()
        base_f = fg["base_frames"]
        stride_f = fg["frame_stride"]
        base_t = fg["base_latent_t"]
        stride_t = fg["latent_t_stride"]
        if num_frames < base_f or (num_frames - base_f) % stride_f != 0:
            raise ValueError(
                f"MiniMax-H3 num_frames={num_frames} does not lie on frame_grid "
                f"(base_frames={base_f}, frame_stride={stride_f})"
            )
        steps = (num_frames - base_f) // stride_f
        return base_t + steps * stride_t

    def _resolve_sigma_shifts(
        self,
        video_shift_arg: float | None,
        audio_shift_arg: float | None,
    ) -> tuple[float, float]:
        sv = video_shift_arg
        if sv is None:
            sv = self._av_sigma_shift_video
        if sv is None and self._preset is not None:
            sv = getattr(self._preset, "default_video_sigma_shift", None)
        if sv is None:
            sv = self.config.get("default_video_sigma_shift")

        sa = audio_shift_arg
        if sa is None:
            sa = self._av_sigma_shift_audio
        if sa is None and self._preset is not None:
            sa = getattr(self._preset, "default_audio_sigma_shift", None)
        if sa is None:
            sa = self.config.get("default_audio_sigma_shift")

        if sv is None or sa is None:
            raise ValueError(
                f"MiniMax-H3 requires positive av_sigma_shift_video and av_sigma_shift_audio "
                f"(got video={sv!r}, audio={sa!r})"
            )
        if isinstance(sv, bool) or isinstance(sa, bool):
            raise ValueError("MiniMax-H3 av_sigma_shift values must be positive floats, not bool")
        fsv, fsa = float(sv), float(sa)
        if not math.isfinite(fsv) or fsv <= 0.0 or not math.isfinite(fsa) or fsa <= 0.0:
            raise ValueError(
                f"MiniMax-H3 av_sigma_shift_video ({sv!r}) and av_sigma_shift_audio ({sa!r}) "
                f"must both be finite and > 0"
            )
        return fsv, fsa

    # ── Deprecated audio_enhance & clean AV conditions API ──────────────────

    @property
    def audio_enhance(self) -> bool:
        return self._audio_enhance

    @audio_enhance.setter
    def audio_enhance(self, enabled: bool) -> None:
        self.set_audio_enhance(enabled)

    def set_audio_enhance(self, enabled: bool) -> None:
        """Deprecated compatibility switch: warns once when True and NEVER enters options_json."""
        self._audio_enhance = bool(enabled)
        if self._audio_enhance and not self._audio_enhance_warned:
            self._audio_enhance_warned = True
            msg = (
                "MiniMax-H3 audio_enhance is deprecated and ignored by the QuantFunc engine; "
                "it is never forwarded in session options_json."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            logger.warning(msg)

    def set_av_conditions(
        self,
        keyframes: Sequence[Mapping[str, Any]] | None = None,
        refs: Sequence[Mapping[str, Any]] | None = None,
        text_token_tags: Any = None,
    ) -> None:
        """Bind clean (un-noised) visual/audio conditions and optional text vision token tags."""
        in_ch = self._video_in_channels()
        aud_ch = self._audio_in_channels()
        stereo = self._audio_stereo_channels()
        _, ph, pw = self._patch_sizes()

        validated_kf: list[dict[str, Any]] = []
        if keyframes is not None:
            for idx, entry in enumerate(keyframes):
                if not isinstance(entry, Mapping):
                    raise ValueError(f"MiniMax-H3 keyframes[{idx}] must be a mapping")
                lat = entry.get("latent")
                if not torch.is_tensor(lat) or lat.ndim not in (_RANK_4D, _RANK_5D):
                    raise ValueError(
                        f"MiniMax-H3 keyframes[{idx}]['latent'] must be a clean 4D or 5D tensor"
                    )
                if not bool(torch.isfinite(lat).all().item()):
                    raise ValueError(f"MiniMax-H3 keyframes[{idx}]['latent'] must be finite")
                if int(lat.shape[0]) != 1 or int(lat.shape[1]) != in_ch:
                    raise ValueError(
                        f"MiniMax-H3 keyframes[{idx}] latent shape {tuple(lat.shape)} must have B=1, C={in_ch}"
                    )
                if int(lat.shape[-2]) % ph != 0 or int(lat.shape[-1]) % pw != 0:
                    raise ValueError(
                        f"MiniMax-H3 keyframes[{idx}] spatial dims must be divisible by patch ({ph}, {pw})"
                    )
                raw_fidx = entry.get("frame_index", entry.get("resolved_frame_index", 0))
                if isinstance(raw_fidx, bool) or not isinstance(raw_fidx, int):
                    raise ValueError(f"MiniMax-H3 keyframes[{idx}] frame_index must be int, got {raw_fidx!r}")
                validated_kf.append({"latent": lat, "frame_index": 0 if raw_fidx == 0 else -1})

        validated_refs: list[dict[str, Any]] = []
        if refs is not None:
            for idx, entry in enumerate(refs):
                if not isinstance(entry, Mapping):
                    raise ValueError(f"MiniMax-H3 refs[{idx}] must be a mapping")
                kind = entry.get("kind")
                if kind not in _VALID_REF_KINDS:
                    raise ValueError(
                        f"MiniMax-H3 refs[{idx}] kind must be one of {sorted(_VALID_REF_KINDS)}, got {kind!r}"
                    )
                item: dict[str, Any] = {"kind": str(kind)}
                if kind in ("image", "video", "video_audio"):
                    lat = entry.get("latent")
                    if not torch.is_tensor(lat) or lat.ndim not in (_RANK_4D, _RANK_5D):
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] ({kind}) requires a clean 4D or 5D 'latent' tensor"
                        )
                    if not bool(torch.isfinite(lat).all().item()):
                        raise ValueError(f"MiniMax-H3 refs[{idx}] latent must be finite")
                    if int(lat.shape[0]) != 1 or int(lat.shape[1]) != in_ch:
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] latent shape {tuple(lat.shape)} must have B=1, C={in_ch}"
                        )
                    if kind == "image" and lat.ndim == _RANK_5D and int(lat.shape[2]) != 1:
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] image latent must have T=1, got T={int(lat.shape[2])}"
                        )
                    if int(lat.shape[-2]) % ph != 0 or int(lat.shape[-1]) % pw != 0:
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] spatial dims must be divisible by patch ({ph}, {pw})"
                        )
                    item["latent"] = lat
                if kind in ("audio", "video_audio"):
                    alat = entry.get("audio_latent")
                    if not torch.is_tensor(alat) or alat.ndim not in (_RANK_3D, _RANK_4D):
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] ({kind}) requires a clean 3D or 4D 'audio_latent' tensor"
                        )
                    if not bool(torch.isfinite(alat).all().item()):
                        raise ValueError(f"MiniMax-H3 refs[{idx}] audio_latent must be finite")
                    if alat.ndim == _RANK_4D:
                        if (
                            int(alat.shape[0]) != 1
                            or int(alat.shape[1]) != aud_ch
                            or int(alat.shape[2]) != stereo
                            or int(alat.shape[3]) <= 0
                        ):
                            raise ValueError(
                                f"MiniMax-H3 refs[{idx}] 4D audio_latent must be [1, {aud_ch}, {stereo}, Ta>0], "
                                f"got {tuple(alat.shape)}"
                            )
                    elif int(alat.shape[0]) != 1 or int(alat.shape[2]) != aud_ch or int(alat.shape[1]) <= 0:
                        raise ValueError(
                            f"MiniMax-H3 refs[{idx}] 3D audio_latent must be [1, Na>0, {aud_ch}], "
                            f"got {tuple(alat.shape)}"
                        )
                    item["audio_latent"] = alat
                validated_refs.append(item)

        vision_ranges = _derive_text_vision_row_ranges(text_token_tags)
        self._av_cond_version += 1
        self._av_cond_device_cache.clear()
        if not validated_kf and not validated_refs and not vision_ranges:
            self._av_conditions = None
            return
        self._av_conditions = {
            "keyframes": validated_kf,
            "refs": validated_refs,
            "text_vision_row_ranges": vision_ranges,
        }

    def clear_av_conditions(self) -> None:
        """Remove bound AV conditions for subsequent sessions."""
        self._av_conditions = None
        self._av_cond_version += 1
        self._av_cond_device_cache.clear()

    def close(self) -> None:
        """Close any active session, clear cached condition tensors, and destroy the engine handle."""
        self._av_cond_device_cache.clear()
        super().close()

    def unload_vram(self, bytes_requested: int | None = None) -> int:
        """Release cached device condition tensors and unload engine GPU memory."""
        self._av_cond_device_cache.clear()
        if bytes_requested is None:
            return super().unload_vram()
        return super().unload_vram(bytes_requested)

    def to(self, *args: Any, **kwargs: Any) -> QuantFuncMiniMaxH3Transformer3DModel:
        """Move/cast transformer and release device-cached AV conditions on CPU/meta or cross-GPU migration."""
        target_dev, _ = _parse_to_args(*args, **kwargs)
        if target_dev is not None and (
            target_dev.type in ("cpu", "meta")
            or (
                target_dev.type == "cuda"
                and (int(target_dev.index) if target_dev.index is not None else 0) != self._device_id
            )
        ):
            self._av_cond_device_cache.clear()
        super().to(*args, **kwargs)
        return self

    def _has_bound_av_conds(self) -> bool:
        if not self._av_conditions:
            return False
        return bool(self._av_conditions.get("keyframes") or self._av_conditions.get("refs"))

    def _serialize_av_conditions(self, device: torch.device) -> tuple[list[dict[str, Any]], list[list[int]]]:
        if not self._av_conditions:
            return [], []

        cache_key = (str(device), self._av_cond_version)
        cached = self._av_cond_device_cache.get(cache_key)
        if cached is None:
            conds: list[dict[str, Any]] = []
            keepalive: list[torch.Tensor] = []
            for entry in self._av_conditions.get("keyframes", ()):
                lat = entry["latent"].to(device=device).contiguous()
                keepalive.append(lat)
                conds.append(
                    {
                        "kind": "keyframe",
                        "frame_index": int(entry["frame_index"]),
                        "latent_ptr": hex(lat.data_ptr()),
                        "latent_dims": [int(d) for d in lat.shape],
                        "latent_dtype": int(_torch_dtype_to_qf(lat.dtype)),
                    }
                )

            for entry in self._av_conditions.get("refs", ()):
                rec: dict[str, Any] = {"kind": entry["kind"]}
                if "latent" in entry:
                    lat = entry["latent"].to(device=device).contiguous()
                    keepalive.append(lat)
                    rec["latent_ptr"] = hex(lat.data_ptr())
                    rec["latent_dims"] = [int(d) for d in lat.shape]
                    rec["latent_dtype"] = int(_torch_dtype_to_qf(lat.dtype))
                if "audio_latent" in entry:
                    alat = entry["audio_latent"].to(device=device).contiguous()
                    keepalive.append(alat)
                    rec["audio_latent_ptr"] = hex(alat.data_ptr())
                    rec["audio_latent_dims"] = [int(d) for d in alat.shape]
                    rec["audio_latent_dtype"] = int(_torch_dtype_to_qf(alat.dtype))
                conds.append(rec)

            self._av_cond_device_cache[cache_key] = (conds, keepalive)
            cached_conds = conds
        else:
            cached_conds = cached[0]

        conds_out = [dict(rec) for rec in cached_conds]
        ranges = [list(r) for r in self._av_conditions.get("text_vision_row_ranges", ())]
        return conds_out, ranges

    # ── Timestep & packed-sequence decoding ─────────────────────────────────

    def _extract_t_video(
        self,
        timestep: Any,
        *,
        timestep_indices: Any,
        video_indices: Any,
        batch: int,
    ) -> list[float]:
        if torch.is_tensor(timestep):
            flat = timestep.detach().float().reshape(-1)
            if flat.numel() == 0:
                raise ValueError("MiniMax-H3 timestep tensor must be non-empty")
            if flat.numel() > 1 and timestep_indices is not None and video_indices is not None:
                if _seq_len(video_indices) == 0:
                    raise ValueError("MiniMax-H3 video_indices must be non-empty")
                last_vid_row = _index_at(video_indices, -1)
                t_slot = _index_at(timestep_indices, last_vid_row)
                if t_slot < 0 or t_slot >= flat.numel():
                    raise ValueError(
                        f"MiniMax-H3 timestep_indices[{last_vid_row}]={t_slot} out of bounds for "
                        f"timestep length {flat.numel()}"
                    )
                t_val = self._normalize_scalar_sigma(float(flat[t_slot].item()))
                return [t_val] * batch
            return self._extract_batch_sigmas(flat, batch)
        return self._extract_batch_sigmas(timestep, batch)

    def _known_cond_prefix_len(
        self,
        total_rows: int,
        explicit_prefix: int | None,
        *,
        modality_indices: Any,
        timestep_indices: Any,
    ) -> int:
        if explicit_prefix is not None:
            if isinstance(explicit_prefix, bool) or not isinstance(explicit_prefix, int):
                raise ValueError(f"MiniMax-H3 cond_prefix_len must be int, got {explicit_prefix!r}")
            if explicit_prefix < 0 or explicit_prefix >= total_rows:
                raise ValueError(
                    f"MiniMax-H3 cond_prefix_len={explicit_prefix} out of range [0, {total_rows})"
                )
            return explicit_prefix

        if modality_indices is not None and timestep_indices is not None and _seq_len(modality_indices) == total_rows:
            gen_row = _index_at(modality_indices, -1)
            gen_slot = _index_at(timestep_indices, gen_row)
            prefix = 0
            for idx in range(total_rows):
                row_idx = _index_at(modality_indices, idx)
                if _index_at(timestep_indices, row_idx) != gen_slot:
                    prefix += 1
                else:
                    break
            if 0 < prefix < total_rows:
                return prefix

        return 0

    def _infer_cond_prefix_len(
        self,
        total_rows: int,
        explicit_prefix: int | None,
        *,
        modality_indices: Any,
        timestep_indices: Any,
        expected_gen_rows: int | None,
    ) -> int:
        prefix = self._known_cond_prefix_len(
            total_rows,
            explicit_prefix,
            modality_indices=modality_indices,
            timestep_indices=timestep_indices,
        )
        if explicit_prefix is not None or prefix > 0:
            return prefix

        if expected_gen_rows is not None and total_rows != expected_gen_rows:
            if total_rows > expected_gen_rows > 0 and self._has_bound_av_conds():
                return total_rows - expected_gen_rows
            raise ValueError(
                f"MiniMax-H3 3D hidden_states row count ({total_rows}) != expected generation rows "
                f"({expected_gen_rows}); pass explicit cond_prefix_len or bind av_conds via set_av_conditions()"
            )

        return 0

    def _resolve_video_geometry(
        self,
        hidden_states: torch.Tensor,
        *,
        cond_prefix_len: int | None,
        num_frames: int | None,
        height: int | None,
        width: int | None,
        timestep_indices: Any,
        video_indices: Any,
    ) -> tuple[torch.Tensor, bool, int, int, int, int, int, int]:
        """Return (vid_gen, was_5d, vid_prefix_len, t_lat, lat_h, lat_w, num_frames, pixel_h/w tuple -> packed)."""
        in_ch = self._video_in_channels()
        pt, ph, pw = self._patch_sizes()
        patch_dim = in_ch * pt * ph * pw
        spatial_scale = self._spatial_scale()

        if hidden_states.ndim == _RANK_5D:
            _, ch, t_lat, lat_h, lat_w = (int(d) for d in hidden_states.shape)
            if ch != in_ch:
                raise ValueError(
                    f"MiniMax-H3 5D hidden_states channels={ch} must match config in_channels={in_ch}"
                )
            if t_lat % pt != 0 or lat_h % ph != 0 or lat_w % pw != 0:
                raise ValueError(
                    f"MiniMax-H3 5D latent dims ({t_lat}, {lat_h}, {lat_w}) "
                    f"must be divisible by patch_size ({pt}, {ph}, {pw})"
                )
            derived_frames = self._frames_from_latent_t(t_lat)
            if num_frames is not None and int(num_frames) != derived_frames:
                _ = self._latent_t_from_frames(int(num_frames))
                raise ValueError(
                    f"MiniMax-H3 num_frames={num_frames} mismatches 5D latent T={t_lat} "
                    f"(expected {derived_frames} on frame_grid)"
                )
            return hidden_states.contiguous(), True, 0, t_lat, lat_h, lat_w, derived_frames, spatial_scale

        if hidden_states.ndim == _RANK_3D:
            _, total_rows, ch = (int(d) for d in hidden_states.shape)
            if ch != patch_dim:
                raise ValueError(
                    f"MiniMax-H3 3D hidden_states last dim={ch} must match video_patch_dim={patch_dim}"
                )
            if num_frames is None or height is None or width is None:
                raise ValueError(
                    "MiniMax-H3 3D hidden_states requires num_frames, height, and width"
                )
            eff_frames = int(num_frames)
            t_lat = self._latent_t_from_frames(eff_frames)
            if t_lat % pt != 0:
                raise ValueError(
                    f"MiniMax-H3 latent T={t_lat} must be divisible by patch_size[0]={pt}"
                )
            h_val, w_val = int(height), int(width)
            if h_val <= 0 or w_val <= 0:
                raise ValueError(
                    f"MiniMax-H3 height and width must be positive, got ({h_val}, {w_val})"
                )

            known_prefix = self._known_cond_prefix_len(
                total_rows,
                cond_prefix_len,
                modality_indices=video_indices,
                timestep_indices=timestep_indices,
            )
            target_gen = total_rows - known_prefix
            t_patches = t_lat // pt

            latent_valid = (h_val % ph == 0) and (w_val % pw == 0)
            latent_gen = t_patches * (h_val // ph) * (w_val // pw) if latent_valid else -1

            pixel_divisible = (
                h_val >= spatial_scale
                and w_val >= spatial_scale
                and h_val % spatial_scale == 0
                and w_val % spatial_scale == 0
            )
            pix_lat_h = (h_val // spatial_scale) if pixel_divisible else -1
            pix_lat_w = (w_val // spatial_scale) if pixel_divisible else -1
            pixel_valid = pixel_divisible and (pix_lat_h % ph == 0) and (pix_lat_w % pw == 0)
            pixel_gen = t_patches * (pix_lat_h // ph) * (pix_lat_w // pw) if pixel_valid else -1

            if latent_valid and latent_gen == target_gen:
                lat_h, lat_w = h_val, w_val
            elif pixel_valid and pixel_gen == target_gen:
                lat_h, lat_w = pix_lat_h, pix_lat_w
            elif pixel_divisible and (not latent_valid or latent_gen > total_rows):
                lat_h, lat_w = pix_lat_h, pix_lat_w
            elif latent_valid:
                lat_h, lat_w = h_val, w_val
            else:
                lat_h = h_val // spatial_scale if (h_val >= spatial_scale and h_val % spatial_scale == 0) else h_val
                lat_w = w_val // spatial_scale if (w_val >= spatial_scale and w_val % spatial_scale == 0) else w_val

            if lat_h % ph != 0 or lat_w % pw != 0:
                raise ValueError(
                    f"MiniMax-H3 latent spatial dims ({lat_h}, {lat_w}) must be divisible by ({ph}, {pw})"
                )
            expected_gen = t_patches * (lat_h // ph) * (lat_w // pw)

            prefix_len = self._infer_cond_prefix_len(
                total_rows,
                cond_prefix_len,
                modality_indices=video_indices,
                timestep_indices=timestep_indices,
                expected_gen_rows=expected_gen,
            )
            gen_rows = total_rows - prefix_len
            if gen_rows != expected_gen:
                raise ValueError(
                    f"MiniMax-H3 stripped generation rows ({gen_rows}) != expected {expected_gen} "
                    f"for T={t_lat}, lat_h={lat_h}, lat_w={lat_w}"
                )
            vid_gen = hidden_states[:, prefix_len:, :].contiguous()
            return vid_gen, False, prefix_len, t_lat, lat_h, lat_w, eff_frames, spatial_scale

        raise ValueError(f"MiniMax-H3 hidden_states must be 3D or 5D, got {hidden_states.ndim}D")

    def _resolve_audio_geometry(
        self,
        audio_hidden_states: torch.Tensor | None,
        *,
        batch: int,
        audio_cond_prefix_len: int | None,
        audio_indices: Any,
        timestep_indices: Any,
    ) -> tuple[torch.Tensor | None, bool, int, int]:
        """Return (aud_gen, was_4d, aud_prefix_len, audio_t)."""
        if audio_hidden_states is None:
            return None, False, 0, 0
        aud_ch = self._audio_in_channels()
        stereo = self._audio_stereo_channels()

        if audio_hidden_states.ndim == _RANK_4D:
            ab, ca, ka, ta = (int(d) for d in audio_hidden_states.shape)
            if ab != batch or ca != aud_ch or ka != stereo or ta <= 0:
                raise ValueError(
                    f"MiniMax-H3 4D audio_hidden_states must be [{batch}, {aud_ch}, {stereo}, Ta>0], "
                    f"got {tuple(audio_hidden_states.shape)}"
                )
            return audio_hidden_states.contiguous(), True, 0, ta

        if audio_hidden_states.ndim == _RANK_3D:
            ab, total_rows, ca = (int(d) for d in audio_hidden_states.shape)
            if ab != batch or ca != aud_ch or total_rows <= 0:
                raise ValueError(
                    f"MiniMax-H3 3D audio_hidden_states must be [{batch}, N_aud>0, {aud_ch}], "
                    f"got {tuple(audio_hidden_states.shape)}"
                )
            prefix_len = self._infer_cond_prefix_len(
                total_rows,
                audio_cond_prefix_len,
                modality_indices=audio_indices,
                timestep_indices=timestep_indices,
                expected_gen_rows=None,
            )
            gen_rows = total_rows - prefix_len
            if gen_rows <= 0 or gen_rows % stereo != 0:
                raise ValueError(
                    f"MiniMax-H3 3D audio generation rows ({gen_rows}) must be positive and divisible by stereo={stereo}"
                )
            aud_gen = audio_hidden_states[:, prefix_len:, :].contiguous()
            return aud_gen, False, prefix_len, gen_rows // stereo

        raise ValueError(
            f"MiniMax-H3 audio_hidden_states must be 3D or 4D, got {audio_hidden_states.ndim}D"
        )

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | float | int | None = None,
        timestep_indices: Any = None,
        video_indices: Any = None,
        audio_indices: Any = None,
        cond_prefix_len: int | None = None,
        audio_cond_prefix_len: int | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        data_direction: bool | None = None,
        av_sigma_shift_video: float | None = None,
        av_sigma_shift_audio: float | None = None,
        audio_scale: float = 1.0,
        denoise_mask: Any = None,
        inpaint_mask: Any = None,
        mask: Any = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor, ...]:
        """Run one MiniMax-H3 AV denoising step via qf_xfm_step_av."""
        if denoise_mask is not None or inpaint_mask is not None or mask is not None:
            raise ValueError(
                "MiniMax-H3 does not support denoise_mask / masked inpainting; remove the mask input"
            )
        if kwargs.get("controlnet_block_samples") is not None or kwargs.get("control") is not None:
            raise ValueError("MiniMax-H3 does not support ControlNet conditioning")
        if encoder_hidden_states is None:
            raise ValueError("MiniMax-H3 requires encoder_hidden_states")
        if timestep is None:
            raise ValueError("MiniMax-H3 forward requires timestep")

        if isinstance(audio_scale, bool) or not isinstance(audio_scale, int | float):
            raise ValueError(f"MiniMax-H3 audio_scale must be a positive finite float, got {audio_scale!r}")
        eff_audio_scale = float(audio_scale)
        if not math.isfinite(eff_audio_scale) or eff_audio_scale <= 0.0:
            raise ValueError(f"MiniMax-H3 audio_scale must be finite and > 0, got {audio_scale!r}")

        shift_v, shift_a = self._resolve_sigma_shifts(av_sigma_shift_video, av_sigma_shift_audio)
        eff_data_dir = self._data_direction if data_direction is None else bool(data_direction)

        orig_vid_dtype = hidden_states.dtype
        orig_aud_dtype = audio_hidden_states.dtype if audio_hidden_states is not None else orig_vid_dtype
        dev = hidden_states.device
        batch = int(hidden_states.shape[0])

        vid_gen, vid_was_5d, vid_prefix_len, t_lat, lat_h, lat_w, eff_frames, spatial_scale = (
            self._resolve_video_geometry(
                hidden_states,
                cond_prefix_len=cond_prefix_len,
                num_frames=num_frames,
                height=height,
                width=width,
                timestep_indices=timestep_indices,
                video_indices=video_indices,
            )
        )
        aud_gen, aud_was_4d, aud_prefix_len, audio_t = self._resolve_audio_geometry(
            audio_hidden_states,
            batch=batch,
            audio_cond_prefix_len=audio_cond_prefix_len,
            audio_indices=audio_indices,
            timestep_indices=timestep_indices,
        )

        ctx = encoder_hidden_states
        text_dim = self._text_dim()
        if ctx.ndim != _RANK_3D or int(ctx.shape[0]) != batch or int(ctx.shape[2]) != text_dim:
            raise ValueError(
                f"MiniMax-H3 encoder_hidden_states shape {tuple(ctx.shape)} must be [{batch}, S_txt, {text_dim}]"
            )

        t_videos = self._extract_t_video(
            timestep,
            timestep_indices=timestep_indices,
            video_indices=video_indices,
            batch=batch,
        )
        engine_sigmas: list[float] = []
        for tv in t_videos:
            sig = (1.0 - tv) if eff_data_dir else tv
            if not math.isfinite(sig) or sig < 0.0:
                raise ValueError(
                    f"MiniMax-H3 derived engine sigma={sig!r} from t_video={tv!r} "
                    f"(data_direction={eff_data_dir}) is invalid"
                )
            engine_sigmas.append(sig)

        eff_fps = self._default_fps() if fps is None else float(fps)
        if not math.isfinite(eff_fps) or eff_fps <= 0.0:
            raise ValueError(f"MiniMax-H3 fps must be finite and positive, got {eff_fps!r}")

        act_dtype = self.dtype if self.dtype in (torch.float16, torch.bfloat16, torch.float32) else orig_vid_dtype
        vid_in = vid_gen if vid_gen.dtype == act_dtype else vid_gen.to(dtype=act_dtype)
        ctx_in = ctx if (ctx.dtype == act_dtype and ctx.device == dev) else ctx.to(device=dev, dtype=act_dtype)

        aud_ch = self._audio_in_channels()
        stereo = self._audio_stereo_channels()
        pixel_h = lat_h * spatial_scale
        pixel_w = lat_w * spatial_scale

        out_vid_gen = torch.empty_like(vid_in)
        out_aud_gen = (
            torch.empty(aud_gen.shape, dtype=torch.float32, device=dev) if aud_gen is not None else None
        )

        geom_key = (
            t_lat,
            lat_h,
            lat_w,
            audio_t,
            shift_v,
            shift_a,
            eff_fps,
            bool(self._av_conditions),
            self._av_cond_version,
        )
        stream_ptr = self._producer_stream_ptr(vid_in)
        cond_qf_dtype = _torch_dtype_to_qf(ctx_in.dtype)

        for i in range(batch):
            vi = vid_in[i : i + 1].contiguous()
            vo = self._ensure_scratch("h3_vid_out", tuple(vi.shape), vi.dtype, dev)
            ci = ctx_in[i : i + 1].contiguous()
            sig_i = engine_sigmas[i]
            ctx_id = self._context_fingerprint(ci, i)

            opts = self._build_base_session_options(
                has_refs=False,
                batch=batch,
                single_cond_group=(batch == 1),
            )
            opts["num_frames"] = int(eff_frames)
            opts["fps"] = float(eff_fps)
            opts["av_sigma_shift_video"] = float(shift_v)
            opts["av_sigma_shift_audio"] = float(shift_a)
            if aud_gen is not None:
                opts["audio_dims"] = [1, int(aud_ch), int(stereo), int(audio_t)]

            cond_list, vision_ranges = self._serialize_av_conditions(dev)
            if cond_list:
                opts["av_conds"] = cond_list
            if vision_ranges:
                opts["text_vision_row_ranges"] = vision_ranges

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

            if aud_gen is not None and out_aud_gen is not None:
                ai = aud_gen[i : i + 1].to(device=dev, dtype=torch.float32).contiguous()
                ao = self._ensure_scratch("h3_aud_out", tuple(ai.shape), torch.float32, dev)
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
                    audio_scale=eff_audio_scale,
                    audio_reference_active=0,
                    producer_stream=stream_ptr,
                )
                out_aud_gen[i : i + 1] = ao
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
            out_vid_gen[i : i + 1] = vo

        if eff_data_dir:
            out_vid_gen = -out_vid_gen
            if out_aud_gen is not None:
                out_aud_gen = -out_aud_gen

        if not vid_was_5d and vid_prefix_len > 0:
            vid_pad = torch.zeros(
                (batch, vid_prefix_len, int(out_vid_gen.shape[2])),
                dtype=out_vid_gen.dtype,
                device=out_vid_gen.device,
            )
            vid_out = torch.cat([vid_pad, out_vid_gen], dim=1).to(dtype=orig_vid_dtype)
        else:
            vid_out = out_vid_gen.to(dtype=orig_vid_dtype)

        if out_aud_gen is not None:
            if not aud_was_4d and aud_prefix_len > 0:
                aud_pad = torch.zeros(
                    (batch, aud_prefix_len, int(out_aud_gen.shape[2])),
                    dtype=out_aud_gen.dtype,
                    device=out_aud_gen.device,
                )
                aud_out = torch.cat([aud_pad, out_aud_gen], dim=1).to(dtype=orig_aud_dtype)
            else:
                aud_out = out_aud_gen.to(dtype=orig_aud_dtype)
            return (vid_out, aud_out)

        if return_dict:
            return Transformer2DModelOutput(sample=vid_out)
        return (vid_out,)
