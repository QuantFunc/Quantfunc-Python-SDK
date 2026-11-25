"""Generic latent upscaler protocol and spatial interpolation implementations."""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from quantfunc.sampling.bundle import LatentBundle

_VALID_INTERP_MODES = ("nearest", "nearest-exact", "bilinear", "bicubic", "area", "trilinear")
_ALIGN_CORNERS_MODES = ("bilinear", "bicubic", "trilinear")


@runtime_checkable
class LatentUpscaler(Protocol):
    """Protocol implemented by spatial or learned latent upscalers."""

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        ...


def _snap_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(round(value)))
    return max(multiple, int(round(value / multiple)) * multiple)


class InterpolateLatentUpscaler:
    """Spatial interpolation upscaler supporting 4D image and 5D video latents."""

    def __init__(
        self,
        scale_by: float = 1.5,
        *,
        mode: str = "bilinear",
        canvas_multiple: int = 16,
        spatial_compression: int | None = None,
        antialias: bool = False,
    ) -> None:
        if isinstance(scale_by, bool) or not isinstance(scale_by, int | float):
            raise ValueError(f"scale_by must be a positive finite float, got {scale_by!r}.")
        scale_f = float(scale_by)
        if not math.isfinite(scale_f) or scale_f <= 0.0:
            raise ValueError(f"scale_by must be finite and > 0, got {scale_by!r}.")
        if mode not in _VALID_INTERP_MODES:
            raise ValueError(
                f"Unsupported interpolation mode {mode!r}; expected one of {_VALID_INTERP_MODES}."
            )
        if isinstance(canvas_multiple, bool) or not isinstance(canvas_multiple, int) or canvas_multiple < 1:
            raise ValueError(f"canvas_multiple must be an integer >= 1, got {canvas_multiple!r}.")
        if spatial_compression is not None:
            if (
                isinstance(spatial_compression, bool)
                or not isinstance(spatial_compression, int)
                or spatial_compression < 1
            ):
                raise ValueError(
                    f"spatial_compression must be an integer >= 1, got {spatial_compression!r}."
                )

        self.scale_by = scale_f
        self.mode = mode
        self.canvas_multiple = canvas_multiple
        self.spatial_compression = spatial_compression
        self.antialias = bool(antialias)

    def _compute_target_hw(
        self,
        in_h: int,
        in_w: int,
        meta: dict[str, Any],
    ) -> tuple[int, int]:
        compression = self.spatial_compression or meta.get("spatial_compression")
        if compression is None:
            prev_h = meta.get("prev_height")
            prev_w = meta.get("prev_width")
            if (
                isinstance(prev_h, int)
                and isinstance(prev_w, int)
                and in_h > 0
                and in_w > 0
                and prev_h % in_h == 0
                and prev_w % in_w == 0
                and (prev_h // in_h) == (prev_w // in_w)
            ):
                compression = prev_h // in_h

        if isinstance(compression, int) and compression >= 1:
            lat_mult = max(1, math.lcm(self.canvas_multiple, compression) // compression)
        else:
            lat_mult = self.canvas_multiple

        out_h = _snap_to_multiple(in_h * self.scale_by, lat_mult)
        out_w = _snap_to_multiple(in_w * self.scale_by, lat_mult)
        return out_h, out_w

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        del target_height, target_width, kwargs
        tensor = bundle.video_or_image
        if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (4, 5):
            raise ValueError(
                "InterpolateLatentUpscaler requires a 4D [B,C,H,W] or 5D [B,C,F,H,W] tensor."
            )

        orig_dtype = tensor.dtype
        work = tensor.to(dtype=torch.float32) if orig_dtype in (torch.bfloat16, torch.float16) else tensor

        if work.ndim == 4:
            in_h, in_w = int(work.shape[-2]), int(work.shape[-1])
            out_h, out_w = self._compute_target_hw(in_h, in_w, bundle.meta)
            interp_kwargs: dict[str, Any] = {"size": (out_h, out_w), "mode": self.mode}
            if self.mode in _ALIGN_CORNERS_MODES:
                interp_kwargs["align_corners"] = False
            if self.antialias and self.mode in ("bilinear", "bicubic"):
                interp_kwargs["antialias"] = True
            upscaled = F.interpolate(work, **interp_kwargs)
        else:
            bsz, channels, frames, in_h, in_w = work.shape
            out_h, out_w = self._compute_target_hw(int(in_h), int(in_w), bundle.meta)
            if self.mode == "trilinear":
                upscaled = F.interpolate(
                    work,
                    size=(int(frames), out_h, out_w),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                flat = work.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, in_h, in_w)
                interp_kwargs = {"size": (out_h, out_w), "mode": self.mode}
                if self.mode in _ALIGN_CORNERS_MODES:
                    interp_kwargs["align_corners"] = False
                if self.antialias and self.mode in ("bilinear", "bicubic"):
                    interp_kwargs["antialias"] = True
                flat_up = F.interpolate(flat, **interp_kwargs)
                upscaled = flat_up.reshape(bsz, frames, channels, out_h, out_w).permute(0, 2, 1, 3, 4)

        upscaled = upscaled.to(dtype=orig_dtype)
        new_meta = dict(bundle.meta)
        new_meta["reference_latent"] = tensor
        return LatentBundle(video_or_image=upscaled, audio=bundle.audio, meta=new_meta)


class _VideoOnlyUpscaler:
    """Wrapper ensuring an inner upscaler modifies only the visual latent stream."""

    def __init__(self, inner: LatentUpscaler) -> None:
        if not callable(inner):
            raise TypeError(f"inner upscaler must be callable, got {type(inner).__name__}.")
        self._inner = inner

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        visual_only = LatentBundle(
            video_or_image=bundle.video_or_image,
            audio=None,
            meta=dict(bundle.meta),
        )
        upscaled = self._inner(
            visual_only,
            target_height=target_height,
            target_width=target_width,
            **kwargs,
        )
        return LatentBundle(
            video_or_image=upscaled.video_or_image,
            audio=bundle.audio,
            meta=dict(upscaled.meta),
        )


def video_only_upscaler(inner: LatentUpscaler) -> LatentUpscaler:
    """Wrap ``inner`` so that ``bundle.audio`` is preserved byte-for-byte."""
    return _VideoOnlyUpscaler(inner)
