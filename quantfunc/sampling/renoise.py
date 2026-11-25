"""Flow-matching latent re-noising and optional latent conditioning filters."""

from __future__ import annotations

import math
from typing import Any

import torch

from quantfunc.sampling.bundle import LatentBundle

_ADAIN_EPS = 1e-6


class FlowMatchRenoise:
    """Re-noise clean latents via linear flow-matching interpolation.

    Computes ``(1 - sigma) * x0 + sigma * eps`` with optional Adaptive Instance
    Normalization (AdaIN) and sigmoid tone-mapping before noise injection.
    """

    def __init__(
        self,
        *,
        adain_factor: float = 0.0,
        tone_map_compression: float = 0.0,
    ) -> None:
        if isinstance(adain_factor, bool) or not isinstance(adain_factor, int | float):
            raise ValueError(f"adain_factor must be a finite number, got {adain_factor!r}.")
        if not math.isfinite(float(adain_factor)):
            raise ValueError(f"adain_factor must be finite, got {adain_factor!r}.")
        if isinstance(tone_map_compression, bool) or not isinstance(
            tone_map_compression, int | float
        ):
            raise ValueError(
                f"tone_map_compression must be a float in [0, 1], got {tone_map_compression!r}."
            )
        comp = float(tone_map_compression)
        if not math.isfinite(comp) or comp < 0.0 or comp > 1.0:
            raise ValueError(
                f"tone_map_compression must be in [0, 1], got {tone_map_compression!r}."
            )
        self.adain_factor = float(adain_factor)
        self.tone_map_compression = comp

    @staticmethod
    def adain_filter(
        latents: torch.Tensor,
        reference: torch.Tensor,
        factor: float = 1.0,
    ) -> torch.Tensor:
        """Blend ``latents`` toward per-channel mean/std statistics of ``reference``."""
        if latents.ndim < 2 or reference.ndim < 2:
            raise ValueError("adain_filter requires tensors with at least 2 dimensions [B, C, ...].")
        if latents.shape[:2] != reference.shape[:2]:
            raise ValueError(
                f"Batch/channel mismatch in adain_filter: {latents.shape[:2]} vs {reference.shape[:2]}."
            )
        reduce_dims = tuple(range(2, latents.ndim))
        ref_reduce_dims = tuple(range(2, reference.ndim))
        if not reduce_dims:
            return latents.clone()

        ref = reference.to(device=latents.device, dtype=torch.float32)
        cur = latents.to(dtype=torch.float32)

        r_std, r_mean = torch.std_mean(ref, dim=ref_reduce_dims, keepdim=True)
        c_std, c_mean = torch.std_mean(cur, dim=reduce_dims, keepdim=True)
        matched = ((cur - c_mean) / c_std.clamp_min(_ADAIN_EPS)) * r_std + r_mean
        blended = torch.lerp(cur, matched, float(factor))
        return blended.to(dtype=latents.dtype)

    @staticmethod
    def tone_map(latents: torch.Tensor, compression: float) -> torch.Tensor:
        """Apply smooth sigmoid dynamic-range compression to latent values."""
        comp = float(compression)
        if not math.isfinite(comp) or comp < 0.0 or comp > 1.0:
            raise ValueError(f"compression must be in [0, 1], got {compression!r}.")
        if comp == 0.0:
            return latents
        scale = comp * 0.75
        abs_latents = torch.abs(latents)
        sigmoid_term = torch.sigmoid(4.0 * scale * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale * sigmoid_term
        return latents * scales

    def renoise_tensor(
        self,
        x0: torch.Tensor,
        sigma: float,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply optional filters and compute ``(1 - sigma) * x0 + sigma * eps``."""
        if isinstance(sigma, bool) or not isinstance(sigma, int | float):
            raise ValueError(f"sigma must be a float in [0, 1], got {sigma!r}.")
        sigma_f = float(sigma)
        if not math.isfinite(sigma_f) or sigma_f < 0.0 or sigma_f > 1.0:
            raise ValueError(f"sigma must be finite and in [0, 1], got {sigma!r}.")

        clean = x0
        if self.adain_factor != 0.0 and reference is not None:
            clean = self.adain_filter(clean, reference, self.adain_factor)
        if self.tone_map_compression > 0.0:
            clean = self.tone_map(clean, self.tone_map_compression)

        if noise is None:
            gen_device = (
                generator.device
                if isinstance(generator, torch.Generator)
                else clean.device
            )
            sample_dtype = (
                torch.float32
                if (gen_device.type == "cpu" and clean.dtype == torch.float16)
                else clean.dtype
            )
            eps = torch.randn(
                clean.shape,
                generator=generator,
                device=gen_device,
                dtype=sample_dtype,
            )
            if eps.device != clean.device or eps.dtype != clean.dtype:
                eps = eps.to(device=clean.device, dtype=clean.dtype)
        else:
            if tuple(noise.shape) != tuple(clean.shape):
                raise ValueError(
                    f"noise shape {tuple(noise.shape)} does not match latent shape {tuple(clean.shape)}."
                )
            eps = noise.to(device=clean.device, dtype=clean.dtype)

        return (1.0 - sigma_f) * clean + sigma_f * eps

    def __call__(
        self,
        x_or_bundle: torch.Tensor | LatentBundle,
        sigma: float,
        *,
        noise: torch.Tensor | None = None,
        audio_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        reference: torch.Tensor | None = None,
        renoise_audio: bool = False,
    ) -> Any:
        if isinstance(x_or_bundle, LatentBundle):
            ref = reference if reference is not None else x_or_bundle.meta.get("reference_latent")
            new_visual = self.renoise_tensor(
                x_or_bundle.video_or_image,
                sigma,
                noise=noise,
                generator=generator,
                reference=ref,
            )
            new_audio = x_or_bundle.audio
            if renoise_audio and new_audio is not None:
                new_audio = self.renoise_tensor(
                    new_audio,
                    sigma,
                    noise=audio_noise,
                    generator=generator,
                )
            return LatentBundle(
                video_or_image=new_visual,
                audio=new_audio,
                meta=dict(x_or_bundle.meta),
            )

        return self.renoise_tensor(
            x_or_bundle,
            sigma,
            noise=noise,
            generator=generator,
            reference=reference,
        )
