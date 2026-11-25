"""LTX-2.5 AV pipeline loader, neural upscaler, scheduler guard, and two-stage helper."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from quantfunc.models.ltx25 import QuantFuncLTX25Transformer3DModel
from quantfunc.pipelines._common import finalize_pipeline_load, import_diffusers_class
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.renoise import FlowMatchRenoise
from quantfunc.sampling.runner import StageSpec

_DEFAULT_STAGE2_SIGMAS: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0)


class LTX2NeuralLatentUpscaler:
    """Learned 2x video latent upscaler implementing L3 ``LatentUpscaler``.

    Operates in unnormalized latent space matching ``LTX2Pipeline(output_type="latent")``
    and never double-divides ``vae.latents_std`` before Stage-2 ``prepare_latents``.
    Audio latents in ``LatentBundle.audio`` are preserved unchanged.
    """

    def __init__(
        self,
        upsampler: Any,
        *,
        vae: Any | None = None,
        latents_normalized: bool = False,
        adain_factor: float = 0.0,
        tone_map_compression: float = 0.0,
    ) -> None:
        if upsampler is None:
            raise ValueError("upsampler must not be None.")
        self._upsampler = upsampler
        self._vae = vae if vae is not None else getattr(upsampler, "vae", None)
        self.latents_normalized = bool(latents_normalized)
        self.adain_factor = float(adain_factor)
        self.tone_map_compression = float(tone_map_compression)

    def _denormalize_once(self, latents: torch.Tensor) -> torch.Tensor:
        if self._vae is None:
            raise ValueError(
                "latents_normalized=True requires a VAE with latents_mean and latents_std."
            )
        mean = getattr(self._vae, "latents_mean", None)
        std = getattr(self._vae, "latents_std", None)
        if mean is None or std is None:
            raise ValueError("Provided VAE is missing latents_mean or latents_std.")
        vae_cfg = getattr(self._vae, "config", None)
        scaling_factor = float(getattr(vae_cfg, "scaling_factor", 1.0)) if vae_cfg is not None else 1.0
        mean_t = mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        std_t = std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        return latents * std_t / scaling_factor + mean_t

    def __call__(
        self,
        bundle: LatentBundle,
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        **kwargs: Any,
    ) -> LatentBundle:
        del target_height, target_width, kwargs
        video = bundle.video_or_image
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(
                "LTX2NeuralLatentUpscaler requires a 5D video latent [B, C, F, H, W], "
                f"got {type(video).__name__} with shape {getattr(video, 'shape', None)}."
            )

        work = self._denormalize_once(video) if self.latents_normalized else video
        orig_dtype = work.dtype

        neural_module = getattr(self._upsampler, "latent_upsampler", self._upsampler)
        mod_dtype = getattr(neural_module, "dtype", None)
        if isinstance(mod_dtype, torch.dtype) and work.dtype != mod_dtype:
            model_in = work.to(dtype=mod_dtype)
        else:
            model_in = work

        with torch.no_grad():
            upsampled = neural_module(model_in)
            if isinstance(upsampled, tuple):
                upsampled = upsampled[0]
            elif hasattr(upsampled, "sample"):
                upsampled = upsampled.sample
            elif hasattr(upsampled, "frames"):
                upsampled = upsampled.frames

        if self.adain_factor != 0.0:
            upsampled = FlowMatchRenoise.adain_filter(upsampled, work, self.adain_factor)
        if self.tone_map_compression > 0.0:
            upsampled = FlowMatchRenoise.tone_map(upsampled, self.tone_map_compression)

        upsampled = upsampled.to(dtype=orig_dtype)
        new_meta = dict(bundle.meta)
        new_meta["latents_normalized"] = False
        return LatentBundle(video_or_image=upsampled, audio=bundle.audio, meta=new_meta)


def _set_config_value(sched: Any, key: str, value: Any) -> bool:
    cfg = getattr(sched, "config", None)
    if cfg is None:
        return False

    has_key = False
    cur_val: Any = None
    if isinstance(cfg, Mapping):
        if key in cfg:
            has_key = True
            cur_val = cfg[key]
    elif hasattr(cfg, key):
        has_key = True
        cur_val = getattr(cfg, key)

    if not has_key or cur_val == value:
        return False

    if callable(getattr(sched, "register_to_config", None)):
        try:
            sched.register_to_config(**{key: value})
            return True
        except Exception:
            pass

    if isinstance(cfg, dict):
        try:
            cfg[key] = value
            return True
        except TypeError:
            pass

    if isinstance(cfg, Mapping):
        updated = dict(cfg)
        updated[key] = value
        if hasattr(sched, "_internal_dict"):
            try:
                sched._internal_dict = type(cfg)(updated)
                return True
            except Exception:
                try:
                    sched._internal_dict = updated
                    return True
                except Exception:
                    pass
        try:
            object.__setattr__(sched, "config", type(cfg)(updated))
            return True
        except Exception:
            object.__setattr__(sched, "config", updated)
            return True

    setattr(cfg, key, value)
    return True


def ensure_deterministic_scheduler(pipe_or_scheduler: Any) -> Any:
    """Neutralize stochastic scheduler parameters (``eta``, ``s_churn``, ``stochastic_sampling``).

    Emits at most one ``UserWarning`` when any non-deterministic setting is neutralized.
    """
    sched = getattr(pipe_or_scheduler, "scheduler", pipe_or_scheduler)
    if sched is None:
        return None

    changes: list[str] = []

    for attr_name in ("eta", "s_churn"):
        changed = False
        cur = getattr(sched, attr_name, None)
        if isinstance(cur, int | float) and not isinstance(cur, bool) and float(cur) != 0.0:
            setattr(sched, attr_name, 0.0)
            changed = True
        cfg = getattr(sched, "config", None)
        cfg_val = cfg.get(attr_name) if isinstance(cfg, Mapping) else getattr(cfg, attr_name, None)
        if isinstance(cfg_val, int | float) and not isinstance(cfg_val, bool) and float(cfg_val) != 0.0:
            if _set_config_value(sched, attr_name, 0.0):
                changed = True
        if changed:
            changes.append(f"{attr_name}->0.0")

    stoch_changed = False
    if bool(getattr(sched, "stochastic_sampling", False)):
        sched.stochastic_sampling = False
        stoch_changed = True
    cfg = getattr(sched, "config", None)
    cfg_stoch = (
        cfg.get("stochastic_sampling", False)
        if isinstance(cfg, Mapping)
        else getattr(cfg, "stochastic_sampling", False)
    )
    if bool(cfg_stoch):
        if _set_config_value(sched, "stochastic_sampling", False):
            stoch_changed = True
    if stoch_changed:
        changes.append("stochastic_sampling->False")

    if changes:
        warnings.warn(
            "Neutralized stochastic scheduler settings for LTX-2.5 audio-video stability: "
            + ", ".join(changes),
            UserWarning,
            stacklevel=2,
        )
    return sched


def _extract_video_audio(output: Any) -> tuple[Any, Any]:
    if isinstance(output, tuple):
        video = output[0] if len(output) >= 1 else None
        audio = output[1] if len(output) >= 2 else None
        return video, audio
    video = getattr(output, "frames", None)
    if video is None:
        video = getattr(output, "latents", output)
    audio = getattr(output, "audio", None)
    return video, audio


def _decode_ltx25_outputs(
    pipe: Any,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor | None,
    *,
    output_type: str,
    decode_timestep: float = 0.0,
) -> Any:
    video_out: Any = video_latents
    vae = getattr(pipe, "vae", None)
    if vae is not None and callable(getattr(vae, "decode", None)):
        vae_dtype = getattr(vae, "dtype", None)
        v_in = video_latents.to(dtype=vae_dtype) if isinstance(vae_dtype, torch.dtype) else video_latents
        vae_cfg = getattr(vae, "config", None)
        if vae_cfg is not None and bool(getattr(vae_cfg, "timestep_conditioning", False)):
            ts = torch.full(
                (v_in.shape[0],),
                float(decode_timestep),
                device=v_in.device,
                dtype=v_in.dtype,
            )
            decoded_v = vae.decode(v_in, ts, return_dict=False)
        else:
            decoded_v = vae.decode(v_in, return_dict=False)
        video_out = decoded_v[0] if isinstance(decoded_v, tuple) else getattr(decoded_v, "sample", decoded_v)
        video_processor = getattr(pipe, "video_processor", None)
        if video_processor is not None and callable(getattr(video_processor, "postprocess_video", None)):
            video_out = video_processor.postprocess_video(video_out, output_type=output_type)

    audio_out: Any = audio_latents
    audio_vae = getattr(pipe, "audio_vae", None)
    if audio_latents is not None and audio_vae is not None and callable(getattr(audio_vae, "decode", None)):
        avae_dtype = getattr(audio_vae, "dtype", None)
        a_in = audio_latents.to(dtype=avae_dtype) if isinstance(avae_dtype, torch.dtype) else audio_latents
        decoded_mel = audio_vae.decode(a_in, return_dict=False)
        mel = decoded_mel[0] if isinstance(decoded_mel, tuple) else getattr(decoded_mel, "sample", decoded_mel)
        vocoder = getattr(pipe, "vocoder", None)
        audio_out = vocoder(mel) if callable(vocoder) else mel

    return SimpleNamespace(frames=video_out, audio=audio_out)


def make_ltx25_two_stage_stages(
    *,
    upscaler: Any,
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    stage1_steps: int = 20,
    stage2_steps: int = 3,
    stage2_sigmas: Sequence[float] = _DEFAULT_STAGE2_SIGMAS,
    guidance_scale: float = 3.0,
    stage2_guidance_scale: float = 1.0,
    freeze_audio_in_stage2: bool = True,
    audio_freeze_mask: torch.Tensor | None = None,
    stage2_quality_enhance: bool = True,
    decode_timestep: float = 0.0,
    output_type: str = "pil",
) -> list[StageSpec]:
    """Create a 2-stage LTX-2.5 AV schedule with latent-level frozen audio restore.

    Stage 2 always runs the pipeline with ``output_type="latent"`` (zero reliance
    on ``callback_on_step_end`` for ``audio_latents``), applies
    ``restore_frozen_audio`` to the audio latents before ``pipe.audio_vae.decode``
    and ``pipe.vocoder``, and decodes the final video and audio cleanly.
    """
    requested_output_type = output_type

    def _stage2_adapter(
        pipe: Any,
        stage: StageSpec,
        bundle: LatentBundle | None,
        call_kwargs: dict[str, Any],
    ) -> Any:
        del stage
        stage1_audio = bundle.audio if bundle is not None else call_kwargs.get("audio_latents")
        latent_kwargs = dict(call_kwargs)
        latent_kwargs["output_type"] = "latent"
        raw_out = pipe(**latent_kwargs)

        stage2_video, stage2_audio = _extract_video_audio(raw_out)

        restored_audio = stage2_audio
        if (
            stage1_audio is not None
            and stage2_audio is not None
            and (freeze_audio_in_stage2 or audio_freeze_mask is not None)
        ):
            if audio_freeze_mask is not None:
                mask = audio_freeze_mask
            else:
                mask = torch.zeros(
                    stage2_audio.shape,
                    device=stage2_audio.device,
                    dtype=stage2_audio.dtype,
                )
            restore_fn = getattr(
                getattr(pipe, "transformer", None),
                "restore_frozen_audio",
                QuantFuncLTX25Transformer3DModel.restore_frozen_audio,
            )
            restored_audio = restore_fn(stage2_audio, stage1_audio, mask)

        if requested_output_type == "latent":
            return SimpleNamespace(frames=stage2_video, audio=restored_audio)

        return _decode_ltx25_outputs(
            pipe,
            stage2_video,
            restored_audio,
            output_type=requested_output_type,
            decode_timestep=decode_timestep,
        )

    stage1 = StageSpec(
        name="base",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=stage1_steps,
        guidance_scale=guidance_scale,
        output_type="latent",
    )
    stage2 = StageSpec(
        name="upscale_refine",
        width=width * 2,
        height=height * 2,
        num_frames=num_frames,
        num_inference_steps=stage2_steps,
        sigmas=tuple(float(s) for s in stage2_sigmas),
        guidance_scale=stage2_guidance_scale,
        upscaler=upscaler,
        stage_call_adapter=_stage2_adapter,
        transformer_overrides={"quality_enhance": True} if stage2_quality_enhance else None,
        output_type=requested_output_type,
    )
    return [stage1, stage2]


def load_ltx25_pipeline(
    model_id_or_path: str,
    *,
    transformer_path: str | None = None,
    connectors_path: str | None = None,
    torch_dtype: torch.dtype | None = None,
    attention_backend: str = "auto",
    offload_mode: str | None = None,
    attach_hooks: bool = True,
    deterministic_scheduler: bool = True,
    pipeline_cls: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Load an LTX-2.5 AV pipeline with a QuantFunc-accelerated transformer."""
    cls = pipeline_cls or import_diffusers_class(
        "diffusers",
        "LTX2Pipeline",
        repo_hint="diffusers>=0.37 or diffusers@main",
    )

    transformer = kwargs.pop("transformer", None)
    if transformer is None:
        xfm_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "attention_backend": attention_backend,
        }
        if connectors_path is not None:
            xfm_kwargs["connectors_path"] = connectors_path
        transformer = QuantFuncLTX25Transformer3DModel.from_pretrained(
            transformer_path or model_id_or_path,
            **xfm_kwargs,
        )

    pipe_kwargs: dict[str, Any] = dict(kwargs)
    if torch_dtype is not None:
        pipe_kwargs["torch_dtype"] = torch_dtype
    pipe = cls.from_pretrained(model_id_or_path, transformer=transformer, **pipe_kwargs)

    if deterministic_scheduler:
        ensure_deterministic_scheduler(pipe)
    return finalize_pipeline_load(pipe, attach_hooks=attach_hooks, offload=offload_mode)
