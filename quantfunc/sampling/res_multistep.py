"""Flow-matching 2nd-order exponential multistep scheduler (``res_multistep``).

Provides exact mathematical parity with deterministic 2nd-order exponential
multistep integration (``eta=0.0``, https://arxiv.org/pdf/2308.02157) for
rectified-flow / flow-matching pipelines, while remaining a seamless drop-in
subclass of ``diffusers.FlowMatchEulerDiscreteScheduler`` when ``diffusers`` is
installed (with a lightweight pure-PyTorch fallback when absent).
"""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Literal

import torch

from quantfunc.accelerate.hooks import _get_hook_state, attach_session_hooks
from quantfunc.core.overrides import VALID_SAMPLERS

try:
    from diffusers import FlowMatchEulerDiscreteScheduler as _DiffusersFlowMatchEuler

    try:
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteSchedulerOutput as _DiffusersOutput,
        )
    except ImportError:  # pragma: no cover - defensive across diffusers versions
        _DiffusersOutput = None
    _HAS_DIFFUSERS = True
except ImportError:
    _DiffusersFlowMatchEuler = None  # type: ignore[assignment,misc]
    _DiffusersOutput = None  # type: ignore[assignment,misc]
    _HAS_DIFFUSERS = False

_DEGENERATE_EPS: float = 1e-12
_SMALL_H_THRESHOLD: float = 1e-5
_PER_TOKEN_LOWER_EPS: float = 1e-6


def _validate_sampler_mode(mode: Any) -> Literal["res_multistep", "euler"]:
    if not isinstance(mode, str) or mode not in VALID_SAMPLERS:
        raise ValueError(
            f"Unsupported sampler_mode {mode!r}; expected one of {VALID_SAMPLERS}."
        )
    return mode  # type: ignore[return-value]


def _phi1_phi2(neg_h: float) -> tuple[float, float]:
    """Evaluate ``phi1(u) = (e^u - 1) / u`` and ``phi2(u) = (phi1(u) - 1) / u`` safely."""
    if abs(neg_h) < _SMALL_H_THRESHOLD:
        u = neg_h
        u2 = u * u
        phi1 = 1.0 + u * 0.5 + u2 / 6.0 + (u2 * u) / 24.0
        phi2 = 0.5 + u / 6.0 + u2 / 24.0 + (u2 * u) / 120.0
        return phi1, phi2
    phi1 = math.expm1(neg_h) / neg_h
    phi2 = (phi1 - 1.0) / neg_h
    return phi1, phi2


def _make_step_output(prev_sample: torch.Tensor) -> Any:
    if _DiffusersOutput is not None:
        return _DiffusersOutput(prev_sample=prev_sample)
    return SimpleNamespace(prev_sample=prev_sample)


class _SchedulerConfig(dict[str, Any]):
    """Lightweight dict with attribute access mimicking ``diffusers.FrozenDict``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Scheduler config has no attribute {name!r}") from exc


class _PureTorchFlowMatchBase:
    """Pure-PyTorch fallback base mirroring ``FlowMatchEulerDiscreteScheduler``."""

    order: int = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int | None = 256,
        max_image_seq_len: int | None = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool | None = False,
        use_exponential_sigmas: bool | None = False,
        use_beta_sigmas: bool | None = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ) -> None:
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("`time_shift_type` must either be 'exponential' or 'linear'.")

        self.config = _SchedulerConfig(
            num_train_timesteps=int(num_train_timesteps),
            shift=float(shift),
            use_dynamic_shifting=bool(use_dynamic_shifting),
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=bool(invert_sigmas),
            shift_terminal=shift_terminal,
            use_karras_sigmas=bool(use_karras_sigmas),
            use_exponential_sigmas=bool(use_exponential_sigmas),
            use_beta_sigmas=bool(use_beta_sigmas),
            time_shift_type=time_shift_type,
            stochastic_sampling=bool(stochastic_sampling),
        )

        n = int(num_train_timesteps)
        timesteps = torch.linspace(float(n), 1.0, n, dtype=torch.float32)
        sigmas = timesteps / float(n)
        if not use_dynamic_shifting:
            sigmas = float(shift) * sigmas / (1.0 + (float(shift) - 1.0) * sigmas)

        self.timesteps: torch.Tensor = sigmas * float(n)
        self.sigmas: torch.Tensor = sigmas.to("cpu")
        self.sigma_min: float = float(self.sigmas[-1].item())
        self.sigma_max: float = float(self.sigmas[0].item())
        self._shift: float = float(shift)
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.num_inference_steps: int | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> Any:
        if config is None:
            raise ValueError("Please provide a config mapping to `from_config`.")
        if isinstance(config, Mapping):
            raw = dict(config)
        elif hasattr(config, "__dict__"):
            raw = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")

        raw.update(kwargs)
        sig = inspect.signature(cls.__init__)
        accepted = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        init_kwargs = {k: v for k, v in raw.items() if k in accepted}
        unused = {k: v for k, v in raw.items() if k not in accepted and not k.startswith("_")}
        inst = cls(**init_kwargs)
        if return_unused_kwargs:
            return inst, unused
        return inst

    @property
    def shift(self) -> float:
        return self._shift

    def set_shift(self, shift: float) -> None:
        self._shift = float(shift)

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def _sigma_to_t(self, sigma: float) -> float:
        return float(sigma) * float(self.config.num_train_timesteps)

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        if self.config.time_shift_type == "exponential":
            exp_mu = math.exp(float(mu))
            return exp_mu / (exp_mu + (1.0 / t - 1.0) ** float(sigma))
        return float(mu) / (float(mu) + (1.0 / t - 1.0) ** float(sigma))

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        one_minus_z = 1.0 - t
        scale_factor = one_minus_z[-1] / (1.0 - float(self.config.shift_terminal))
        return 1.0 - (one_minus_z / scale_factor)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: Sequence[float] | torch.Tensor | None = None,
        mu: float | None = None,
        timesteps: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to `True`.")

        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise ValueError("`sigmas` and `timesteps` should have the same length.")

        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (
                timesteps is not None and len(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` and `timesteps` should have the same length as `num_inference_steps`."
                )
        elif sigmas is not None:
            num_inference_steps = len(sigmas)
        elif timesteps is not None:
            num_inference_steps = len(timesteps)
        else:
            raise ValueError("One of `num_inference_steps`, `sigmas`, or `timesteps` must be provided.")

        self.num_inference_steps = int(num_inference_steps)
        is_timesteps_provided = timesteps is not None

        if is_timesteps_provided:
            ts_t = torch.as_tensor(timesteps, dtype=torch.float32)
        else:
            ts_t = None

        if sigmas is None:
            if ts_t is None:
                ts_t = torch.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    self.num_inference_steps,
                    dtype=torch.float32,
                )
            sigmas_t = ts_t / float(self.config.num_train_timesteps)
        else:
            sigmas_t = torch.as_tensor(sigmas, dtype=torch.float32)
            self.num_inference_steps = int(sigmas_t.numel())

        if self.config.use_dynamic_shifting:
            assert mu is not None
            sigmas_t = self.time_shift(float(mu), 1.0, sigmas_t)
        else:
            sigmas_t = self.shift * sigmas_t / (1.0 + (self.shift - 1.0) * sigmas_t)

        if self.config.shift_terminal:
            sigmas_t = self.stretch_shift_to_terminal(sigmas_t)

        sigmas_t = sigmas_t.to(dtype=torch.float32, device=device)
        if not is_timesteps_provided:
            ts_out = sigmas_t * float(self.config.num_train_timesteps)
        else:
            assert ts_t is not None
            ts_out = ts_t.to(dtype=torch.float32, device=device)

        if self.config.invert_sigmas:
            sigmas_t = 1.0 - sigmas_t
            ts_out = sigmas_t * float(self.config.num_train_timesteps)
            sigmas_t = torch.cat([sigmas_t, torch.ones(1, device=sigmas_t.device, dtype=torch.float32)])
        else:
            sigmas_t = torch.cat([sigmas_t, torch.zeros(1, device=sigmas_t.device, dtype=torch.float32)])

        self.timesteps = ts_out
        self.sigmas = sigmas_t
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self,
        timestep: float | torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        if len(indices) == 0:
            t_val = float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep)
            return int(torch.argmin((schedule_timesteps - t_val).abs()).item())
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def _init_step_index(self, timestep: float | torch.Tensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: float | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(sample)
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = self.timesteps.to(device=sample.device)
        ts = (
            timestep.to(device=sample.device)
            if isinstance(timestep, torch.Tensor)
            else torch.tensor([float(timestep)], device=sample.device, dtype=schedule_timesteps.dtype)
        )
        if ts.ndim == 0:
            ts = ts.unsqueeze(0)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in ts]
        elif self.step_index is not None:
            step_indices = [self.step_index] * int(ts.shape[0])
        else:
            step_indices = [self.begin_index] * int(ts.shape[0])

        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        per_token_timesteps: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        del s_churn, s_tmin, s_tmax, s_noise, generator
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor)
            and timestep.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
        ):
            raise ValueError(
                "Passing integer indices as timesteps to `step()` is not supported. "
                "Pass one of `scheduler.timesteps` instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        orig_dtype = sample.dtype
        x = sample.to(dtype=torch.float32)
        v = model_output.to(device=x.device, dtype=torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps.to(device=x.device, dtype=torch.float32) / float(
                self.config.num_train_timesteps
            )
            sigmas_3d = self.sigmas.to(device=x.device, dtype=torch.float32)[:, None, None]
            lower_mask = sigmas_3d < per_token_sigmas[None] - _PER_TOKEN_LOWER_EPS
            lower_sigmas = (lower_mask * sigmas_3d).max(dim=0).values
            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            assert self._step_index is not None
            sigma = float(self.sigmas[self._step_index].item())
            sigma_next = float(self.sigmas[self._step_index + 1].item())
            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            x0 = x - current_sigma * v
            noise = torch.randn_like(x)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            prev_sample = x + dt * v

        assert self._step_index is not None
        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(dtype=orig_dtype)

        if not return_dict:
            return (prev_sample,)
        return SimpleNamespace(prev_sample=prev_sample)

    def __len__(self) -> int:
        return int(self.config.num_train_timesteps)


_BaseScheduler: type[Any] = _DiffusersFlowMatchEuler if _HAS_DIFFUSERS else _PureTorchFlowMatchBase


class FlowMatchResMultistepScheduler(_BaseScheduler):  # type: ignore[valid-type,misc]
    """2nd-order exponential multistep scheduler for Flow Matching (``res_multistep``).

    Matches ComfyUI's ``sample_res_multistep`` (``eta=0.0``, ``cfg_pp=False``)
    while remaining a drop-in ``FlowMatchEulerDiscreteScheduler`` subclass when
    ``diffusers`` is installed. Supports zero-cost online per-request switching
    between ``"res_multistep"`` and ``"euler"`` via :attr:`sampler_mode`.
    """

    order: int = 2

    def __init__(
        self,
        *args: Any,
        sampler_mode: Literal["res_multistep", "euler"] = "res_multistep",
        **kwargs: Any,
    ) -> None:
        validated_mode = _validate_sampler_mode(sampler_mode)
        super().__init__(*args, **kwargs)
        self._sampler_mode: Literal["res_multistep", "euler"] = validated_mode
        self._old_denoised: torch.Tensor | None = None
        self._prev_sigma: float | None = None

    @property
    def sampler_mode(self) -> Literal["res_multistep", "euler"]:
        """Active integration mode: ``"res_multistep"`` (default) or ``"euler"``."""
        return self._sampler_mode

    @sampler_mode.setter
    def sampler_mode(self, value: Literal["res_multistep", "euler"]) -> None:
        self._sampler_mode = _validate_sampler_mode(value)
        self.reset_history()

    def reset_history(self) -> None:
        """Clear cached multistep history tensors and previous sigma."""
        self._old_denoised = None
        self._prev_sigma = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any = None,
        return_unused_kwargs: bool = False,
        **kwargs: Any,
    ) -> Any:
        if isinstance(config, Mapping) and not isinstance(config, dict):
            config = dict(config)
        return super().from_config(config=config, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: Sequence[float] | torch.Tensor | None = None,
        mu: float | None = None,
        timesteps: Sequence[float] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
            **kwargs,
        )
        self.reset_history()

    def set_begin_index(self, begin_index: int = 0) -> None:
        super().set_begin_index(begin_index=begin_index)
        self.reset_history()

    def __deepcopy__(self, memo: dict[int, Any]) -> FlowMatchResMultistepScheduler:
        """Clone scheduler with completely isolated multistep history (and hook-safe)."""
        cls = self.__class__
        clone = cls.__new__(cls)
        # Register clone in memo BEFORE copying __dict__ so instance method hooks
        # (e.g. _DeepcopySafeMethodHook) rebind cleanly to `clone` instead of `self`.
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            if key in ("_old_denoised", "_prev_sigma"):
                clone.__dict__[key] = None
            else:
                clone.__dict__[key] = copy.deepcopy(value, memo)
        return clone

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        per_token_timesteps: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor)
            and timestep.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
        ):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to "
                "`FlowMatchResMultistepScheduler.step()` is not supported. Pass one of "
                "`scheduler.timesteps` as a timestep."
            )

        # Delegate per-token schedules and explicit stochastic sampling cleanly to base
        if per_token_timesteps is not None or bool(getattr(self.config, "stochastic_sampling", False)):
            self.reset_history()
            return super().step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                generator=generator,
                per_token_timesteps=per_token_timesteps,
                return_dict=return_dict,
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        assert self._step_index is not None
        sigma_idx = int(self._step_index)
        sigma = float(self.sigmas[sigma_idx].item())
        sigma_next = float(self.sigmas[sigma_idx + 1].item())

        orig_dtype = sample.dtype
        x = sample.to(dtype=torch.float32)
        v = model_output.to(device=x.device, dtype=torch.float32)

        start_index = 0 if self.begin_index is None else int(self.begin_index)
        if (
            sigma_idx == 0
            or sigma_idx == start_index
            or (
                self._old_denoised is not None
                and (
                    self._old_denoised.shape != x.shape
                    or self._old_denoised.device != x.device
                )
            )
        ):
            self.reset_history()

        if self._sampler_mode == "euler":
            self.reset_history()
            x_next = x + (sigma_next - sigma) * v
        else:
            denoised = x - sigma * v
            invert = bool(getattr(self.config, "invert_sigmas", False))
            if (
                sigma_next == 0.0
                or self._old_denoised is None
                or self._prev_sigma is None
                or sigma <= 0.0
                or sigma_next < 0.0
                or self._prev_sigma <= 0.0
                or invert
            ):
                x_next = denoised if sigma_next == 0.0 else (x + (sigma_next - sigma) * v)
            else:
                t = -math.log(sigma)
                t_next = -math.log(sigma_next)
                t_prev = -math.log(self._prev_sigma)
                h = t_next - t
                if abs(h) < _DEGENERATE_EPS:
                    x_next = x + (sigma_next - sigma) * v
                else:
                    c2 = (t_prev - t) / h
                    if abs(c2) < _DEGENERATE_EPS:
                        x_next = x + (sigma_next - sigma) * v
                    else:
                        phi1, phi2 = _phi1_phi2(-h)
                        raw_b1 = phi1 - phi2 / c2
                        raw_b2 = phi2 / c2
                        b1 = 0.0 if math.isnan(raw_b1) else raw_b1
                        b2 = 0.0 if math.isnan(raw_b2) else raw_b2
                        if not (math.isfinite(b1) and math.isfinite(b2)):
                            x_next = x + (sigma_next - sigma) * v
                        else:
                            x_next = math.exp(-h) * x + h * (
                                b1 * denoised + b2 * self._old_denoised
                            )

            if sigma_next == 0.0:
                self.reset_history()
            else:
                self._old_denoised = denoised.detach()
                self._prev_sigma = sigma

        self._step_index += 1
        prev_sample = x_next.to(dtype=orig_dtype)

        if not return_dict:
            return (prev_sample,)
        return _make_step_output(prev_sample)


def _attach_base_init_signature() -> None:
    """Expose base ``__init__`` parameters + ``sampler_mode`` for ``ConfigMixin.from_config``."""
    base_sig = inspect.signature(_BaseScheduler.__init__)
    params = list(base_sig.parameters.values())
    if "sampler_mode" not in base_sig.parameters:
        sampler_param = inspect.Parameter(
            "sampler_mode",
            inspect.Parameter.KEYWORD_ONLY,
            default="res_multistep",
        )
        insert_at = len(params)
        for idx, p in enumerate(params):
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                insert_at = idx
                break
        params.insert(insert_at, sampler_param)
    FlowMatchResMultistepScheduler.__init__.__signature__ = base_sig.replace(parameters=params)  # type: ignore[attr-defined]


_attach_base_init_signature()


def _extract_scheduler_config(sched: Any) -> dict[str, Any]:
    raw_cfg = getattr(sched, "config", None)
    if isinstance(raw_cfg, Mapping):
        return dict(raw_cfg)
    if raw_cfg is not None and hasattr(raw_cfg, "__dict__"):
        return {k: v for k, v in vars(raw_cfg).items() if not k.startswith("_")}
    return {}


def _is_flow_match_euler_scheduler(sched: Any) -> bool:
    if _DiffusersFlowMatchEuler is not None and isinstance(sched, _DiffusersFlowMatchEuler):
        return True
    return type(sched).__name__ == "FlowMatchEulerDiscreteScheduler"


def use_res_multistep_scheduler(
    pipe: Any,
    *,
    primary_scheduler: Any | None = None,
    force: bool = False,
) -> Any:
    """Replace a pipeline's ``FlowMatchEulerDiscreteScheduler`` with ``FlowMatchResMultistepScheduler``.

    Idempotent when the active scheduler is already a ``FlowMatchResMultistepScheduler``.
    Non-FlowMatch-Euler custom or mock schedulers are preserved untouched unless ``force=True``.
    If session hooks are already attached to the replaced scheduler, they are cleanly re-attached.
    """
    sched = primary_scheduler if primary_scheduler is not None else getattr(pipe, "scheduler", None)
    if sched is None:
        return None
    if isinstance(sched, FlowMatchResMultistepScheduler):
        return sched
    if not _is_flow_match_euler_scheduler(sched) and not force:
        return sched

    cfg = _extract_scheduler_config(sched)
    new_sched = FlowMatchResMultistepScheduler.from_config(cfg)

    orig_shift = getattr(sched, "shift", None)
    if (
        isinstance(orig_shift, int | float)
        and not isinstance(orig_shift, bool)
        and math.isfinite(float(orig_shift))
        and callable(getattr(new_sched, "set_shift", None))
    ):
        cur_shift = getattr(new_sched, "shift", None)
        if cur_shift is None or float(orig_shift) != float(cur_shift):
            new_sched.set_shift(float(orig_shift))

    hook_state = _get_hook_state(pipe)
    was_hooked_primary = hook_state is not None and hook_state.primary_scheduler is sched

    if primary_scheduler is None or getattr(pipe, "scheduler", None) is sched:
        pipe.scheduler = new_sched
    elif hasattr(pipe, "__dict__"):
        for attr, val in list(pipe.__dict__.items()):
            if val is sched:
                pipe.__dict__[attr] = new_sched

    if was_hooked_primary:
        attach_session_hooks(
            pipe,
            primary_scheduler=new_sched if primary_scheduler is not None else None,
        )

    return new_sched


__all__ = [
    "FlowMatchResMultistepScheduler",
    "use_res_multistep_scheduler",
]
