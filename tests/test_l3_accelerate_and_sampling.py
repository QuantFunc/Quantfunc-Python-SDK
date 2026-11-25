"""L3 unit & mutation tests: instance-level hooks (accelerate) and multi-stage sampling."""

from __future__ import annotations

import copy
import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from conftest import FakeQFLib

from quantfunc.accelerate.hooks import (
    _HOOK_STATE_ATTR,
    apply_cache_on_pipe,
    apply_sol_on_pipe,
    attach_session_hooks,
    detach_session_hooks,
    enable_offload,
    pipe_session,
)
from quantfunc.core.session import StagedSchedule
from quantfunc.core.sol import SolConfig
from quantfunc.models.krea2 import QuantFuncKrea2Transformer2DModel
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.renoise import FlowMatchRenoise
from quantfunc.sampling.runner import MultiStageRunner, StageResult, StageSpec, run_stages
from quantfunc.sampling.sigmas import build_linear_sigmas, denoise_sigmas, split_sigmas
from quantfunc.sampling.upscalers import (
    InterpolateLatentUpscaler,
    LatentUpscaler,
    video_only_upscaler,
)


class _StubTransformer:
    """Pure-CPU stub matching QuantFuncBaseTransformer hook surface + uncopyable lock."""

    _is_quantfunc_transformer = True

    def __init__(self) -> None:
        self._lock = threading.RLock()  # intentionally uncopyable for copy.deepcopy test
        self.session_open: bool = False
        self.close_calls: int = 0
        self.unload_calls: int = 0
        self.staged_schedules: list[StagedSchedule] = []
        self.sol_sigmas_history: list[list[float]] = []
        self.quality_enhance: bool = False
        self.block_cache: float = 0.0
        self.step_cache: float = 0.0
        self.sol_config: SolConfig | None = None
        self.active_loras: list[Any] = []
        self.attention_backend: str = "auto"
        self.cursor_history: list[int] = []

    def set_hook_step_cursor(self, idx: int) -> None:
        self.cursor_history.append(int(idx))

    def open_session(self) -> None:
        self.session_open = True

    def end_session_if_open(self) -> tuple[bool, bool]:
        was_open = self.session_open
        self.session_open = False
        self.close_calls += 1
        return (was_open, True)

    def unload_vram(self) -> int:
        self.unload_calls += 1
        return 0

    def set_staged_schedule(self, schedule: StagedSchedule) -> None:
        self.staged_schedules.append(schedule)

    def set_sol_sigmas(self, sigmas: list[float]) -> None:
        self.sol_sigmas_history.append(list(sigmas))

    def set_cache(self, block_cache: float | None = None, step_cache: float | None = None) -> None:
        if block_cache is not None:
            self.block_cache = float(block_cache)
        if step_cache is not None:
            self.step_cache = float(step_cache)

    def set_sol(self, sol_config: SolConfig) -> None:
        self.sol_config = sol_config

    def set_quality_enhance(self, enabled: bool) -> None:
        self.quality_enhance = bool(enabled)

    def set_loras(self, loras: list[Any]) -> None:
        self.active_loras = list(loras)

    def unload_loras(self) -> None:
        self.active_loras = []

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = str(backend)


class _PostIncScheduler:
    """Scheduler modeling FlowMatchEulerDiscreteScheduler post-increment step_index."""

    def __init__(self) -> None:
        self.timesteps: list[float] = []
        self.sigmas: list[float] = []
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(self, num_steps: int, sigmas: list[float] | None = None) -> None:
        if sigmas is not None:
            self.sigmas = list(sigmas)
            self.timesteps = [s * 1000.0 for s in self.sigmas[:-1]]
        else:
            self.sigmas = build_linear_sigmas(num_steps)
            self.timesteps = [s * 1000.0 for s in self.sigmas[:-1]]
        self._step_index = None

    def step(self, model_output: Any, timestep: Any, sample: Any) -> tuple[Any]:
        del model_output, timestep
        if self._step_index is None:
            self._step_index = 0
        self._step_index += 1
        return (sample,)


class _NoStepIndexScheduler:
    """Scheduler modeling DDIMScheduler without a ``step_index`` attribute."""

    def __init__(self) -> None:
        self.timesteps: list[float] = []
        self.sigmas: list[float] = []

    def set_timesteps(self, num_steps: int) -> None:
        self.sigmas = build_linear_sigmas(num_steps)
        self.timesteps = list(range(num_steps, 0, -1))

    def step(self, model_output: Any, timestep: Any, sample: Any) -> tuple[Any]:
        del model_output, timestep
        return (sample,)


class _StubPipe:
    def __init__(self, *, scheduler: Any, with_free_hooks: bool = True) -> None:
        self.transformer = _StubTransformer()
        self.scheduler = scheduler
        self.audio_scheduler = _PostIncScheduler()
        self.guidance_scale = 3.5
        self.do_classifier_free_guidance = True
        self.interrupt = False
        self.free_hooks_calls = 0
        self.model_offload_calls = 0
        self.seq_offload_calls = 0
        if not with_free_hooks:
            self._disable_free = True

    def enable_model_cpu_offload(self) -> None:
        self.model_offload_calls += 1

    def enable_sequential_cpu_offload(self) -> None:
        self.seq_offload_calls += 1


class _StubPipeWithFreeHooks(_StubPipe):
    def maybe_free_model_hooks(self) -> None:
        self.free_hooks_calls += 1


# ── Test 26: Instance hooks, scheduler isolation & exact close ──────────────


def test_instance_hooks_scheduler_isolation_and_exact_close() -> None:
    # (1) pipe.__class__ unchanged & zero cross-instance pollution
    pipe1 = _StubPipeWithFreeHooks(scheduler=_PostIncScheduler())
    pipe2 = _StubPipeWithFreeHooks(scheduler=_PostIncScheduler())
    orig_cls = pipe1.__class__

    attach_session_hooks(pipe1)
    assert pipe1.__class__ is orig_cls
    assert pipe2.__class__ is orig_cls
    assert _HOOK_STATE_ATTR not in pipe2.__dict__

    pipe2.scheduler.set_timesteps(4)
    assert pipe2.transformer.staged_schedules == []

    # (2) Auxiliary audio_scheduler.set_timesteps must NOT overwrite primary schedule/Sol
    pipe1.scheduler.set_timesteps(4)
    primary_sigmas = list(pipe1.scheduler.sigmas)
    assert len(pipe1.transformer.staged_schedules) == 1
    assert list(pipe1.transformer.staged_schedules[-1].sigmas) == pytest.approx(primary_sigmas)
    assert pipe1.transformer.sol_sigmas_history[-1] == pytest.approx(primary_sigmas)

    pipe1.audio_scheduler.set_timesteps(7)
    assert len(pipe1.transformer.staged_schedules) == 1
    assert pipe1.transformer.sol_sigmas_history[-1] == pytest.approx(primary_sigmas)

    # (3) 4-step post-increment step_index=[1,2,3,4]: open on steps 1..3, closed ONLY on step 4
    pipe1.transformer.close_calls = 0
    pipe1.transformer.open_session()
    for step_num in (1, 2, 3):
        pipe1.scheduler.step(None, step_num, None)
        assert pipe1.scheduler.step_index == step_num
        assert pipe1.transformer.session_open is True
        assert pipe1.transformer.close_calls == 0

    pipe1.scheduler.step(None, 4, None)
    assert pipe1.scheduler.step_index == 4
    assert pipe1.transformer.session_open is False
    assert pipe1.transformer.close_calls == 1

    # (4) total_steps == 1 single-step inference closes cleanly on step 1
    pipe1.scheduler.set_timesteps(1)
    pipe1.transformer.close_calls = 0
    pipe1.transformer.open_session()
    pipe1.scheduler.step(None, 0, None)
    assert pipe1.transformer.session_open is False
    assert pipe1.transformer.close_calls == 1

    # (5) Scheduler without step_index (DDIM-style) closes on len(timesteps) step
    ddim_pipe = _StubPipe(scheduler=_NoStepIndexScheduler(), with_free_hooks=False)
    attach_session_hooks(ddim_pipe)
    ddim_pipe.scheduler.set_timesteps(3)
    ddim_pipe.transformer.close_calls = 0
    ddim_pipe.transformer.open_session()
    ddim_pipe.scheduler.step(None, 3, None)
    ddim_pipe.scheduler.step(None, 2, None)
    assert ddim_pipe.transformer.session_open is True
    assert ddim_pipe.transformer.close_calls == 0
    ddim_pipe.scheduler.step(None, 1, None)
    assert ddim_pipe.transformer.session_open is False
    assert ddim_pipe.transformer.close_calls == 1

    # (6) pipe.interrupt = True closes early mid-loop
    pipe1.scheduler.set_timesteps(4)
    pipe1.transformer.close_calls = 0
    pipe1.transformer.open_session()
    pipe1.scheduler.step(None, 0, None)
    assert pipe1.transformer.session_open is True
    pipe1.interrupt = True
    pipe1.scheduler.step(None, 1, None)
    assert pipe1.transformer.session_open is False
    assert pipe1.transformer.close_calls == 1
    pipe1.interrupt = False

    # (7) copy.deepcopy(pipe.scheduler) succeeds and does NOT fire primary hooks
    copied_sched = copy.deepcopy(pipe1.scheduler)
    schedules_before = len(pipe1.transformer.staged_schedules)
    pipe1.transformer.close_calls = 0
    pipe1.transformer.open_session()
    copied_sched.set_timesteps(2)
    copied_sched.step(None, 0, None)
    copied_sched.step(None, 1, None)
    assert len(pipe1.transformer.staged_schedules) == schedules_before
    assert pipe1.transformer.session_open is True
    assert pipe1.transformer.close_calls == 0

    # (8) No maybe_free_model_hooks + pipe_session exception cleanup
    assert not hasattr(ddim_pipe, "maybe_free_model_hooks")
    ddim_pipe.transformer.close_calls = 0
    ddim_pipe.transformer.open_session()
    with pytest.raises(RuntimeError, match="boom"):
        with pipe_session(ddim_pipe):
            raise RuntimeError("boom")
    assert ddim_pipe.transformer.session_open is False
    assert ddim_pipe.transformer.close_calls == 1

    # Step exception also closes session immediately
    class _FailingSched(_PostIncScheduler):
        def step(self, model_output: Any, timestep: Any, sample: Any) -> tuple[Any]:
            raise RuntimeError("step_boom")

    fail_pipe = _StubPipe(scheduler=_FailingSched(), with_free_hooks=False)
    attach_session_hooks(fail_pipe)
    fail_pipe.scheduler.set_timesteps(4)
    fail_pipe.transformer.open_session()
    with pytest.raises(RuntimeError, match="step_boom"):
        fail_pipe.scheduler.step(None, 0, None)
    assert fail_pipe.transformer.session_open is False

    # (9) detach_session_hooks restores scheduler & pipe instance dicts cleanly
    detach_session_hooks(pipe1)
    assert _HOOK_STATE_ATTR not in pipe1.__dict__
    assert "set_timesteps" not in pipe1.scheduler.__dict__
    assert "step" not in pipe1.scheduler.__dict__
    assert "maybe_free_model_hooks" not in pipe1.__dict__

    schedules_after_detach = len(pipe1.transformer.staged_schedules)
    pipe1.scheduler.set_timesteps(3)
    assert len(pipe1.transformer.staged_schedules) == schedules_after_detach


# ── Test 27: apply_cache_on_pipe, apply_sol_on_pipe, enable_offload ─────────


def test_cache_sol_and_enable_offload() -> None:
    pipe = _StubPipeWithFreeHooks(scheduler=_PostIncScheduler())

    apply_cache_on_pipe(pipe, residual_diff_threshold=0.05, step_cache=0.12)
    assert pipe.transformer.block_cache == pytest.approx(0.05)
    assert pipe.transformer.step_cache == pytest.approx(0.12)
    assert _HOOK_STATE_ATTR in pipe.__dict__

    with pytest.raises(ValueError, match="Conflicting"):
        apply_cache_on_pipe(pipe, block_cache=0.05, residual_diff_threshold=0.09)

    sol = SolConfig(enabled=True, tau=1.3, start_percent=0.0, end_percent=0.9)
    apply_sol_on_pipe(pipe, sol)
    assert pipe.transformer.sol_config is sol

    with pytest.raises(TypeError, match="SolConfig"):
        apply_sol_on_pipe(pipe, {"enabled": True})  # type: ignore[arg-type]

    # Pipeline with no QuantFunc transformer fails closed
    empty_pipe = SimpleNamespace(scheduler=_PostIncScheduler())
    with pytest.raises(ValueError, match="No QuantFunc transformer"):
        apply_cache_on_pipe(empty_pipe, block_cache=0.05)
    with pytest.raises(ValueError, match="No QuantFunc transformer"):
        apply_sol_on_pipe(empty_pipe, sol)

    # Offload modes: model vs low_vram vs invalid
    enable_offload(pipe, mode="model")
    assert pipe.model_offload_calls == 1
    pipe.scheduler.set_timesteps(1)
    pipe.transformer.open_session()
    pipe.scheduler.step(None, 0, None)
    assert pipe.transformer.unload_calls == 0

    enable_offload(pipe, mode="low_vram")
    assert pipe.seq_offload_calls == 1
    pipe.scheduler.set_timesteps(1)
    pipe.transformer.open_session()
    pipe.scheduler.step(None, 0, None)
    assert pipe.transformer.session_open is False
    assert pipe.transformer.unload_calls == 1

    with pytest.raises(ValueError, match="Unsupported offload mode"):
        enable_offload(pipe, mode="invalid")


# ── Test 28: Sampling primitives & bounds ───────────────────────────────────


def test_sampling_primitives_and_bounds() -> None:
    bundle = LatentBundle(video_or_image=torch.ones(1, 4, 8, 8), audio=None, meta=None)
    assert bundle.meta == {}

    # build_linear_sigmas
    lin = build_linear_sigmas(4, shift=1.0)
    assert lin == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    shifted = build_linear_sigmas(4, shift=3.0)
    assert shifted[0] == pytest.approx(1.0)
    assert shifted[-1] == pytest.approx(0.0)
    assert shifted[2] > lin[2]
    assert all(a > b for a, b in zip(shifted, shifted[1:], strict=False))

    for bad_steps in (0, -3, True):
        with pytest.raises(ValueError):
            build_linear_sigmas(bad_steps)  # type: ignore[arg-type]
    for bad_shift in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            build_linear_sigmas(4, shift=bad_shift)

    # split_sigmas
    s1, s2 = split_sigmas(lin, 2)
    assert s1 == pytest.approx([1.0, 0.75, 0.5])
    assert s2 == pytest.approx([0.5, 0.25, 0.0])
    for bad_split in (-1, 0, 4, 5, True):
        with pytest.raises(ValueError):
            split_sigmas(lin, bad_split)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        split_sigmas([1.0, 0.5, 0.5, 0.0], 1)
    with pytest.raises(ValueError):
        split_sigmas([0.2, 0.8, 0.0], 1)

    # denoise_sigmas
    assert denoise_sigmas(lin, 1.0) == pytest.approx(lin)
    assert denoise_sigmas(lin, 0.5) == pytest.approx([0.5, 0.25, 0.0])
    assert denoise_sigmas([1.0, 0.0], 0.5) == pytest.approx([0.5, 0.0])
    for bad_denoise in (0.0, -0.2, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            denoise_sigmas(lin, bad_denoise)

    # FlowMatchRenoise exact formula + AdaIN / tone-map + bounds
    x0 = torch.full((1, 4, 8, 8), 2.0)
    eps = torch.full((1, 4, 8, 8), -4.0)
    renoise = FlowMatchRenoise()
    out = renoise(x0, 0.25, noise=eps)
    expected = 0.75 * x0 + 0.25 * eps
    assert torch.allclose(out, expected)

    b_in = LatentBundle(video_or_image=x0, audio=torch.ones(1, 2, 4, 4))
    b_out = renoise(b_in, 0.5, noise=eps)
    assert isinstance(b_out, LatentBundle)
    assert torch.allclose(b_out.video_or_image, 0.5 * x0 + 0.5 * eps)
    assert torch.equal(b_out.audio, b_in.audio)

    filtered = FlowMatchRenoise(adain_factor=0.5, tone_map_compression=0.5)(
        torch.randn(1, 4, 8, 8),
        0.1,
        noise=torch.zeros(1, 4, 8, 8),
        reference=torch.randn(1, 4, 8, 8) * 2.0,
    )
    assert torch.isfinite(filtered).all()

    with pytest.raises(ValueError):
        renoise(x0, 1.5, noise=eps)
    with pytest.raises(ValueError):
        renoise(x0, 0.5, noise=torch.zeros(1, 4, 4, 4))
    with pytest.raises(ValueError):
        FlowMatchRenoise(tone_map_compression=1.5)

    # InterpolateLatentUpscaler (4D & 5D) + video_only_upscaler
    up4 = InterpolateLatentUpscaler(scale_by=1.5, mode="bilinear", canvas_multiple=16)
    assert isinstance(up4, LatentUpscaler)
    b4 = LatentBundle(video_or_image=torch.randn(1, 8, 64, 64))
    b4_up = up4(b4)
    assert tuple(b4_up.video_or_image.shape) == (1, 8, 96, 96)

    up5 = InterpolateLatentUpscaler(scale_by=2.0, mode="bilinear", canvas_multiple=1)
    audio_orig = torch.randn(1, 8, 12, 16)
    audio_clone = audio_orig.clone()
    b5 = LatentBundle(video_or_image=torch.randn(1, 4, 3, 8, 8), audio=audio_orig)
    vo_up = video_only_upscaler(up5)
    assert isinstance(vo_up, LatentUpscaler)
    b5_up = vo_up(b5)
    assert tuple(b5_up.video_or_image.shape) == (1, 4, 3, 16, 16)
    assert b5_up.audio is audio_orig
    assert torch.equal(b5_up.audio, audio_clone)

    with pytest.raises(ValueError):
        InterpolateLatentUpscaler(scale_by=0.0)
    with pytest.raises(ValueError):
        InterpolateLatentUpscaler(scale_by=1.5, mode="unsupported")


# ── Test 29: MultiStageRunner & run_stages ──────────────────────────────────


def test_multi_stage_runner_adapters_and_restore() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiStageRunner([])

    pipe = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    pipe.transformer.set_quality_enhance(False)
    pipe.transformer.set_cache(block_cache=0.02, step_cache=0.01)
    pipe.transformer.set_loras([("base.safetensors", 1.0)])

    observed: list[dict[str, Any]] = []

    def _pipe_call(**kwargs: Any) -> Any:
        pipe.transformer.open_session()
        observed.append(
            {
                "width": kwargs.get("width"),
                "height": kwargs.get("height"),
                "quality_enhance": pipe.transformer.quality_enhance,
                "block_cache": pipe.transformer.block_cache,
                "loras": list(pipe.transformer.active_loras),
                "has_latents": "latents" in kwargs,
                "custom_cond": kwargs.get("custom_cond"),
            }
        )
        lat = kwargs.get("latents")
        if lat is None:
            lat = torch.ones(1, 4, 16, 16)
        if kwargs.get("output_type") == "latent":
            return SimpleNamespace(images=lat)
        return SimpleNamespace(images=[lat * 2.0])

    pipe.__call__ = _pipe_call  # test callable pipe class below

    class _CallablePipe(_StubPipe):
        def __call__(self, **kwargs: Any) -> Any:
            return _pipe_call(**kwargs)

    cpipe = _CallablePipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    cpipe.transformer = pipe.transformer

    def _cond_cb(_p: Any, stage: StageSpec, _b: LatentBundle | None, kw: dict[str, Any]) -> dict[str, Any]:
        kw = dict(kw)
        kw["custom_cond"] = f"cond_for_{stage.name}"
        return kw

    def _custom_adapter(p: Any, stage: StageSpec, bundle: LatentBundle | None, kw: dict[str, Any]) -> Any:
        assert stage.name == "refine"
        assert bundle is not None and tuple(bundle.video_or_image.shape) == (1, 4, 32, 32)
        return p(**kw)

    stages = [
        StageSpec(
            name="draft",
            width=256,
            height=256,
            num_inference_steps=4,
            output_type="latent",
            transformer_overrides={"block_cache": 0.08},
            latent_extractor=lambda out, _s: LatentBundle(video_or_image=out.images),
        ),
        StageSpec(
            name="mid",
            width=384,
            height=384,
            num_inference_steps=4,
            denoise=0.6,
            upscaler=InterpolateLatentUpscaler(scale_by=1.5, canvas_multiple=1),
            renoise=FlowMatchRenoise(),
            transformer_overrides={"loras": [("mid.safetensors", 0.5)]},
            conditioning_callback=_cond_cb,
            output_type="latent",
        ),
        StageSpec(
            name="refine",
            width=512,
            height=512,
            num_inference_steps=4,
            denoise=0.5,
            upscaler=InterpolateLatentUpscaler(scale_by=4 / 3, canvas_multiple=1),
            renoise=FlowMatchRenoise(),
            stage_call_adapter=_custom_adapter,
            transformer_overrides={"quality_enhance": True, "block_cache": 0.0, "loras": []},
            output_type="pil",
        ),
    ]

    results = run_stages(cpipe, stages, return_all_stages=True, prompt="test prompt")
    assert len(results) == 3
    assert all(isinstance(r, StageResult) for r in results)

    # Verify per-stage overrides took effect during each stage
    assert observed[0]["block_cache"] == pytest.approx(0.08)
    assert observed[1]["loras"] == [("mid.safetensors", 0.5)]
    assert observed[1]["custom_cond"] == "cond_for_mid"
    assert observed[2]["quality_enhance"] is True
    assert observed[2]["block_cache"] == pytest.approx(0.0)
    assert observed[2]["loras"] == []

    # Verify all transformer dials were restored cleanly and sessions closed after each stage
    assert cpipe.transformer.quality_enhance is False
    assert cpipe.transformer.block_cache == pytest.approx(0.02)
    assert cpipe.transformer.step_cache == pytest.approx(0.01)
    assert cpipe.transformer.active_loras == [("base.safetensors", 1.0)]
    assert cpipe.transformer.session_open is False
    assert cpipe.transformer.close_calls == 3

    # Verify restoration and session cleanup EVEN when a stage raises RuntimeError
    def _failing_adapter(_p: Any, _s: StageSpec, _b: LatentBundle | None, _kw: dict[str, Any]) -> Any:
        cpipe.transformer.open_session()
        assert cpipe.transformer.quality_enhance is True
        raise RuntimeError("stage_failure")

    fail_stages = [
        StageSpec(
            name="boom",
            num_inference_steps=2,
            transformer_overrides={"quality_enhance": True, "block_cache": 0.99, "loras": []},
            stage_call_adapter=_failing_adapter,
        )
    ]
    with pytest.raises(RuntimeError, match="stage_failure"):
        MultiStageRunner(fail_stages).run(cpipe)

    assert cpipe.transformer.quality_enhance is False
    assert cpipe.transformer.block_cache == pytest.approx(0.02)
    assert cpipe.transformer.active_loras == [("base.safetensors", 1.0)]
    assert cpipe.transformer.session_open is False


# ── Test 30 / Mutation M4: hooks & MultiStageRunner ─────────────────────────


def test_mutation_l3_hooks_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantfunc.accelerate.hooks as hooks_mod
    import quantfunc.sampling.runner as runner_mod

    # M4(a1): mutating pipe.__class__ MUST fail class-identity assertion
    pipe = _StubPipeWithFreeHooks(scheduler=_PostIncScheduler())
    orig_cls = pipe.__class__

    def _broken_attach_mutates_class(p: Any, *, primary_scheduler: Any | None = None) -> Any:
        attach_session_hooks(p, primary_scheduler=primary_scheduler)
        p.__class__ = type("MutatedPipe", (p.__class__,), {})
        return p

    _broken_attach_mutates_class(pipe)
    with pytest.raises(AssertionError):
        assert pipe.__class__ is orig_cls

    # M4(a2): off-by-one close condition (cur_idx >= total - 1) closes early on step 3 of 4 -> MUST FAIL
    pipe_obo = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    sched = pipe_obo.scheduler
    orig_step = sched.step
    sched.set_timesteps(4)
    total = len(sched.timesteps)
    pipe_obo.transformer.open_session()

    def _mutated_off_by_one_step(*args: Any, **kwargs: Any) -> Any:
        res = orig_step(*args, **kwargs)
        cur_idx = getattr(sched, "step_index", None)
        if cur_idx is not None and cur_idx >= total - 1:  # MUTATION: off-by-one!
            pipe_obo.transformer.end_session_if_open()
        return res

    sched.step = _mutated_off_by_one_step  # type: ignore[method-assign]
    for s in (1, 2, 3):
        sched.step(None, s, None)
    with pytest.raises(AssertionError):
        assert pipe_obo.transformer.session_open is True

    # Verify real attach_session_hooks passes the exact same 4-step assertion (green control)
    pipe_good = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    hooks_mod.attach_session_hooks(pipe_good)
    pipe_good.scheduler.set_timesteps(4)
    pipe_good.transformer.open_session()
    for s in (1, 2, 3):
        pipe_good.scheduler.step(None, s, None)
    assert pipe_good.transformer.session_open is True
    pipe_good.scheduler.step(None, 4, None)
    assert pipe_good.transformer.session_open is False

    # M4(b): removing finally restoration of transformer_overrides MUST fail post-exception check
    pipe_m4b = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    pipe_m4b.transformer.set_quality_enhance(False)

    monkeypatch.setattr(runner_mod, "_restore_transformer_overrides", lambda _saved: None)
    bad_stage = StageSpec(
        name="fail_stage",
        transformer_overrides={"quality_enhance": True},
        stage_call_adapter=lambda _p, _s, _b, _kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner_mod.MultiStageRunner([bad_stage]).run(pipe_m4b)

    with pytest.raises(AssertionError):
        assert pipe_m4b.transformer.quality_enhance is False


# ── Test 30b: Real diffusers schedulers E2E (FlowMatchEuler, DDIM, Heun) ────


def _make_dummy_krea2_ckpt(path: Path) -> Path:
    raw = struct.pack("<2f", 1.0, 2.0)
    header = json.dumps(
        {"transformer_blocks.0.attn.to_q.weight": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, len(raw)]}},
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(raw)
    return path


def test_real_diffusers_schedulers_e2e_krea2(tmp_path: Path) -> None:
    diffusers = pytest.importorskip("diffusers")
    from diffusers import (
        DDIMScheduler,
        FlowMatchEulerDiscreteScheduler,
        HeunDiscreteScheduler,
    )

    _ = diffusers
    ckpt = _make_dummy_krea2_ckpt(tmp_path / "krea2-quantfunc-int4.safetensors")

    def _make_pipe(sched: Any) -> tuple[Any, FakeQFLib, QuantFuncKrea2Transformer2DModel]:
        lib = FakeQFLib()
        xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
        pipe = SimpleNamespace(
            transformer=xfm,
            scheduler=sched,
            guidance_scale=1.0,
            do_classifier_free_guidance=False,
            interrupt=False,
        )
        attach_session_hooks(pipe)
        return pipe, lib, xfm

    ctx = torch.randn(1, 4, 2560, dtype=torch.float32)

    # (1) FlowMatchEulerDiscreteScheduler(4): timesteps=[1000, 667, 334, 1.0]
    #     Verifies exact step_index=[0, 1, 2, 3], final sigma=0.001 (NOT 1.0!), close ONLY on step 4.
    pipe_fm, lib_fm, xfm_fm = _make_pipe(FlowMatchEulerDiscreteScheduler())
    try:
        pipe_fm.scheduler.set_timesteps(4)
        staged = xfm_fm._staged_schedule
        assert staged is not None
        assert staged.num_steps == 4
        assert staged.timestep_scale == pytest.approx(1000.0)
        assert staged.use_cursor_indexing is False
        assert xfm_fm._session_tracker._step_cursor == 0

        lat = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        for idx, t in enumerate(pipe_fm.scheduler.timesteps):
            vel = xfm_fm(lat, encoder_hidden_states=ctx, timestep=t).sample
            lat = pipe_fm.scheduler.step(vel, t, lat, return_dict=False)[0]
            if idx < 3:
                assert lib_fm.sessions[-1]["ended"] is False
                assert xfm_fm._session_tracker._step_cursor == idx + 1
            else:
                assert lib_fm.sessions[-1]["ended"] is True
                assert xfm_fm._session_tracker._step_cursor == 0

        assert len(lib_fm.sessions) == 1
        assert [s["step_index"] for s in lib_fm.steps] == [0, 1, 2, 3]
        assert [s["sigma"] for s in lib_fm.steps] == pytest.approx([1.0, 0.667, 0.334, 0.001], abs=1e-3)
    finally:
        xfm_fm.close()

    # (2) DDIMScheduler(4): sigmas=None, no step_index attr
    #     Verifies use_cursor_indexing=True, step_indices=[0, 1, 2, 3], close ONLY on step 4.
    pipe_ddim, lib_ddim, xfm_ddim = _make_pipe(DDIMScheduler())
    try:
        pipe_ddim.scheduler.set_timesteps(4)
        staged = xfm_ddim._staged_schedule
        assert staged is not None
        assert staged.num_steps == 4
        assert staged.sigmas == ()
        assert staged.timestep_scale == pytest.approx(1000.0)
        assert staged.use_cursor_indexing is True
        assert xfm_ddim._session_tracker._step_cursor == 0

        lat = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        for idx, t in enumerate(pipe_ddim.scheduler.timesteps):
            vel = xfm_ddim(lat, encoder_hidden_states=ctx, timestep=t).sample
            lat = pipe_ddim.scheduler.step(vel, t, lat, return_dict=False)[0]
            if idx < 3:
                assert lib_ddim.sessions[-1]["ended"] is False
                assert xfm_ddim._session_tracker._step_cursor == idx + 1
            else:
                assert lib_ddim.sessions[-1]["ended"] is True
                assert xfm_ddim._session_tracker._step_cursor == 0

        assert len(lib_ddim.sessions) == 1
        assert [s["step_index"] for s in lib_ddim.steps] == [0, 1, 2, 3]
    finally:
        xfm_ddim.close()

    # (3) HeunDiscreteScheduler(4): 7 calls, EDM sigmas > 1.0 + duplicate adjacent sigmas
    #     Verifies use_cursor_indexing=True, step_indices=[0, 1, 2, 3, 4, 5, 6], close ONLY on call 7.
    pipe_heun, lib_heun, xfm_heun = _make_pipe(HeunDiscreteScheduler())
    try:
        pipe_heun.scheduler.set_timesteps(4)
        staged = xfm_heun._staged_schedule
        assert staged is not None
        assert staged.num_steps == 7
        assert staged.timestep_scale == pytest.approx(1000.0)
        assert staged.use_cursor_indexing is True
        assert xfm_heun._session_tracker._step_cursor == 0

        lat = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        for idx, t in enumerate(pipe_heun.scheduler.timesteps):
            vel = xfm_heun(lat, encoder_hidden_states=ctx, timestep=t).sample
            lat = pipe_heun.scheduler.step(vel, t, lat, return_dict=False)[0]
            if idx < 6:
                assert lib_heun.sessions[-1]["ended"] is False
                assert xfm_heun._session_tracker._step_cursor == idx + 1
            else:
                assert lib_heun.sessions[-1]["ended"] is True
                assert xfm_heun._session_tracker._step_cursor == 0

        assert len(lib_heun.sessions) == 1
        assert [s["step_index"] for s in lib_heun.steps] == [0, 1, 2, 3, 4, 5, 6]
    finally:
        xfm_heun.close()

    # (4) Edge-case detection arms for hooks.py: unit-scale timesteps, duplicate sigmas in [0, 1], length mismatch
    class _CustomSched:
        def __init__(self, timesteps: list[float] | None, sigmas: list[float] | None) -> None:
            self.timesteps = timesteps
            self.sigmas = sigmas

        def set_timesteps(self, *_a: Any, **_kw: Any) -> None:
            pass

        def step(self, _m: Any, _t: Any, sample: Any) -> tuple[Any]:
            return (sample,)

    # Unit-scale timesteps [1.0, 0.5] -> timestep_scale=1.0, use_cursor_indexing=False
    p_unit = _StubPipe(scheduler=_CustomSched([1.0, 0.5], [1.0, 0.5, 0.0]), with_free_hooks=False)
    attach_session_hooks(p_unit)
    p_unit.scheduler.set_timesteps(2)
    assert p_unit.transformer.staged_schedules[-1].timestep_scale == pytest.approx(1.0)
    assert p_unit.transformer.staged_schedules[-1].use_cursor_indexing is False
    assert p_unit.transformer.cursor_history[-1] == 0

    # Duplicate adjacent sigmas inside [0, 1] -> use_cursor_indexing=True
    p_dup = _StubPipe(scheduler=_CustomSched([1.0, 0.5, 0.5], [1.0, 0.5, 0.5, 0.0]), with_free_hooks=False)
    attach_session_hooks(p_dup)
    p_dup.scheduler.set_timesteps(3)
    assert p_dup.transformer.staged_schedules[-1].use_cursor_indexing is True

    # Mismatched len(sigmas) != total_calls + 1 -> use_cursor_indexing=True
    p_len = _StubPipe(scheduler=_CustomSched([1000.0, 500.0, 100.0], [1.0, 0.5]), with_free_hooks=False)
    attach_session_hooks(p_len)
    p_len.scheduler.set_timesteps(3)
    assert p_len.transformer.staged_schedules[-1].use_cursor_indexing is True

    # Missing timesteps -> timestep_scale=None
    p_nots = _StubPipe(scheduler=_CustomSched(None, [1.0, 0.5, 0.0]), with_free_hooks=False)
    attach_session_hooks(p_nots)
    p_nots.scheduler.set_timesteps(2)
    assert p_nots.transformer.staged_schedules[-1].timestep_scale is None
    assert p_nots.transformer.staged_schedules[-1].use_cursor_indexing is False

    # (5) Mutation arm: forcing use_cursor_indexing=False on Heun collapses duplicate sigmas -> fails [0..6]
    pipe_mut, lib_mut, xfm_mut = _make_pipe(HeunDiscreteScheduler())
    try:
        pipe_mut.scheduler.set_timesteps(4)
        bad_sched = StagedSchedule(
            sigmas=tuple(float(s) for s in pipe_mut.scheduler.sigmas.tolist()),
            num_steps=7,
            timestep_scale=1000.0,
            use_cursor_indexing=False,  # MUTATION!
        )
        xfm_mut.set_staged_schedule(bad_sched)
        lat = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        for t in pipe_mut.scheduler.timesteps:
            vel = xfm_mut(lat, encoder_hidden_states=ctx, timestep=t).sample
            lat = pipe_mut.scheduler.step(vel, t, lat, return_dict=False)[0]
        with pytest.raises(AssertionError):
            assert [s["step_index"] for s in lib_mut.steps] == [0, 1, 2, 3, 4, 5, 6]
    finally:
        xfm_mut.close()


# ── Rev 4 Helpers & Tests 43-45 (request_scope, runner dedup, Mutation M9) ──


def _make_dummy_lora_safetensors(path: Path, val: float = 1.0) -> Path:
    raw = struct.pack("<4f", val, val + 1.0, val + 2.0, val + 3.0)
    header = json.dumps(
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, len(raw)],
            },
            "transformer_blocks.0.attn.to_q.lora_B.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, len(raw)],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(raw)
    return path


def _make_dummy_ckpt_with_keys(path: Path, keys: list[str]) -> Path:
    raw = struct.pack("<2f", 1.0, 2.0)
    hdr_dict = {
        k: {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, len(raw)]}
        for k in keys
    }
    header = json.dumps(hdr_dict, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(raw)
    return path


# ── Test 43: request_scope sticky/non-sticky LoRAs, dials & conditions ──────


def test_request_scope_sticky_loras_dials_and_conditions(tmp_path: Path) -> None:
    from quantfunc.accelerate.serving import (
        get_pipeline_lock,
        request_scope,
    )
    from quantfunc.core.overrides import RuntimeOverrides
    from quantfunc.models.ltx25 import QuantFuncLTX25Transformer3DModel
    from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel
    from quantfunc.native.errors import EngineBusyError, UnsupportedError

    lora_a = _make_dummy_lora_safetensors(tmp_path / "lora_a.safetensors", val=1.0)
    lora_b = _make_dummy_lora_safetensors(tmp_path / "lora_b.safetensors", val=2.0)
    ckpt = _make_dummy_krea2_ckpt(tmp_path / "krea2-r4-scope.safetensors")

    lib = FakeQFLib()
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    pipe = SimpleNamespace(transformer=xfm, scheduler=_PostIncScheduler())

    try:
        # (1) sticky_loras=True: Req1 [A:0.8] -> Req2 [A:0.8] (0 extra binds) -> Req3 [] -> Req4 default_loras=() (0 extra binds)
        with request_scope(pipe, {"loras": [(str(lora_a), 0.8)]}, sticky_loras=True):
            assert len(lib.loras_history) == 1
            assert len(xfm.active_loras) == 1 and xfm.active_loras[0].scale == pytest.approx(0.8)
        # Still bound after exit because sticky_loras=True
        assert len(xfm.active_loras) == 1

        with request_scope(pipe, {"loras": [(str(lora_a), 0.8)]}, sticky_loras=True):
            assert len(lib.loras_history) == 1  # O(1) hit: 0 extra qf_xfm_set_loras!

        with request_scope(pipe, {"loras": []}, sticky_loras=True):
            assert len(lib.loras_history) == 2
            assert lib.loras_history[-1] == []
            assert xfm.active_loras == []

        with request_scope(pipe, {}, sticky_loras=True, default_loras=()):
            assert len(lib.loras_history) == 2  # Already empty -> O(1) no-op!
            assert xfm.active_loras == []

        # (2) sticky_loras=False restores LoRAs on normal exit AND exception exit
        xfm.set_loras([(str(lora_a), 0.5)])
        binds_base = len(lib.loras_history)
        with request_scope(pipe, {"loras": [(str(lora_b), 0.9)]}, sticky_loras=False):
            assert len(xfm.active_loras) == 1 and xfm.active_loras[0].scale == pytest.approx(0.9)
            assert len(lib.loras_history) == binds_base + 1
        assert len(xfm.active_loras) == 1 and xfm.active_loras[0].scale == pytest.approx(0.5)

        with pytest.raises(RuntimeError, match="scope_err"):
            with request_scope(pipe, {"loras": [(str(lora_b), 0.9)]}, sticky_loras=False):
                assert xfm.active_loras[0].scale == pytest.approx(0.9)
                raise RuntimeError("scope_err")
        assert len(xfm.active_loras) == 1 and xfm.active_loras[0].scale == pytest.approx(0.5)

        # (3) All runtime dials (incl. sol=None disabling default Sol AND sol=SolConfig when baseline None)
        base_sol = SolConfig(enabled=True, tau=1.2, start_percent=0.0, end_percent=0.9)
        xfm.set_quality_enhance(False)
        xfm.set_cache(block_cache=0.02, step_cache=0.03)
        xfm.set_sol(base_sol)
        xfm.set_attention_backend("auto")
        xfm.set_default_steps(12)

        with request_scope(
            pipe,
            {
                "quality_enhance": True,
                "fbcache": 0.08,
                "stepcache": 0.15,
                "sol": None,
                "attention_backend": "int8",
                "default_steps": 6,
            },
        ):
            assert xfm.quality_enhance is True
            assert xfm.block_cache == pytest.approx(0.08)
            assert xfm.step_cache == pytest.approx(0.15)
            assert xfm.sol_config is None
            assert xfm.attention_backend == "int8"
            assert xfm._default_steps == 6

        # 100% restored on normal exit
        assert xfm.quality_enhance is False
        assert xfm.block_cache == pytest.approx(0.02)
        assert xfm.step_cache == pytest.approx(0.03)
        assert xfm.sol_config == base_sol
        assert xfm.attention_backend == "auto"
        assert xfm._default_steps == 12

        # Baseline sol=None overridden with SolConfig restores cleanly back to None on exception!
        xfm.set_sol(None)
        x_in = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        ctx = torch.randn(1, 4, 2560, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="mid_step_boom"):
            with request_scope(
                pipe,
                RuntimeOverrides.from_dict(
                    {
                        "quality_enhance": True,
                        "fbcache": 0.09,
                        "sol": {"enabled": True, "tau": 1.35},
                        "attention_backend": "fp16",
                        "default_steps": 4,
                    }
                ),
            ):
                assert xfm.sol_config is not None and xfm.sol_config.enabled is True
                xfm(x_in, encoder_hidden_states=ctx, timestep=0.5)
                assert xfm._session_is_open() is True
                raise RuntimeError("mid_step_boom")

        assert xfm._session_is_open() is False
        assert xfm.quality_enhance is False
        assert xfm.block_cache == pytest.approx(0.02)
        assert xfm.sol_config is None
        assert xfm.attention_backend == "auto"
        assert xfm._default_steps == 12
        # Lock must be released after exception
        lock = get_pipeline_lock(pipe)
        assert lock.acquire(blocking=False) is True
        lock.release()

        # (4) Unsupported ref_audio / av_conditions on image-only Krea-2 -> UnsupportedError (None is no-op)
        with request_scope(pipe, {}, ref_audio=None, av_conditions=None):
            pass
        with pytest.raises(UnsupportedError):
            with request_scope(pipe, {}, ref_audio=torch.ones(1, 4, 128)):
                pass
        with pytest.raises(UnsupportedError):
            with request_scope(pipe, {}, av_conditions={"keyframes": []}):
                pass

        # Pre-lock validation: illegal av_conditions key & invalid overrides raise ValueError BEFORE acquiring lock!
        held = threading.Event()
        release = threading.Event()

        def _hold_lock() -> None:
            with lock:
                held.set()
                release.wait(timeout=5.0)

        holder = threading.Thread(target=_hold_lock)
        holder.start()
        assert held.wait(timeout=2.0)
        try:
            with pytest.raises(ValueError, match="Unknown av_conditions"):
                with request_scope(pipe, {}, busy_timeout=0.0, av_conditions={"bad_key": 1}):
                    pass
            with pytest.raises(ValueError, match="Unknown RuntimeOverrides"):
                with request_scope(pipe, {"unknown_dial": 1}, busy_timeout=0.0):
                    pass
            # Control: valid input with busy_timeout=0.0 raises EngineBusyError while held
            with pytest.raises(EngineBusyError):
                with request_scope(pipe, {}, busy_timeout=0.0):
                    pass
        finally:
            release.set()
            holder.join(timeout=5.0)
    finally:
        xfm.close()

    # (5) LTX-2.5 ref_audio injection & guaranteed finally cleanup (normal + exception)
    lib_ltx = FakeQFLib()
    ckpt_ltx = _make_dummy_ckpt_with_keys(
        tmp_path / "ltx25-scope.safetensors",
        [
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    xfm_ltx = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_ltx, lib=lib_ltx, torch_dtype=torch.float32)
    pipe_ltx = SimpleNamespace(transformer=xfm_ltx, scheduler=_PostIncScheduler())
    try:
        ref_a = torch.randn(1, 4, 128, dtype=torch.float32)
        with request_scope(pipe_ltx, {}, ref_audio=ref_a):
            assert xfm_ltx._ref_audio_state is not None
        assert xfm_ltx._ref_audio_state is None

        with pytest.raises(RuntimeError, match="ltx_boom"):
            with request_scope(pipe_ltx, {}, ref_audio=ref_a):
                assert xfm_ltx._ref_audio_state is not None
                raise RuntimeError("ltx_boom")
        assert xfm_ltx._ref_audio_state is None
    finally:
        xfm_ltx.close()

    # (6) MiniMax-H3 av_conditions injection & guaranteed finally cleanup (normal + exception)
    lib_h3 = FakeQFLib()
    ckpt_h3 = _make_dummy_ckpt_with_keys(tmp_path / "minimax-h3-scope.safetensors", ["blocks.0.w"])
    xfm_h3 = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h3, lib=lib_h3, torch_dtype=torch.float32)
    pipe_h3 = SimpleNamespace(transformer=xfm_h3, scheduler=_PostIncScheduler())
    try:
        kf = [{"latent": torch.randn(1, 24, 1, 4, 4, dtype=torch.float32), "frame_index": 0}]
        with request_scope(pipe_h3, {}, av_conditions={"keyframes": kf}):
            assert xfm_h3._av_conditions is not None
        assert xfm_h3._av_conditions is None
        assert xfm_h3._av_cond_device_cache == {}

        with pytest.raises(RuntimeError, match="h3_boom"):
            with request_scope(pipe_h3, {}, av_conditions={"keyframes": kf}):
                assert xfm_h3._av_conditions is not None
                raise RuntimeError("h3_boom")
        assert xfm_h3._av_conditions is None
        assert xfm_h3._av_cond_device_cache == {}
    finally:
        xfm_h3.close()


# ── Test 44: request_scope concurrency, busy_timeout & MultiStageRunner dedup ──


def test_request_scope_concurrency_busy_timeout_and_runner_dedup(tmp_path: Path) -> None:
    from quantfunc.accelerate.serving import (
        apply_runtime_overrides,
        request_scope,
        restore_runtime_overrides,
    )
    from quantfunc.core.overrides import RuntimeOverrides
    from quantfunc.native.errors import EngineBusyError, UnsupportedError

    pipe = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    pipe.transformer.set_quality_enhance(False)
    pipe.transformer.set_cache(block_cache=0.01, step_cache=0.02)

    # Invalid busy_timeout rejected before lock
    for bad_t in (-0.1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="busy_timeout"):
            with request_scope(pipe, {}, busy_timeout=bad_t):  # type: ignore[arg-type]
                pass

    # Concurrency serialization & busy_timeout
    t1_entered = threading.Event()
    t1_release = threading.Event()
    t1_observed: dict[str, Any] = {}
    t2_errors: list[BaseException] = []
    t3_observed: dict[str, Any] = {}

    def _thread1() -> None:
        with request_scope(pipe, {"quality_enhance": True, "fbcache": 0.07}):
            t1_entered.set()
            assert t1_release.wait(timeout=5.0)
            # Thread 2's rejected call must NOT have mutated Thread 1's active dials!
            t1_observed["qe"] = pipe.transformer.quality_enhance
            t1_observed["bc"] = pipe.transformer.block_cache

    def _thread2() -> None:
        try:
            with request_scope(pipe, {"quality_enhance": False, "fbcache": 0.0}, busy_timeout=0.02):
                pass
        except BaseException as exc:
            t2_errors.append(exc)

    def _thread3() -> None:
        with request_scope(pipe, {"stepcache": 0.11}, busy_timeout=2.0):
            # Must see restored baseline quality_enhance=False & block_cache=0.01
            t3_observed["qe"] = pipe.transformer.quality_enhance
            t3_observed["bc"] = pipe.transformer.block_cache
            t3_observed["sc"] = pipe.transformer.step_cache

    th1 = threading.Thread(target=_thread1)
    th1.start()
    assert t1_entered.wait(timeout=2.0)

    th2 = threading.Thread(target=_thread2)
    th2.start()
    th2.join(timeout=2.0)
    assert len(t2_errors) == 1 and isinstance(t2_errors[0], EngineBusyError)

    # Start Thread 3 waiting on lock before releasing Thread 1
    th3 = threading.Thread(target=_thread3)
    th3.start()
    t1_release.set()
    th1.join(timeout=5.0)
    th3.join(timeout=5.0)

    assert t1_observed == {"qe": True, "bc": pytest.approx(0.07)}
    assert t3_observed == {"qe": False, "bc": pytest.approx(0.01), "sc": pytest.approx(0.11)}
    assert pipe.transformer.step_cache == pytest.approx(0.02)

    # Same-thread nested request_scope (RLock) does not deadlock and nests cleanly
    with request_scope(pipe, {"quality_enhance": True, "fbcache": 0.05}):
        assert pipe.transformer.quality_enhance is True
        with request_scope(pipe, {"quality_enhance": False, "fbcache": 0.09}, busy_timeout=0.0):
            assert pipe.transformer.quality_enhance is False
            assert pipe.transformer.block_cache == pytest.approx(0.09)
        assert pipe.transformer.quality_enhance is True
        assert pipe.transformer.block_cache == pytest.approx(0.05)
    assert pipe.transformer.quality_enhance is False
    assert pipe.transformer.block_cache == pytest.approx(0.01)

    # Deduplicated MultiStageRunner with RuntimeOverrides, fbcache/stepcache aliases & sol dict
    stage_seen: list[dict[str, Any]] = []

    class _RunPipe(_StubPipe):
        def __call__(self, **kw: Any) -> Any:
            stage_seen.append(
                {
                    "bc": self.transformer.block_cache,
                    "sc": self.transformer.step_cache,
                    "sol": self.transformer.sol_config,
                    "loras": list(self.transformer.active_loras),
                }
            )
            return SimpleNamespace(images=torch.ones(1, 4, 8, 8))

    rpipe = _RunPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    rpipe.transformer.set_cache(block_cache=0.01, step_cache=0.01)
    rpipe.transformer.set_sol(None)

    stages = [
        StageSpec(
            name="s1",
            num_inference_steps=2,
            output_type="latent",
            transformer_overrides=RuntimeOverrides(block_cache=0.06),
        ),
        StageSpec(
            name="s2",
            num_inference_steps=2,
            output_type="pil",
            transformer_overrides={
                "fbcache": 0.07,
                "stepcache": 0.12,
                "sol": {"enabled": True, "tau": 1.2},
                "loras": [("mid.safetensors", 0.5)],
            },
        ),
    ]
    MultiStageRunner(stages).run(rpipe, prompt="test")
    assert stage_seen[0]["bc"] == pytest.approx(0.06)
    assert stage_seen[1]["bc"] == pytest.approx(0.07)
    assert stage_seen[1]["sc"] == pytest.approx(0.12)
    assert isinstance(stage_seen[1]["sol"], SolConfig) and stage_seen[1]["sol"].enabled is True
    # Duck-typed _StubTransformer preserves raw tuple list!
    assert stage_seen[1]["loras"] == [("mid.safetensors", 0.5)]
    # All restored after stages
    assert rpipe.transformer.block_cache == pytest.approx(0.01)
    assert rpipe.transformer.step_cache == pytest.approx(0.01)
    assert rpipe.transformer.sol_config is None
    assert rpipe.transformer.active_loras == []

    # Duck-typed _StubTransformer MUST fail closed with UnsupportedError when explicit security policy passed
    for bad_policy_kw in (
        {"allowed_lora_roots": [str(tmp_path)]},
        {"lora_registry": {"style": str(tmp_path / "a.safetensors")}},
        {"strict_lora_registry": True},
    ):
        with pytest.raises(UnsupportedError):
            apply_runtime_overrides(
                rpipe,
                {"loras": [("x.safetensors", 0.5)]},
                allow_extra_attrs=True,
                **bad_policy_kw,
            )

    # Real QuantFuncBaseTransformer configured with strict_registry=True MUST NOT be downgraded by strict_lora_registry=None
    lora_ok = _make_dummy_lora_safetensors(tmp_path / "ok.safetensors", val=1.0)
    lora_unreg = _make_dummy_lora_safetensors(tmp_path / "unreg.safetensors", val=2.0)
    ckpt = _make_dummy_krea2_ckpt(tmp_path / "krea2-strict-inherit.safetensors")
    lib = FakeQFLib()
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        xfm.configure_lora_policy(lora_registry={"ok": str(lora_ok)}, strict_registry=True)
        qpipe = SimpleNamespace(transformer=xfm, scheduler=_PostIncScheduler())
        with pytest.raises(PermissionError):
            apply_runtime_overrides(
                qpipe,
                {"loras": [str(lora_unreg)]},
                allow_extra_attrs=True,
                strict_lora_registry=None,
            )
        saved = apply_runtime_overrides(
            qpipe,
            {"loras": [("ok", 0.6)]},
            allow_extra_attrs=True,
            strict_lora_registry=None,
        )
        assert len(xfm.active_loras) == 1
        restore_runtime_overrides(saved)
        assert xfm.active_loras == []
    finally:
        xfm.close()


# ── Test 45 / Mutation Arm M9: request_scope invariants ─────────────────────


def test_mutation_m9_request_scope_invariants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import quantfunc.accelerate.serving as serving_mod
    from quantfunc.native.errors import EngineBusyError

    # M9(a): Removing lock acquisition (no-op lock) MUST break concurrent busy_timeout=0 -> EngineBusyError
    pipe_a = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)

    class _NoOpLock:
        def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool:
            _ = (blocking, timeout)
            return True

        def release(self) -> None:
            pass

    def _run_concurrent_busy_check(pipe_obj: Any, lock_override: Any | None) -> bool:
        entered = threading.Event()
        release = threading.Event()
        got_busy = False

        def _holder() -> None:
            with serving_mod.request_scope(pipe_obj, {}, lock=lock_override):
                entered.set()
                release.wait(timeout=5.0)

        t = threading.Thread(target=_holder)
        t.start()
        assert entered.wait(timeout=2.0)
        try:
            with serving_mod.request_scope(pipe_obj, {}, lock=lock_override, busy_timeout=0.0):
                pass
        except EngineBusyError:
            got_busy = True
        finally:
            release.set()
            t.join(timeout=5.0)
        return got_busy

    # Mutated (unlocked) fails the concurrency invariant
    with pytest.raises(AssertionError):
        assert _run_concurrent_busy_check(pipe_a, _NoOpLock()) is True
    # Control (real lock) passes
    assert _run_concurrent_busy_check(pipe_a, None) is True

    # M9(b): Skipping reset_request_state() or restore_runtime_overrides() in finally MUST fail post-exception check
    lib_h3 = FakeQFLib()
    ckpt_h3 = _make_dummy_ckpt_with_keys(tmp_path / "h3-m9b.safetensors", ["blocks.0.w"])
    from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel

    xfm_h3 = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h3, lib=lib_h3, torch_dtype=torch.float32)
    pipe_b = SimpleNamespace(transformer=xfm_h3, scheduler=_PostIncScheduler())
    kf = [{"latent": torch.randn(1, 24, 1, 4, 4, dtype=torch.float32), "frame_index": 0}]
    try:
        # Mutate reset_request_state to no-op
        with monkeypatch.context() as mp:
            mp.setattr(serving_mod, "_reset_transformer_request_state", lambda _t: None)
            with pytest.raises(RuntimeError, match="m9b"):
                with serving_mod.request_scope(pipe_b, {"quality_enhance": True}, av_conditions={"keyframes": kf}):
                    raise RuntimeError("m9b")
            with pytest.raises(AssertionError):
                assert xfm_h3._av_conditions is None
        xfm_h3.clear_av_conditions()

        # Mutate restore_runtime_overrides to no-op
        with monkeypatch.context() as mp:
            mp.setattr(serving_mod, "restore_runtime_overrides", lambda _saved: None)
            with pytest.raises(RuntimeError, match="m9b2"):
                with serving_mod.request_scope(pipe_b, {"quality_enhance": True}):
                    raise RuntimeError("m9b2")
            with pytest.raises(AssertionError):
                assert xfm_h3.quality_enhance is False
        xfm_h3.set_quality_enhance(False)

        # Green control: both cleaned up after exception
        with pytest.raises(RuntimeError, match="m9b_ok"):
            with serving_mod.request_scope(pipe_b, {"quality_enhance": True}, av_conditions={"keyframes": kf}):
                raise RuntimeError("m9b_ok")
        assert xfm_h3._av_conditions is None
        assert xfm_h3.quality_enhance is False
    finally:
        xfm_h3.close()

    # M9(c): Breaking sol=None restore branch MUST fail SolConfig restoration assertion
    pipe_c = _StubPipe(scheduler=_PostIncScheduler(), with_free_hooks=False)
    base_sol = SolConfig(enabled=True, tau=1.3, start_percent=0.0, end_percent=0.9)
    pipe_c.transformer.set_sol(None)

    def _broken_restore_ignores_none(saved_records: list[tuple[Any, list[tuple[str, Any]]]]) -> None:
        for transformer, saved in reversed(saved_records):
            for key, prev in reversed(saved):
                if key == "sol" and prev is not None:  # MUTATION: skips restoring None!
                    transformer.set_sol(prev)

    with monkeypatch.context() as mp:
        mp.setattr(serving_mod, "restore_runtime_overrides", _broken_restore_ignores_none)
        with serving_mod.request_scope(pipe_c, {"sol": base_sol}):
            assert pipe_c.transformer.sol_config == base_sol
        with pytest.raises(AssertionError):
            assert pipe_c.transformer.sol_config is None

    # Green control: sol=None baseline restored AND sol=base_sol restored after explicit sol=None override
    pipe_c.transformer.set_sol(None)
    with serving_mod.request_scope(pipe_c, {"sol": base_sol}):
        assert pipe_c.transformer.sol_config == base_sol
    assert pipe_c.transformer.sol_config is None

    pipe_c.transformer.set_sol(base_sol)
    with serving_mod.request_scope(pipe_c, {"sol": None}):
        assert pipe_c.transformer.sol_config is None
    assert pipe_c.transformer.sol_config == base_sol


def test_apply_runtime_overrides_multi_transformer_atomic_rollback_sticky_loras(tmp_path: Path) -> None:
    """Mid-apply failure on a second transformer must atomically roll back transformer 1 (incl. sticky LoRAs)."""
    from quantfunc.accelerate.serving import apply_runtime_overrides
    from quantfunc.core.overrides import RuntimeOverrides

    lora_a = _make_dummy_lora_safetensors(tmp_path / "rb_a.safetensors", val=1.0)
    ckpt1 = _make_dummy_krea2_ckpt(tmp_path / "krea_rb1.safetensors")
    ckpt2 = _make_dummy_krea2_ckpt(tmp_path / "krea_rb2.safetensors")
    t1 = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt1, lib=FakeQFLib(), torch_dtype=torch.float32)
    t2 = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt2, lib=FakeQFLib(), torch_dtype=torch.float32)
    pipe = SimpleNamespace(transformer=t1, transformer_2=t2)
    try:
        orig_set_loras = t2.set_loras

        def _boom_loras(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("t2_lora_bind_failed")

        t2.set_loras = _boom_loras  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="t2_lora_bind_failed"):
            apply_runtime_overrides(
                pipe,
                RuntimeOverrides.from_dict(
                    {
                        "quality_enhance": True,
                        "fbcache": 0.08,
                        "sol": {"enabled": True, "tau": 1.3},
                        "loras": [(str(lora_a), 0.75)],
                    }
                ),
                sticky_loras=True,
            )

        # Transformer 1 must be completely rolled back (including sticky LoRAs and all dials)
        assert t1.quality_enhance is False
        assert t1.block_cache == pytest.approx(0.0)
        assert t1.sol_config is None
        assert t1.active_loras == []
        # Transformer 2 dials mutated before set_loras must also be rolled back
        assert t2.quality_enhance is False
        assert t2.block_cache == pytest.approx(0.0)
        assert t2.sol_config is None
        t2.set_loras = orig_set_loras  # type: ignore[method-assign]
    finally:
        t1.close()
        t2.close()


