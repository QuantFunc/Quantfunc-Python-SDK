"""L4 unit & mutation tests: model pipelines, helpers, CLI (incl. update-engine), and M5."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import stat
import sys
import types
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from conftest import FakeQFLib

from quantfunc import cli
from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel
from quantfunc.native.errors import InvalidArgError
from quantfunc.native.handle import estimate_vram, make_create_params
from quantfunc.pipelines._common import (
    align_size,
    finalize_pipeline_load,
    import_diffusers_class,
)
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
from quantfunc.sampling.bundle import LatentBundle
from quantfunc.sampling.runner import MultiStageRunner
from quantfunc.sampling.upscalers import LatentUpscaler


class _MiniTransformer:
    _is_quantfunc_transformer = True

    def __init__(self) -> None:
        self.quality_enhance = False
        self.close_calls = 0

    def set_quality_enhance(self, enabled: bool) -> None:
        self.quality_enhance = bool(enabled)

    def end_session_if_open(self) -> tuple[bool, bool]:
        self.close_calls += 1
        return (True, True)


class _DummyScheduler:
    def __init__(self) -> None:
        self.timesteps = [1000.0, 500.0]
        self.sigmas = [1.0, 0.5, 0.0]
        self.step_index: int | None = None

    def set_timesteps(self, n: int) -> None:
        self.timesteps = list(range(n, 0, -1))
        self.sigmas = [i / n for i in range(n, -1, -1)]
        self.step_index = None

    def step(self, *args: Any, **kwargs: Any) -> tuple[Any]:
        del args, kwargs
        self.step_index = (self.step_index or 0) + 1
        return (None,)


class _FrozenConfig(Mapping[str, Any]):
    """Minimal immutable Mapping mimicking diffusers FrozenDict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)
        for k, v in data.items():
            object.__setattr__(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


# ── Test 31: L4 loaders, LTX2NeuralLatentUpscaler & two-stage factories ─────


def test_l4_loaders_upscaler_and_stage_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    # (1) Missing upstream pipeline classes in old diffusers raise clear ImportError mentioning diffusers@main
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__version__ = "0.36.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    for loader_fn in (
        load_krea2_pipeline,
        load_qwenimage21_pipeline,
        load_ltx25_pipeline,
        load_minimax_h3_pipeline,
    ):
        with pytest.raises(ImportError, match="diffusers@main"):
            loader_fn("org/model", transformer=_MiniTransformer())

    # (2) Loaders succeed with explicit pipeline_cls and wire hooks / deterministic scheduler
    class _FakePipeCls:
        @classmethod
        def from_pretrained(cls, model_id: str, transformer: Any, **kwargs: Any) -> Any:
            del model_id, kwargs
            pipe = SimpleNamespace(
                transformer=transformer,
                scheduler=_DummyScheduler(),
                model_offload_called=False,
            )
            pipe.scheduler.config = _FrozenConfig({"stochastic_sampling": True})
            pipe.enable_model_cpu_offload = lambda: setattr(pipe, "model_offload_called", True)
            return pipe

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ltx_pipe = load_ltx25_pipeline(
            "Lightricks/LTX-2.5",
            transformer=_MiniTransformer(),
            pipeline_cls=_FakePipeCls,
            offload_mode="model",
        )
    assert ltx_pipe.model_offload_called is True
    assert ltx_pipe.scheduler.config["stochastic_sampling"] is False
    assert len(rec) == 1

    # (3) Krea-2 two-stage factory: 3D packed <-> 4D adapter + Stage-2 quality_enhance override
    krea_xfm = _MiniTransformer()
    krea_seen: list[dict[str, Any]] = []

    class _MockKrea2Pipe:
        def __init__(self) -> None:
            self.transformer = krea_xfm
            self.scheduler = _DummyScheduler()

        def __call__(self, **kwargs: Any) -> Any:
            krea_seen.append(
                {
                    "output_type": kwargs.get("output_type"),
                    "latents_shape": tuple(kwargs["latents"].shape) if "latents" in kwargs else None,
                    "quality_enhance": self.transformer.quality_enhance,
                }
            )
            if kwargs.get("output_type") == "latent":
                # Emit 3D packed latent [B=1, 16*16=256, C=8]
                return SimpleNamespace(images=torch.ones(1, 256, 8))
            return SimpleNamespace(images=[kwargs["latents"]])

    krea_pipe = _MockKrea2Pipe()
    krea_stages = make_krea2_two_stage_stages(
        width=256,
        height=256,
        scale_by=1.5,
        stage1_steps=4,
        stage2_steps=4,
        canvas_multiple=16,
    )
    assert len(krea_stages) == 2
    krea_out = MultiStageRunner(krea_stages).run(krea_pipe, prompt="sunrise")
    assert krea_seen[0]["quality_enhance"] is False
    assert krea_seen[1]["quality_enhance"] is True
    # 16x16 grid scaled 1.5x -> 24x24 = 576 tokens packed back to 3D [1, 576, 8]
    assert krea_seen[1]["latents_shape"] == (1, 576, 8)
    assert tuple(krea_out.images[0].shape) == (1, 576, 8)
    assert krea_xfm.quality_enhance is False  # restored!

    # (4) LTX2NeuralLatentUpscaler: Protocol compliance + NEVER double-divides latents_std
    vae_mock = SimpleNamespace(
        latents_mean=torch.full((4,), 10.0),
        latents_std=torch.full((4,), 2.0),
        config=SimpleNamespace(scaling_factor=1.0),
    )

    class _DoubleSpatialNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_inputs: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen_inputs.append(x.clone())
            return x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    net = _DoubleSpatialNet()
    neural_up = LTX2NeuralLatentUpscaler(net, vae=vae_mock, latents_normalized=False)
    assert isinstance(neural_up, LatentUpscaler)

    vid_in = torch.full((1, 4, 2, 4, 4), 3.0)
    aud_in = torch.full((1, 2, 6, 4), 7.0)
    up_bundle = neural_up(LatentBundle(video_or_image=vid_in, audio=aud_in))
    # Input NOT normalized/divided by latents_std (exact 3.0 preserved!)
    assert torch.allclose(net.seen_inputs[-1], vid_in)
    assert tuple(up_bundle.video_or_image.shape) == (1, 4, 2, 8, 8)
    assert torch.allclose(up_bundle.video_or_image, torch.full((1, 4, 2, 8, 8), 3.0))
    assert torch.equal(up_bundle.audio, aud_in)

    # When latents_normalized=True, denormalizes ONCE (3.0 * 2.0 + 10.0 = 16.0) and NEVER divides by std
    neural_up_norm = LTX2NeuralLatentUpscaler(net, vae=vae_mock, latents_normalized=True)
    up_bundle_norm = neural_up_norm(LatentBundle(video_or_image=vid_in, audio=aud_in))
    assert torch.allclose(net.seen_inputs[-1], torch.full_like(vid_in, 16.0))
    assert torch.allclose(up_bundle_norm.video_or_image, torch.full((1, 4, 2, 8, 8), 16.0))

    # (5) make_ltx25_two_stage_stages: Stage-2 runs output_type="latent" -> restore_frozen_audio -> audio_vae.decode -> vocoder
    events: list[str] = []
    stage1_audio = torch.full((1, 2, 6, 4), 5.0)
    stage2_audio_mutated = torch.full((1, 2, 6, 4), -9.0)
    decoded_audio_input: list[torch.Tensor] = []

    class _MockLTXTransformer(_MiniTransformer):
        def restore_frozen_audio(
            self, s2: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            events.append("restore_frozen_audio")
            return torch.where(mask == 0, s1, s2)

    class _MockLTXPipe:
        def __init__(self) -> None:
            self.transformer = _MockLTXTransformer()
            self.scheduler = _DummyScheduler()
            self.vae = SimpleNamespace(
                dtype=torch.float32,
                config=SimpleNamespace(timestep_conditioning=False),
                decode=self._vae_decode,
            )
            self.audio_vae = SimpleNamespace(dtype=torch.float32, decode=self._audio_vae_decode)
            self.vocoder = self._vocoder
            self.stage_calls: list[dict[str, Any]] = []

        def _vae_decode(self, x: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:
            del return_dict
            events.append("vae.decode")
            return (x,)

        def _audio_vae_decode(self, x: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:
            del return_dict
            events.append("audio_vae.decode")
            decoded_audio_input.append(x.clone())
            return (x * 2.0,)

        def _vocoder(self, mel: torch.Tensor) -> torch.Tensor:
            events.append("vocoder")
            return mel + 1.0

        def __call__(self, **kwargs: Any) -> Any:
            self.stage_calls.append(dict(kwargs))
            assert kwargs.get("output_type") == "latent"
            assert "callback_on_step_end" not in kwargs
            if len(self.stage_calls) == 1:
                events.append("pipe.stage1")
                return SimpleNamespace(frames=vid_in.clone(), audio=stage1_audio.clone())
            events.append("pipe.stage2")
            return SimpleNamespace(frames=kwargs["latents"] * 1.0, audio=stage2_audio_mutated.clone())

    mock_ltx_pipe = _MockLTXPipe()
    ltx_stages = make_ltx25_two_stage_stages(
        upscaler=neural_up,
        width=256,
        height=256,
        num_frames=9,
        stage1_steps=4,
        stage2_steps=3,
        freeze_audio_in_stage2=True,
        output_type="pt",
    )
    ltx_result = MultiStageRunner(ltx_stages).run(mock_ltx_pipe, prompt="waves")
    assert events == [
        "pipe.stage1",
        "pipe.stage2",
        "restore_frozen_audio",
        "vae.decode",
        "audio_vae.decode",
        "vocoder",
    ]
    assert events.index("restore_frozen_audio") < events.index("audio_vae.decode")
    # Stage-1 frozen audio (5.0) was restored before audio_vae.decode!
    assert torch.allclose(decoded_audio_input[0], stage1_audio)
    assert torch.allclose(ltx_result.audio, stage1_audio * 2.0 + 1.0)
    assert mock_ltx_pipe.transformer.quality_enhance is False

    # (6) make_minimax_h3_two_stage_stages: modular adapter avoids decode error on Stage 1
    h3_events: list[str] = []
    h3_stage2_kwargs: dict[str, Any] = {}

    class _MockModularH3Pipe:
        def __init__(self) -> None:
            self.transformer = _MiniTransformer()
            self.scheduler = _DummyScheduler()

        def run_denoise(self, **kwargs: Any) -> dict[str, Any]:
            h3_events.append("run_denoise")
            assert "output_type" not in kwargs
            return {
                "latents": torch.ones(1, 4, 2, 8, 8),
                "audio_latents": torch.full((1, 4, 2, 10), 3.0),
            }

        def __call__(self, **kwargs: Any) -> Any:
            if kwargs.get("output_type") == "latent":
                raise ValueError("VideoDecodeStep does not accept output_type='latent'")
            h3_events.append("full_decode_call")
            h3_stage2_kwargs.update(kwargs)
            return SimpleNamespace(frames=kwargs["latents"], audio=kwargs["audio_latents"])

    h3_pipe = _MockModularH3Pipe()
    h3_stages = make_minimax_h3_two_stage_stages(
        width=256,
        height=256,
        num_frames=17,
        scale_by=1.5,
        total_steps=6,
        split_step=4,
        canvas_multiple=1,
        output_type="pil",
    )
    h3_out = MultiStageRunner(h3_stages).run(
        h3_pipe,
        prompt="rain",
        first_frame="stale_stage1_kf.png",
        prompt_embeds=torch.ones(1, 4, 8),
    )
    assert h3_events == ["run_denoise", "full_decode_call"]
    assert "first_frame" not in h3_stage2_kwargs
    assert "prompt_embeds" in h3_stage2_kwargs
    assert tuple(h3_out.frames.shape) == (1, 4, 2, 12, 12)
    assert torch.allclose(h3_out.audio, torch.full((1, 4, 2, 10), 3.0))


# ── Test 32: encode_qwen3_vl_pre_norm, ensure_deterministic_scheduler, H3 cond


def test_l4_te_scheduler_and_h3_cond_helpers() -> None:
    # (1) encode_qwen3_vl_pre_norm probes model.language_model.norm -> language_model.norm -> model.norm
    class _ScalingNorm(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 100.0 + 5.0

    class _Qwen3VL5x(nn.Module):
        def __init__(self, fail_in_norm: bool = False) -> None:
            super().__init__()
            self.fail_in_norm = fail_in_norm
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.norm = _ScalingNorm()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pre = x + 1.0
            if self.fail_in_norm:
                raise RuntimeError("encoder_boom")
            return self.model.language_model.norm(pre)

    te = _Qwen3VL5x()
    inp = torch.tensor([[[1.0, 2.0, 3.0]]])
    pre_norm = encode_qwen3_vl_pre_norm(te, inp)
    expected_pre = inp + 1.0
    assert torch.allclose(pre_norm, expected_pre)
    assert not torch.allclose(pre_norm, expected_pre * 100.0 + 5.0)
    # Hook MUST be cleaned up
    assert len(te.model.language_model.norm._forward_pre_hooks) == 0

    # Hook MUST be cleaned up even on exception
    te_fail = _Qwen3VL5x(fail_in_norm=True)
    with pytest.raises(RuntimeError, match="encoder_boom"):
        encode_qwen3_vl_pre_norm(te_fail, inp)
    assert len(te_fail.model.language_model.norm._forward_pre_hooks) == 0

    # Fallback paths: language_model.norm and model.norm
    class _AltTE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.norm = _ScalingNorm()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.norm(x * 2.0)

    assert torch.allclose(encode_qwen3_vl_pre_norm(_AltTE(), inp), inp * 2.0)

    # Missing norm layer fails closed
    with pytest.raises(AttributeError, match="model.language_model.norm"):
        encode_qwen3_vl_pre_norm(nn.Linear(3, 3), inp)

    # (2) ensure_deterministic_scheduler neutralizes eta, s_churn, config.stochastic_sampling with 1 warning
    class _StochasticSched:
        def __init__(self) -> None:
            self.eta = 0.5
            self.s_churn = 0.25
            self.config = _FrozenConfig({"stochastic_sampling": True, "eta": 0.5})

    pipe = SimpleNamespace(scheduler=_StochasticSched())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ensure_deterministic_scheduler(pipe)
    assert len(rec) == 1
    assert pipe.scheduler.eta == 0.0
    assert pipe.scheduler.s_churn == 0.0
    assert pipe.scheduler.config["stochastic_sampling"] is False

    # Calling again emits ZERO warnings
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        ensure_deterministic_scheduler(pipe)
    assert len(rec2) == 0

    # (3) reuse_h3_stage2_conditioning keeps text/tags, drops stale spatial keys without mutating input
    stage1_cond = {
        "prompt_embeds": torch.ones(1, 8, 16),
        "text_token_tags": [0, 0, 1, 1],
        "attention_mask": torch.ones(1, 8),
        "keyframes": [{"frame_index": 0}],
        "refs": [{"kind": "image"}],
        "first_frame": "f0.png",
        "last_frame": "f1.png",
        "av_conds": ["stale"],
        "custom_spatial": "drop_me",
    }
    reused = reuse_h3_stage2_conditioning(stage1_cond, drop_spatial_keys=["custom_spatial"])
    assert set(reused.keys()) == {"prompt_embeds", "text_token_tags", "attention_mask"}
    assert "keyframes" in stage1_cond  # original not mutated


# ── Test 33: CLI all subcommands (pos & neg, incl. update-engine) ───────────


def test_cli_all_subcommands_pos_neg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # (1) doctor (offline default: does NOT call check_engine_update)
    update_called = False

    def _mock_check_update(**kwargs: Any) -> dict[str, Any]:
        nonlocal update_called
        update_called = True
        return {
            "installed_version": "0.0.18",
            "latest_version": "0.0.19",
            "update_available": True,
        }

    monkeypatch.setattr(cli, "check_engine_update", _mock_check_update)
    monkeypatch.setattr(cli, "engine_install_status", lambda: ("ready", "0.0.18"))
    monkeypatch.setattr(cli, "default_engine_dir", lambda: tmp_path / "engine")

    rc = cli.main(["doctor", "--json"])
    assert rc == 0
    assert update_called is False
    doc_payload = json.loads(capsys.readouterr().out)
    assert "sdk_version" in doc_payload
    assert "engine_update" not in doc_payload

    # doctor --check-update
    rc = cli.main(["doctor", "--check-update", "--json"])
    assert rc == 0
    assert update_called is True
    doc_upd = json.loads(capsys.readouterr().out)
    assert doc_upd["engine_update"]["installed_version"] == "0.0.18"
    assert doc_upd["engine_update"]["latest_version"] == "0.0.19"
    assert doc_upd["engine_update"]["update_available"] is True

    # (2) install-engine (pos: marker dict, skipped None, legacy str, & neg)
    sample_marker = {
        "version": "0.0.18",
        "set": "sm89",
        "cuda": 12,
        "sms": [89],
        "host": "libquantfunc.so",
        "kernel": "libquantfunc_kernels.so",
        "sha256": {
            "libquantfunc.so": "0" * 64,
            "libquantfunc_kernels.so": "1" * 64,
        },
    }
    monkeypatch.setattr(cli, "install_engine", lambda **kw: dict(sample_marker))
    assert cli.main(["install-engine"]) == 0
    out_inst_text = capsys.readouterr().out
    assert "Engine ready" in out_inst_text
    assert "0.0.18" in out_inst_text and "sm89" in out_inst_text

    assert cli.main(["install-engine", "--json"]) == 0
    out_inst_json = json.loads(capsys.readouterr().out)
    assert out_inst_json["status"] == "installed"
    assert out_inst_json["version"] == "0.0.18"
    assert out_inst_json["gpu_set"] == "sm89"
    assert out_inst_json["cuda"] == 12
    assert out_inst_json["installed"] == sample_marker
    assert out_inst_json["installed_path"].endswith("0.0.18-sm89-cu12/libquantfunc.so")

    # install-engine returns None when .dev_lib_lock / QF_NATIVE_SO_PATH skips install
    monkeypatch.setattr(cli, "install_engine", lambda **kw: None)
    monkeypatch.setattr(cli, "engine_install_status", lambda: ("local", ".dev_lib_lock keeps the local build"))
    assert cli.main(["install-engine"]) == 0
    out_skip_text = capsys.readouterr().out
    assert "skipped" in out_skip_text
    assert ".dev_lib_lock" in out_skip_text
    assert "Engine ready at: None" not in out_skip_text

    assert cli.main(["install-engine", "--json"]) == 0
    out_skip_json = json.loads(capsys.readouterr().out)
    assert out_skip_json["status"] == "skipped"
    assert out_skip_json["installed"] is None
    assert out_skip_json["version"] is None
    assert ".dev_lib_lock" in out_skip_json["reason"]

    # install-engine also tolerates legacy str/Path gracefully
    monkeypatch.setattr(cli, "install_engine", lambda **kw: str(tmp_path / "libquantfunc.so"))
    assert cli.main(["install-engine"]) == 0
    assert "Engine ready" in capsys.readouterr().out

    monkeypatch.setattr(
        cli, "install_engine", lambda **kw: (_ for _ in ()).throw(RuntimeError("network down"))
    )
    assert cli.main(["install-engine"]) == 1
    assert "network down" in capsys.readouterr().err

    # (3) update-engine --check and update-engine (pos: marker dict & skipped None, & neg)
    assert cli.main(["update-engine", "--check"]) == 0
    out_check = capsys.readouterr().out
    assert "update available" in out_check
    assert "0.0.18" in out_check and "0.0.19" in out_check

    captured_install_kw: dict[str, Any] = {}

    def _capture_install_marker(**kw: Any) -> dict[str, Any]:
        captured_install_kw.update(kw)
        return dict(sample_marker, version="0.0.19")

    monkeypatch.setattr(cli, "install_engine", _capture_install_marker)
    assert cli.main(["update-engine", "--force", "--device", "1"]) == 0
    assert captured_install_kw["device_idx"] == 1
    assert captured_install_kw["force"] is True
    out_upd_text = capsys.readouterr().out
    assert "Engine updated" in out_upd_text and "0.0.19" in out_upd_text

    assert cli.main(["update-engine", "--force", "--device", "1", "--json"]) == 0
    out_upd_json = json.loads(capsys.readouterr().out)
    assert out_upd_json["status"] == "installed"
    assert out_upd_json["version"] == "0.0.19"
    assert out_upd_json["installed"]["version"] == "0.0.19"
    assert out_upd_json["force"] is True

    monkeypatch.setattr(cli, "install_engine", lambda **kw: None)
    monkeypatch.setattr(cli, "engine_install_status", lambda: ("local", ".dev_lib_lock keeps the local build"))
    assert cli.main(["update-engine", "--json"]) == 0
    out_upd_skip = json.loads(capsys.readouterr().out)
    assert out_upd_skip["status"] == "skipped"
    assert out_upd_skip["installed"] is None
    assert ".dev_lib_lock" in out_upd_skip["reason"]

    assert cli.main(["update-engine"]) == 0
    out_upd_skip_text = capsys.readouterr().out
    assert "skipped" in out_upd_skip_text
    assert ".dev_lib_lock" in out_upd_skip_text

    monkeypatch.setattr(
        cli, "check_engine_update", lambda **kw: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    assert cli.main(["update-engine", "--check"]) == 1
    assert "offline" in capsys.readouterr().err

    # (4) convert-lora (pos & neg)
    src_lora = tmp_path / "in.safetensors"
    dst_lora = tmp_path / "out.safetensors"
    src_lora.write_bytes(b"placeholder")
    monkeypatch.setattr(cli, "convert_lora_file", lambda i, o, model_path=None: str(o))
    assert cli.main(["convert-lora", "--in", str(src_lora), "--out", str(dst_lora)]) == 0
    capsys.readouterr()

    assert cli.main(["convert-lora", "--in", str(tmp_path / "missing.safetensors"), "--out", str(dst_lora)]) == 1
    capsys.readouterr()

    monkeypatch.setattr(
        cli,
        "convert_lora_file",
        lambda i, o, model_path=None: (_ for _ in ()).throw(ValueError("DoRA unsupported")),
    )
    assert cli.main(["convert-lora", "--in", str(src_lora), "--out", str(dst_lora)]) == 1
    assert "DoRA unsupported" in capsys.readouterr().err

    # (5) estimate-vram against REAL unmocked make_create_params + estimate_vram + FakeQFLib
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    fake_lib = FakeQFLib()
    monkeypatch.setattr(cli, "load_lib", lambda **kw: fake_lib)

    parsed_vram = cli.build_parser().parse_args(["estimate-vram", str(weights), "--arch", "krea2"])
    assert parsed_vram.offload_mode == "auto"

    assert cli.main(["estimate-vram", str(weights), "--arch", "krea2", "--json"]) == 0
    est_data = json.loads(capsys.readouterr().out)
    assert est_data["bytes"] == fake_lib.estimate_resident_bytes
    assert est_data["min_offload_bytes"] == fake_lib.estimate_min_offload_bytes
    assert est_data["working_set_floor_bytes"] == fake_lib.estimate_working_floor_bytes

    assert cli.main(["estimate-vram", str(weights), "--arch", "krea2"]) == 0
    est_text = capsys.readouterr().out
    assert "MiB" in est_text

    # Invalid offload-mode "none" must be rejected by CLI parser choices
    with pytest.raises(SystemExit) as bad_offload_exc:
        cli.main(["estimate-vram", str(weights), "--offload-mode", "none"])
    assert bad_offload_exc.value.code != 0
    capsys.readouterr()

    assert cli.main(["estimate-vram", str(tmp_path / "nope.safetensors")]) == 1
    capsys.readouterr()

    # (6) auth set-key (rejects positional arg, accepts --stdin, never echoes raw key, POSIX 0o600)
    synthetic_valid_key = "qf_" + ("0123456789abcdef" * 4)
    keyfile = tmp_path / "cfg" / "config.json"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["auth", "set-key", synthetic_valid_key])
    assert exc_info.value.code != 0
    capsys.readouterr()

    monkeypatch.setattr(sys, "stdin", io.StringIO(synthetic_valid_key + "\n"))
    rc = cli.main(["auth", "set-key", "--stdin", "--keyfile", str(keyfile)])
    assert rc == 0
    out_auth = capsys.readouterr().out
    assert synthetic_valid_key not in out_auth
    assert keyfile.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(keyfile).st_mode) == 0o600

    # auth status never prints raw key
    monkeypatch.delenv("QUANTFUNC_API_KEY", raising=False)
    rc = cli.main(["auth", "status", "--keyfile", str(keyfile), "--json"])
    assert rc == 0
    status_out = capsys.readouterr().out
    assert synthetic_valid_key not in status_out
    status_json = json.loads(status_out)
    assert status_json["configured"] is True

    # Invalid key rejected and raw input NOT echoed to stderr
    bad_key = "qf_not_a_valid_key_secret_token_xyz"
    monkeypatch.setattr(sys, "stdin", io.StringIO(bad_key + "\n"))
    rc = cli.main(["auth", "set-key", "--stdin", "--keyfile", str(keyfile)])
    assert rc == 1
    err_out = capsys.readouterr().err
    assert bad_key not in err_out


# ── Test 34 / Mutation M5: L4 helper mutations ──────────────────────────────


def test_mutation_l4_helpers() -> None:
    # M5(a): ignoring config.stochastic_sampling=True MUST fail deterministic check
    class _SchedStoch:
        def __init__(self) -> None:
            self.eta = 0.0
            self.s_churn = 0.0
            self.config = _FrozenConfig({"stochastic_sampling": True})

    def _mutated_ensure_det(pipe_or_sched: Any) -> Any:
        sched = getattr(pipe_or_sched, "scheduler", pipe_or_sched)
        if getattr(sched, "eta", 0.0):
            sched.eta = 0.0
        # MUTATION: ignores sched.config.stochastic_sampling!
        return sched

    sched_mut = _SchedStoch()
    _mutated_ensure_det(SimpleNamespace(scheduler=sched_mut))
    with pytest.raises(AssertionError):
        assert sched_mut.config["stochastic_sampling"] is False

    # Control: real ensure_deterministic_scheduler passes
    sched_ok = _SchedStoch()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ensure_deterministic_scheduler(SimpleNamespace(scheduler=sched_ok))
    assert sched_ok.config["stochastic_sampling"] is False

    # M5(b): returning post-norm output instead of pre-norm input MUST fail pre-norm check
    class _ScalingNorm(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 50.0

    class _TE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.norm = _ScalingNorm()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.language_model.norm(x + 2.0)

    te = _TE()
    inp = torch.tensor([[[1.0, 2.0]]])
    expected_pre = inp + 2.0

    def _mutated_post_norm(text_encoder: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        return text_encoder(*args, **kwargs)  # MUTATION: returns post-norm!

    post_out = _mutated_post_norm(te, inp)
    with pytest.raises(AssertionError):
        assert torch.allclose(post_out, expected_pre)

    # Control: real encode_qwen3_vl_pre_norm passes
    assert torch.allclose(encode_qwen3_vl_pre_norm(te, inp), expected_pre)

    # M5(c): invalid offload_mode="none" MUST fail real make_create_params + estimate_vram(FakeQFLib)
    fake_lib = FakeQFLib()
    bad_params = make_create_params(
        weights_path="/tmp/fake_krea2.safetensors",
        device_id=0,
        config_json={"model_arch": "krea2", "offload_mode": "none", "vram_budget_mb": 0},
    )
    with pytest.raises(InvalidArgError, match="offload_mode"):
        estimate_vram(fake_lib, bad_params)

    # Control: default offload_mode="auto" passes real FakeQFLib estimate_vram
    good_params = make_create_params(
        weights_path="/tmp/fake_krea2.safetensors",
        device_id=0,
        config_json={"model_arch": "krea2", "offload_mode": "auto", "vram_budget_mb": 0},
    )
    est = estimate_vram(fake_lib, good_params)
    assert est.bytes == fake_lib.estimate_resident_bytes


# ── Test 35: L4 _common helpers & examples conformance ──────────────────────


def _load_example_module(name: str) -> types.ModuleType:
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    path = examples_dir / name
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_l4_common_helpers_and_loader_auto_install(monkeypatch: pytest.MonkeyPatch) -> None:
    # align_size positive & negative bounds
    assert align_size(1024 * 1.5, 16) == 1536
    assert align_size(5, 16) == 16
    for bad_mult in (0, -8, True):
        with pytest.raises(ValueError, match="multiple"):
            align_size(128, bad_mult)  # type: ignore[arg-type]

    # import_diffusers_class: missing module, fallback resolution, missing symbol
    fake_mod = types.ModuleType("_fake_diffusers_common")
    fake_mod.__version__ = "0.36.0"  # type: ignore[attr-defined]
    fake_mod.FallbackPipe = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_fake_diffusers_common", fake_mod)
    assert (
        import_diffusers_class(
            "_fake_diffusers_common",
            "MissingPrimary",
            fallback_names=("FallbackPipe",),
        )
        is object
    )
    with pytest.raises(ImportError, match="diffusers@main"):
        import_diffusers_class("_fake_diffusers_common", "NopePipe", repo_hint="https://example.invalid")
    with pytest.raises(ImportError, match="nonexistent_diffusers_pkg"):
        import_diffusers_class("nonexistent_diffusers_pkg", "AnyPipe")

    # finalize_pipeline_load wires hooks (with primary_scheduler) & offload cleanly
    alt_sched = _DummyScheduler()
    pipe = SimpleNamespace(
        transformer=_MiniTransformer(),
        scheduler=_DummyScheduler(),
        offloaded=False,
    )
    pipe.enable_model_cpu_offload = lambda: setattr(pipe, "offloaded", True)
    res = finalize_pipeline_load(pipe, attach_hooks=True, offload="model", primary_scheduler=alt_sched)
    assert res is pipe
    assert pipe.offloaded is True
    assert "set_timesteps" in alt_sched.__dict__
    assert "set_timesteps" not in pipe.scheduler.__dict__

    # Verify load_krea2_pipeline does NOT override from_pretrained's auto_install default to False
    seen_from_pretrained_kwargs: dict[str, Any] = {}
    import quantfunc.pipelines.krea2 as krea2_mod

    class _SpyKrea2Xfm:
        @classmethod
        def from_pretrained(cls, path: str, **kw: Any) -> Any:
            seen_from_pretrained_kwargs.update(kw)
            return _MiniTransformer()

    class _SpyPipe:
        @classmethod
        def from_pretrained(cls, model_id: str, transformer: Any, **kw: Any) -> Any:
            del model_id, kw
            return SimpleNamespace(transformer=transformer, scheduler=_DummyScheduler())

    monkeypatch.setattr(krea2_mod, "QuantFuncKrea2Transformer2DModel", _SpyKrea2Xfm)
    load_krea2_pipeline("org/krea2", pipeline_cls=_SpyPipe, attach_hooks=False)
    assert "auto_install" not in seen_from_pretrained_kwargs


def test_examples_qwen_pre_norm_and_minimax_vae_encode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    # (1) examples/qwen_image21_t2i.py: under --use-pre-norm passes BOTH prompt_embeds AND prompt_embeds_mask
    qwen_ex = _load_example_module("qwen_image21_t2i.py")

    class _ScalingNorm(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 10.0

    class _FakeQwenTE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.norm = _ScalingNorm()

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            del attention_mask
            pre = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) + 1.0
            return self.model.language_model.norm(pre)

    attn_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    captured_pipe_kwargs: dict[str, Any] = {}

    class _FakeQwenPipe:
        def __init__(self) -> None:
            self.text_encoder = _FakeQwenTE()

        def tokenizer(self, prompt: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            del prompt, return_tensors
            return {
                "input_ids": torch.tensor([[10, 20, 30, 0]], dtype=torch.long),
                "attention_mask": attn_mask.clone(),
            }

        def __call__(self, **kw: Any) -> Any:
            captured_pipe_kwargs.update(kw)
            return SimpleNamespace(images=[Image.new("RGB", (8, 8))])

    out_png = tmp_path / "qwen_out.png"
    monkeypatch.setattr(
        qwen_ex,
        "parse_args",
        lambda: SimpleNamespace(
            model_id="Qwen/Qwen-Image-2.1",
            transformer="fake.safetensors",
            lora=None,
            lora_scale=1.0,
            prompt="lighthouse",
            width=64,
            height=64,
            steps=2,
            seed=7,
            use_pre_norm=True,
            quality_enhance=True,
            offload="none",
            output=str(out_png),
        ),
    )
    monkeypatch.setattr(
        qwen_ex.QuantFuncQwenImage21Transformer2DModel,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _MiniTransformer()),
    )
    monkeypatch.setattr(qwen_ex, "load_qwenimage21_pipeline", lambda *a, **kw: _FakeQwenPipe())
    qwen_ex.main()

    assert "prompt_embeds" in captured_pipe_kwargs
    assert "prompt_embeds_mask" in captured_pipe_kwargs
    assert torch.equal(captured_pipe_kwargs["prompt_embeds_mask"], attn_mask)
    assert out_png.is_file()

    # (2) examples/minimax_h3_ref2va.py, minimax_h3_ref2va_two_stage.py, minimax_h3_fl2va_two_stage.py
    #     encode RGB images through pipe.vae.encode(...) to C=24 latents (never pass raw 3-channel RGB!)
    img_path = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(img_path)

    vae_inputs: list[tuple[int, ...]] = []

    class _FakeH3VAE:
        device = torch.device("cpu")
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, x: torch.Tensor) -> Any:
            vae_inputs.append(tuple(x.shape))
            b, _c, t, h, w = x.shape
            lat = torch.ones((b, 24, t, max(1, h // 16), max(1, w // 16)), dtype=x.dtype)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: lat))

    fake_h3_pipe = SimpleNamespace(vae=_FakeH3VAE())
    no_vae_pipe = SimpleNamespace(vae=None)

    ref2va_ex = _load_example_module("minimax_h3_ref2va.py")
    lat_ref = ref2va_ex._encode_image_latent(fake_h3_pipe, str(img_path), 32, 32)
    assert tuple(lat_ref.shape) == (1, 24, 1, 2, 2)
    assert vae_inputs[-1] == (1, 3, 1, 32, 32)
    with pytest.raises(RuntimeError, match="VAE"):
        ref2va_ex._encode_image_latent(no_vae_pipe, str(img_path), 32, 32)

    ref2va_2s_ex = _load_example_module("minimax_h3_ref2va_two_stage.py")
    lat_ref_2s = ref2va_2s_ex._encode_ref_for_size(fake_h3_pipe, str(img_path), 32, 32)
    assert tuple(lat_ref_2s.shape) == (1, 24, 1, 2, 2)
    assert vae_inputs[-1] == (1, 3, 1, 32, 32)
    with pytest.raises(RuntimeError, match="VAE"):
        ref2va_2s_ex._encode_ref_for_size(no_vae_pipe, str(img_path), 32, 32)

    fl2va_2s_ex = _load_example_module("minimax_h3_fl2va_two_stage.py")
    kfs = fl2va_2s_ex._build_keyframes(fake_h3_pipe, str(img_path), str(img_path), 32, 32, 81)
    assert len(kfs) == 2
    assert kfs[0]["frame_index"] == 0 and kfs[1]["frame_index"] == 80
    assert tuple(kfs[0]["latent"].shape) == (1, 24, 1, 2, 2)
    assert tuple(kfs[1]["latent"].shape) == (1, 24, 1, 2, 2)
    with pytest.raises(RuntimeError, match="VAE"):
        fl2va_2s_ex._encode_frame_latent(no_vae_pipe, str(img_path), 32, 32)

    # Confirm VAE-encoded latents pass real QuantFuncMiniMaxH3Transformer3DModel.set_av_conditions
    # while raw 3-channel RGB is rejected (mutation/regression proof).
    h3_ckpt = tmp_path / "minimax-h3-quantfunc.safetensors"
    h3_ckpt.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    h3_xfm = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(
        h3_ckpt,
        lib=FakeQFLib(),
        torch_dtype=torch.float32,
    )
    try:
        h3_xfm.set_av_conditions(refs=[{"kind": "image", "latent": lat_ref}])
        h3_xfm.set_av_conditions(refs=[{"kind": "image", "latent": lat_ref_2s}])
        h3_xfm.set_av_conditions(keyframes=kfs)
        raw_rgb = torch.zeros(1, 3, 1, 32, 32, dtype=torch.float32)
        with pytest.raises(ValueError, match="C=24"):
            h3_xfm.set_av_conditions(refs=[{"kind": "image", "latent": raw_rgb}])
    finally:
        h3_xfm.close()

    # (3) AST checks: fl2va_two_stage uses set_av_conditions without redundant stage overwrite;
    #     ltx25_t2v_two_stage does not redundantly call ensure_deterministic_scheduler twice.
    examples_dir = Path(__file__).resolve().parent.parent / "examples"

    fl2va_src = (examples_dir / "minimax_h3_fl2va_two_stage.py").read_text(encoding="utf-8")
    assert "set_av_conditions(keyframes=" in fl2va_src
    assert "vae.encode" in fl2va_src
    assert "stages[0].sigmas" not in fl2va_src
    assert "stages[1].sigmas" not in fl2va_src
    assert "stages[1].upscaler" not in fl2va_src

    ltx_src = (examples_dir / "ltx25_t2v_two_stage.py").read_text(encoding="utf-8")
    ltx_tree = ast.parse(ltx_src)
    det_calls = [
        node
        for node in ast.walk(ltx_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ensure_deterministic_scheduler"
    ]
    assert det_calls == []

    ref2va_src = (examples_dir / "minimax_h3_ref2va.py").read_text(encoding="utf-8")
    assert '"audio_latent"' in ref2va_src
