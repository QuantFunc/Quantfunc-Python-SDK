"""L1 unit & M2 mutation tests: Sol, Config/Auth, LoRA, 3-tier Session, Registry & Presets (pure CPU)."""

from __future__ import annotations

import copy
import json
import os
import stat
import struct
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeQFLib

from quantfunc.core import checkpoint, config, lora, overrides, registry, session, sol
from quantfunc.native.capi import QF_DTYPE_BF16, QF_ERR_INTERNAL
from quantfunc.native.handle import QFEngineHandle


def _write_synthetic_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> Path:
    hdr: dict[str, Any] = {}
    if metadata:
        hdr["__metadata__"] = metadata
    blob = bytearray()
    off = 0
    for name, (dtype, shape, raw) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hj += b" " * ((8 - (len(hj) % 8)) % 8)
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + bytes(blob))
    return path


# ── Test 12: SolConfig v2 wire mapping, interpolation, no-sigmas warning, legacy ──


def test_sol_v2_wire_mapping_interpolation_and_legacy() -> None:
    sol._reset_no_sigmas_warning()

    # Disabled Sol returns None
    assert sol.SolConfig(enabled=False).to_native(sigmas=[1.0, 0.5, 0.0]) is None

    # min_sequence_length maps to wire "min_tokens" + all 14 sparse wire keys supported
    cfg = sol.SolConfig(
        enabled=True,
        tau=1.3,
        selector="svg2",
        window_enabled=True,
        start_percent=0.25,
        end_percent=0.75,
        min_sequence_length=8192,
        sink_conditioning="exact_kv_and_rows",
        first_layers_dense=2,
        first_steps_dense=1,
        first_steps_dense_frac=0.2,
        plan_interval=2,
        auto_dense_guard=True,
        require_permute=False,
        top_p=0.9,
        cdf=0.95,
    )
    # Uniform schedule interpolation
    native = cfg.to_native(sigmas=[1.0, 0.5, 0.0])
    assert native is not None
    assert native["min_tokens"] == 8192
    assert "min_sequence_length" not in native
    assert native["window"] == pytest.approx({"sigma_start": 0.75, "sigma_end": 0.25})
    assert set(native.keys()) == {
        "enabled",
        "selector",
        "tau",
        "top_p",
        "cdf",
        "min_tokens",
        "first_layers_dense",
        "first_steps_dense",
        "first_steps_dense_frac",
        "plan_interval",
        "auto_dense_guard",
        "require_permute",
        "window",
        "sink_conditioning",
    }

    # Non-uniform schedule interpolation
    assert sol.percent_to_sigma([1.0, 0.8, 0.0], 0.0) == pytest.approx(1.0)
    assert sol.percent_to_sigma([1.0, 0.8, 0.0], 0.25) == pytest.approx(0.9)
    assert sol.percent_to_sigma([1.0, 0.8, 0.0], 0.5) == pytest.approx(0.8)
    assert sol.percent_to_sigma([1.0, 0.8, 0.0], 0.75) == pytest.approx(0.4)
    assert sol.percent_to_sigma([1.0, 0.8, 0.0], 1.0) == pytest.approx(0.0)

    # to_native(sigmas=None) warns ONCE and degrades gracefully with window=None
    sol._reset_no_sigmas_warning()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n1 = cfg.to_native(sigmas=None)
        n2 = cfg.to_native(sigmas=None)
    assert n1 is not None and n1["window"] is None
    assert n2 is not None and n2["window"] is None
    assert len(caught) == 1

    # Validation refusals
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, tau=5.0)
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, tau=float("nan"))
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, tau=float("inf"))
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, start_percent=0.8, end_percent=0.2)
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, sink_conditioning="unknown_mode")
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, selector="bad_selector")
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, plan_interval=0)

    # Legacy v1 Acklam migration parity
    off1 = sol.from_legacy(1.0)
    assert off1.enabled is False
    assert sol.from_legacy(0.0).enabled is False
    half = sol.from_legacy(0.5)
    assert half.enabled is True
    assert half.tau == pytest.approx(0.0, abs=1e-5)
    assert half.window_enabled is False
    assert half.min_sequence_length == 0
    assert half.sink_conditioning == "off"
    with pytest.raises(ValueError):
        sol.from_legacy(1.5)


# ── Test 13: AttentionBackend, CacheConfig, EngineConfig, QF_SERVER_URL ──────


def test_config_validation_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config.ATTENTION_BACKENDS == ("auto", "int8", "i4xi8", "int4", "fp16")
    assert config.normalize_attention_backend(None) == "auto"
    assert config.normalize_attention_backend("INT4_PV8") == "i4xi8"
    assert config.normalize_attention_backend("sage") == "int8"
    assert config.normalize_attention_backend("fp16_native") == "fp16"
    assert config.normalize_attention_backend("native") == "fp16"
    assert config.normalize_attention_backend("flash") == "fp16"
    with pytest.raises(ValueError):
        config.normalize_attention_backend("nonexistent_backend")

    # CacheConfig boundaries
    cc = config.CacheConfig(step_cache=0.12, block_cache=0.05)
    assert cc.to_session_options() == {"step_cache_thresh": 0.12, "block_cache_thresh": 0.05}
    for bad in (-0.1, float("nan"), float("inf"), 101.0, True):
        with pytest.raises(ValueError):
            config.CacheConfig(step_cache=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            config.CacheConfig(block_cache=bad)  # type: ignore[arg-type]

    # EngineConfig 12-key serialization + bounds
    ec = config.EngineConfig(
        model_arch="auto",
        attention_backend="int4_pv8",
        offload_mode="block_offload",
        pinned_memory=True,
        vram_budget_mb=16384,
        device_id=1,
        producer_stream_mode="torch_current",
    )
    cd = ec.to_create_dict()
    assert cd["attention_backend"] == "i4xi8"
    assert cd["offload_mode"] == "block_offload"
    assert "device_id" not in cd
    assert "producer_stream_mode" not in cd

    with pytest.raises(ValueError):
        config.EngineConfig(offload_mode="invalid_mode")
    with pytest.raises(ValueError):
        config.EngineConfig(vram_budget_mb=-1)
    with pytest.raises(ValueError):
        config.EngineConfig(vram_budget_mb=config.MAX_VRAM_BUDGET_MB + 1)
    with pytest.raises(ValueError):
        config.EngineConfig(server_url="http://insecure.quantfunc.com")

    # QF_SERVER_URL non-HTTPS rejected
    monkeypatch.setenv("QF_SERVER_URL", "http://plaintext.quantfunc.com")
    with pytest.raises(ValueError):
        config.resolve_auth()


# ── Test 14: AuthConfig precedence, redaction, cross-platform atomic save ────


def test_auth_resolution_redaction_and_atomic_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_file = tmp_path / "secure_cfg" / "config.json"
    k_file = "qf_" + "1" * 64
    k_env = "qf_" + "2" * 64
    k_exp = "qf_" + "3" * 64

    monkeypatch.delenv("QUANTFUNC_API_KEY", raising=False)
    monkeypatch.delenv("QF_SERVER_URL", raising=False)

    # Save key atomically on POSIX -> parent 0o700, file 0o600
    saved = config.save_api_key(k_file, keyfile=key_file)
    assert Path(saved) == key_file.resolve()
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(key_file.parent).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(key_file).st_mode) == 0o600

    # Priority: explicit > QUANTFUNC_API_KEY > config.json
    a1 = config.resolve_auth(keyfile=key_file)
    assert a1.api_key == k_file and a1.source == "config_file"

    monkeypatch.setenv("QUANTFUNC_API_KEY", k_env)
    a2 = config.resolve_auth(keyfile=key_file)
    assert a2.api_key == k_env and a2.source == "env"

    a3 = config.resolve_auth(api_key=k_exp, keyfile=key_file)
    assert a3.api_key == k_exp and a3.source == "explicit"

    # Redaction never exposes secret body in repr or validation error
    assert k_exp not in repr(a3)
    assert config.redact_api_key(k_exp) == "qf_3333...3333"
    bad_secret = "qf_NOT_A_VALID_SECRET_TOKEN_12345"
    with pytest.raises(ValueError) as exc_info:
        config.validate_api_key(bad_secret)
    assert bad_secret not in str(exc_info.value)

    # Atomic failure preserves old config.json and leaves no tmp files
    def _boom_replace(src: Any, dst: Any) -> None:
        raise OSError("simulated replace error")

    monkeypatch.setattr(os, "replace", _boom_replace)
    with pytest.raises(OSError, match="simulated replace error"):
        config.save_api_key(k_exp, keyfile=key_file)
    monkeypatch.undo()
    assert json.loads(key_file.read_text(encoding="utf-8"))["api_key"] == k_file
    assert [p for p in key_file.parent.iterdir() if ".tmp-" in p.name] == []

    # Windows mock branch (os.name == "nt", no O_NOFOLLOW, no fchmod)
    win_keyfile = tmp_path / "win_cfg" / "config.json"
    monkeypatch.setattr(config.os, "name", "nt")
    monkeypatch.delattr(config.os, "O_NOFOLLOW", raising=False)
    monkeypatch.delattr(config.os, "fchmod", raising=False)
    monkeypatch.setattr(config.os, "O_BINARY", 0, raising=False)
    config.save_api_key(k_exp, keyfile=win_keyfile)
    assert json.loads(win_keyfile.read_text(encoding="utf-8"))["api_key"] == k_exp


# ── Test 15: LoRA pure-stdlib converter, refusals, LoRAManager cache & unload ─


def test_lora_converter_manager_and_unload(
    tmp_path: Path,
    fake_qf_lib: FakeQFLib,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_a = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    payload_b = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)
    payload_te = struct.pack("<2f", 9.0, 10.0)

    # Kohya + BFL + TE keys -> canonical diffusers with TE dropped and bytes preserved
    kohya_bfl = _write_synthetic_safetensors(
        tmp_path / "kohya_bfl.safetensors",
        {
            "lora_unet_blocks_0_attn_wq.lora_down.weight": ("F32", [2, 2], payload_a),
            "lora_unet_blocks_0_attn_wq.lora_up.weight": ("F32", [2, 2], payload_b),
            "lora_unet_txtfusion_projector.lora_down.weight": ("F32", [2, 2], payload_a),
            "lora_te1_text_model_layers_0.lora_down.weight": ("F32", [2], payload_te),
        },
        metadata={"lora_adapter_metadata": json.dumps({"r": 8})},
    )
    out_st = tmp_path / "converted.safetensors"
    lora.convert_lora_file(kohya_bfl, out_st)
    hdr, blob = lora._read_safetensors(str(out_st))
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in hdr
    assert "transformer_blocks.0.attn.to_q.lora_B.weight" in hdr
    assert "text_fusion.projector.lora_A.weight" in hdr
    assert not any("lora_te" in k for k in hdr)
    b0, b1 = hdr["transformer_blocks.0.attn.to_q.lora_A.weight"]["data_offsets"]
    assert blob[b0:b1] == payload_a

    # PEFT adapter-name form + --model exact inverse
    ckpt = _write_synthetic_safetensors(
        tmp_path / "model_ckpt.safetensors",
        {"custom_block_x.sub_proj.weight": ("F32", [2, 2], payload_a)},
    )
    kohya_custom = _write_synthetic_safetensors(
        tmp_path / "kohya_custom.safetensors",
        {"lora_unet_custom_block_x_sub_proj.lora_down.weight": ("F32", [2, 2], payload_a)},
    )
    out_custom = tmp_path / "converted_custom.safetensors"
    lora.convert_lora_file(kohya_custom, out_custom, model_path=ckpt)
    hdr_c = checkpoint.sniff_safetensors_header(out_custom)
    assert "custom_block_x.sub_proj.lora_A.weight" in hdr_c

    # Fail-closed refusals: DoRA, LyCORIS, OFT, unmapped key, alpha_pattern on BFL rename
    for fname, bad_tensors, bad_meta in (
        ("dora.safetensors", {"blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], payload_a), "blocks.0.attn.to_q.lora_magnitude_vector": ("F32", [2], payload_te)}, None),
        ("lycoris.safetensors", {"blocks.0.attn.to_q.lokr_w1": ("F32", [2, 2], payload_a)}, None),
        ("oft.safetensors", {"blocks.0.attn.to_q.oft_blocks": ("F32", [2, 2], payload_a)}, None),
        ("unmapped.safetensors", {"blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], payload_a), "blocks.0.attn.to_q.diff": ("F32", [2, 2], payload_a)}, None),
        (
            "alpha_pat.safetensors",
            {"diffusion_model.txtfusion.projector.lora_A.weight": ("F32", [2, 2], payload_a)},
            {"lora_adapter_metadata": json.dumps({"r": 8, "alpha_pattern": {"txtfusion.projector": 4}})},
        ),
    ):
        p = _write_synthetic_safetensors(tmp_path / fname, bad_tensors, metadata=bad_meta)
        with pytest.raises(ValueError):
            lora.convert_lora_file(p, tmp_path / f"out_{fname}")

    # LoRAManager: converts once, hits cache on second load, closes open session, unload sends []
    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    eng.session_begin(512, 512, 4, (1, 16, 64), QF_DTYPE_BF16, {})
    assert eng.current_session is not None

    conv_calls = {"n": 0}
    real_conv = lora.convert_lora_file

    def spy_conv(*a: Any, **kw: Any) -> str:
        conv_calls["n"] += 1
        return real_conv(*a, **kw)

    monkeypatch.setattr(lora, "convert_lora_file", spy_conv)
    mgr = lora.LoRAManager(eng, cache_dir=tmp_path / "lora_cache")
    mgr.load_lora(kohya_bfl, scale=0.75)
    assert eng.current_session is None  # closed open session before qf_xfm_set_loras
    assert conv_calls["n"] == 1
    assert len(fake_qf_lib.loras_history[-1]) == 1
    assert fake_qf_lib.loras_history[-1][0]["scale"] == pytest.approx(0.75)

    # Second load hits disk cache without re-converting
    mgr.load_lora(kohya_bfl, scale=0.5)
    assert conv_calls["n"] == 1

    # Unload closes session and sends []
    eng.session_begin(512, 512, 4, (1, 16, 64), QF_DTYPE_BF16, {})
    mgr.unload_loras()
    assert eng.current_session is None
    assert fake_qf_lib.loras_history[-1] == []
    assert mgr.active_loras == ()
    eng.destroy()


# ── Test 16: AutoSessionTracker + SamplingSession 3-tier + BUSY retry ────────


def test_session_3tier_default_steps_and_busy_retry(fake_qf_lib: FakeQFLib) -> None:
    fake_qf_lib.busy_once = True
    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    tracker = session.AutoSessionTracker(eng, busy_retry_delay_s=0.0)
    assert tracker.tier == 2
    assert tracker.default_steps == session.DEFAULT_AUTO_SESSION_MAX_STEPS == 256

    with pytest.raises(ValueError):
        tracker.set_default_steps(0)
    with pytest.raises(ValueError):
        tracker.set_default_steps(-5)

    tracker.set_default_steps(12)
    assert tracker.default_steps == 12

    def open_fn(num_steps: int, opts: dict[str, Any]) -> Any:
        return eng.session_begin(512, 512, num_steps, (1, 64, 128), QF_DTYPE_BF16, opts)

    # Tier 2 first call hits busy_once -> retries and succeeds with num_steps=12
    step0, total0, key1 = tracker.resolve_step_and_session(0.9, ("g", 512), 16, {"attention_backend": "auto"}, open_fn)
    assert (step0, total0, key1) == (0, 12, 1)
    assert fake_qf_lib.sessions[-1]["num_steps"] == 12

    # Sequential split-CFG at same sigma reuses step_index=0 and assigns ctx_key=2
    step0_b, total0_b, key2 = tracker.resolve_step_and_session(0.9, ("g", 512), 16, {"attention_backend": "auto"}, open_fn)
    assert (step0_b, total0_b, key2) == (0, 12, 2)
    assert len(fake_qf_lib.sessions) == 1

    # Decreasing sigma advances step_index=1
    step1, _, _ = tracker.resolve_step_and_session(0.5, ("g", 512), 16, {"attention_backend": "auto"}, open_fn)
    assert step1 == 1
    assert len(fake_qf_lib.sessions) == 1

    # Increasing sigma triggers clean reopen
    step_new, _, _ = tracker.resolve_step_and_session(0.95, ("g", 512), 16, {"attention_backend": "auto"}, open_fn)
    assert step_new == 0
    assert len(fake_qf_lib.sessions) == 2

    # Geometry change triggers reopen
    tracker.resolve_step_and_session(0.8, ("g", 1024), 16, {"attention_backend": "auto"}, open_fn)
    assert len(fake_qf_lib.sessions) == 3

    # Dial change triggers reopen
    tracker.resolve_step_and_session(0.7, ("g", 1024), 16, {"attention_backend": "int8"}, open_fn)
    assert len(fake_qf_lib.sessions) == 4

    # Tier 1 overrides Tier 2
    tracker.close_if_open()
    tracker.set_staged_schedule(session.StagedSchedule(sigmas=(1.0, 0.5, 0.25, 0.0), num_steps=3, do_cfg=True))
    assert tracker.tier == 1
    st_t1, tot_t1, _ = tracker.resolve_step_and_session(0.5, ("g", 512), 16, {}, open_fn)
    assert (st_t1, tot_t1) == (1, 3)

    # Tier 3 overrides Tier 1 and guarantees session close on exception
    with pytest.raises(RuntimeError, match="user crash"):
        with session.SamplingSession(
            eng,
            tracker=tracker,
            width=512,
            height=512,
            num_steps=5,
            max_context_dims=(1, 64, 128),
            cond_dtype=QF_DTYPE_BF16,
            options={},
        ):
            assert tracker.tier == 3
            assert eng.current_session is not None
            _, tot_t3, _ = tracker.resolve_step_and_session(0.9, ("g", 512), 16, {}, open_fn)
            assert tot_t3 == 5
            raise RuntimeError("user crash")

    assert eng.current_session is None
    assert fake_qf_lib.sessions[-1]["ended"] is True
    eng.destroy()


def test_session_cursor_indexing_empty_sigmas_and_schedule_scale(fake_qf_lib: FakeQFLib) -> None:
    # StagedSchedule validation for timestep_scale and use_cursor_indexing
    for bad_scale in (0.0, -10.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="timestep_scale"):
            session.StagedSchedule(sigmas=(1.0, 0.0), num_steps=1, timestep_scale=bad_scale)  # type: ignore[arg-type]

    default_sched = session.StagedSchedule(sigmas=(1.0, 0.5, 0.0), num_steps=2)
    assert default_sched.timestep_scale is None
    assert default_sched.use_cursor_indexing is False

    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    tracker = session.AutoSessionTracker(eng, busy_retry_delay_s=0.0)

    def open_fn(num_steps: int, opts: dict[str, Any]) -> Any:
        return eng.session_begin(512, 512, num_steps, (1, 64, 128), QF_DTYPE_BF16, opts)

    # 1. Cursor indexing with empty sigmas (DDIMScheduler-style) + split-CFG at same cursor
    ddim_sched = session.StagedSchedule(sigmas=(), num_steps=4, timestep_scale=1000.0, use_cursor_indexing=True)
    tracker.set_staged_schedule(ddim_sched)
    assert tracker.step_cursor == 0

    with pytest.raises(ValueError, match="step_cursor"):
        tracker.set_step_cursor(True)  # type: ignore[arg-type]
    tracker.set_step_cursor(-5)
    assert tracker.step_cursor == 0

    ddim_indices: list[int] = []
    for i, sig in enumerate((0.75, 0.50, 0.25, 0.0)):
        tracker.set_step_cursor(i)
        s_idx, tot, _ = tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, open_fn)
        assert tot == 4
        assert tracker.active_branch_count == 1
        ddim_indices.append(s_idx)
        # Second CFG branch at same step keeps identical step_index and bumps branch count
        s_idx_cfg, _, _ = tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, open_fn)
        assert s_idx_cfg == s_idx
        assert tracker.active_branch_count == 2
    assert ddim_indices == [0, 1, 2, 3]
    assert len(fake_qf_lib.sessions) == 1

    # Overflow cursor clamps to eff_total - 1; close_if_open resets cursor to 0
    tracker.set_step_cursor(99)
    clamped_idx, _, _ = tracker.resolve_step_and_session(0.0, ("g", 512), 16, {}, open_fn)
    assert clamped_idx == 3
    tracker.close_if_open()
    assert tracker.step_cursor == 0

    # 2. HeunDiscreteScheduler-style EDM sigmas with duplicate predictor/corrector sigmas
    heun_sigmas = (25.146, 4.496, 4.496, 1.214, 1.214, 0.029, 0.029, 0.0)
    heun_sched = session.StagedSchedule(
        sigmas=heun_sigmas,
        num_steps=7,
        timestep_scale=1000.0,
        use_cursor_indexing=True,
    )
    tracker.set_staged_schedule(heun_sched)
    heun_norm_ts = (0.999, 0.666, 0.666, 0.333, 0.333, 0.0, 0.0)
    heun_indices: list[int] = []
    heun_branches: list[int] = []
    for i, sig in enumerate(heun_norm_ts):
        tracker.set_step_cursor(i)
        s_idx, tot, _ = tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, open_fn)
        assert tot == 7
        heun_indices.append(s_idx)
        heun_branches.append(tracker.active_branch_count)
    assert heun_indices == [0, 1, 2, 3, 4, 5, 6]
    # Duplicate sigma across predictor/corrector steps must NOT be misclassified as split-CFG
    assert heun_branches == [1, 1, 1, 1, 1, 1, 1]

    # on_session_closed & set_schedule reset cursor to 0
    tracker.set_step_cursor(5)
    tracker.on_session_closed()
    assert tracker.step_cursor == 0
    tracker.set_step_cursor(3)
    tracker.set_schedule(session.StagedSchedule(sigmas=(), num_steps=4, use_cursor_indexing=False))
    assert tracker.step_cursor == 0

    # 3. Direct empty-sigmas StagedSchedule WITHOUT use_cursor_indexing falls back monotonically
    fallback_indices: list[int] = []
    for sig in (0.75, 0.75, 0.50, 0.25, 0.0):
        s_idx, tot, _ = tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, open_fn)
        assert tot == 4
        fallback_indices.append(s_idx)
    assert fallback_indices == [0, 0, 1, 2, 3]

    # 4. FlowMatch 4-step sigmas with final 0.001 maps to exact [0, 1, 2, 3]
    tracker.close_if_open()
    fm_sched = session.StagedSchedule(
        sigmas=(1.0, 0.667, 0.334, 0.001, 0.0),
        num_steps=4,
        timestep_scale=1000.0,
        use_cursor_indexing=False,
    )
    tracker.set_staged_schedule(fm_sched)
    fm_indices = [
        tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, open_fn)[0]
        for sig in (1.0, 0.667, 0.334, 0.001)
    ]
    assert fm_indices == [0, 1, 2, 3]

    eng.destroy()


# ── Test 17: Registry 5th model extensibility + Presets + detect_model_arch ──


def test_registry_5th_model_and_presets(tmp_path: Path) -> None:
    # All 4 bundled presets load cleanly including frame_grid & default_fps
    arches = checkpoint.list_bundled_presets()
    assert set(arches) == {"krea2", "qwenimage21", "ltx25", "minimax_h3"}

    h3 = checkpoint.PresetMetadata.from_arch("minimax_h3")
    assert h3.spatial_compression == 16
    assert h3.default_fps == 24.0
    assert h3.frame_grid is not None
    assert h3.frames_from_latent_t(2) == 5
    assert h3.frames_from_latent_t(7) == 22
    assert h3.default_quality_ratio == 0.8

    ltx = checkpoint.PresetMetadata.from_arch("ltx25")
    assert ltx.spatial_compression == 32
    assert ltx.temporal_compression == 8
    assert ltx.audio_latent_channels == 8
    assert ltx.audio_mel_bins == 16

    # Sibling non-transformer repo metadata config.json must NOT shadow bundled transformer config
    meta_dir = tmp_path / "repo_meta"
    meta_dir.mkdir()
    w_meta = meta_dir / "krea2-quantfunc.safetensors"
    w_meta.write_bytes(b"stub")
    (meta_dir / "config.json").write_text('{"framework": "pytorch"}', encoding="utf-8")
    cfg_meta, path_meta = checkpoint.resolve_transformer_config("krea2", weights_path=w_meta)
    assert cfg_meta["in_channels"] == 64
    assert "presets" in path_meta

    # Partial sibling transformer config merges bundled defaults while preserving overrides
    part_dir = tmp_path / "repo_partial"
    part_dir.mkdir()
    w_part = part_dir / "krea2-quantfunc.safetensors"
    w_part.write_bytes(b"stub")
    (part_dir / "config.json").write_text('{"num_layers": 30}', encoding="utf-8")
    cfg_part, _ = checkpoint.resolve_transformer_config("krea2", weights_path=w_part)
    assert cfg_part["num_layers"] == 30
    assert cfg_part["in_channels"] == 64
    assert ltx.frames_from_latent_t(16) == 121

    qi = checkpoint.PresetMetadata.from_arch("qwenimage21")
    assert qi.default_quality_ratio == 0.85

    k2 = checkpoint.PresetMetadata.from_arch("krea2")
    assert k2.default_quality_ratio == 0.8

    # Auto-detection across filename hints and config dicts
    assert registry.detect_model_arch("my-krea2-quantfunc-int4.safetensors") == "krea2"
    assert registry.detect_model_arch({"_class_name": "QwenImage21Transformer2DModel"}) == "qwenimage21"
    assert registry.detect_model_arch({"image_model": "minimax_h3"}) == "minimax_h3"
    assert registry.detect_model_arch({"_class_name": "AVTransformer3DModel"}) == "ltx25"

    # Unrecognizable input fails closed
    with pytest.raises(ValueError, match="Unable to detect"):
        registry.detect_model_arch("completely_unknown_weights.safetensors")

    # Register 5th custom model with default_config_path & engine_arch="auto"
    custom_cfg = tmp_path / "custom5_config.json"
    custom_cfg.write_text(json.dumps({"_class_name": "Custom5Transformer", "in_channels": 16}), encoding="utf-8")

    @registry.register_model(
        "custom5",
        transformer_classes=("Custom5Transformer",),
        file_hints=("*custom5*",),
        default_config_path=custom_cfg,
        engine_arch="auto",
    )
    class _Custom5Model:
        pass

    try:
        assert "custom5" in registry.list_registered_models()
        spec = registry.get_model_spec("custom5")
        assert spec.engine_arch == "auto"
        assert spec.default_config_path == custom_cfg
        assert registry.get_model_class("custom5") is _Custom5Model
        assert registry.detect_model_arch("weights-custom5-v1.safetensors") == "custom5"
        assert registry.detect_model_arch({"_class_name": "Custom5Transformer"}) == "custom5"
        cfg_dict, cfg_path = checkpoint.resolve_transformer_config(default_config_path=spec.default_config_path)
        assert cfg_dict["in_channels"] == 16
        assert Path(cfg_path) == custom_cfg.resolve()
    finally:
        registry.unregister_model("custom5")


# ── Test 18 (Mutation M2): Breaking L1 rules must turn assertions red ────────


def test_mutation_l1_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    dora_file = _write_synthetic_safetensors(
        tmp_path / "m2_dora.safetensors",
        {
            "blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], payload),
            "blocks.0.attn.to_q.lora_magnitude_vector": ("F32", [2], payload[:8]),
        },
    )

    # M2a: DoRA refusal works normally; mutating unsupported_lora_kind to None breaks refusal
    with pytest.raises(ValueError, match="DoRA"):
        lora.convert_lora_file(dora_file, tmp_path / "out1.safetensors")
    monkeypatch.setattr(lora, "unsupported_lora_kind", lambda keys: None)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(ValueError, match="DoRA"):
            lora.convert_lora_file(dora_file, tmp_path / "out2.safetensors")
    monkeypatch.undo()

    # M2b: SolConfig start_percent > end_percent refusal; mutating __post_init__ breaks refusal
    with pytest.raises(ValueError):
        sol.SolConfig(enabled=True, start_percent=0.9, end_percent=0.1)
    monkeypatch.setattr(sol.SolConfig, "__post_init__", lambda self: None)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(ValueError):
            sol.SolConfig(enabled=True, start_percent=0.9, end_percent=0.1)
    monkeypatch.undo()

    # M2c: save_api_key POSIX 0o600 enforcement; mutating fchmod to 0o644 breaks security check
    if os.name == "posix":
        kf = tmp_path / "m2_key" / "config.json"
        valid_k = "qf_" + "a" * 64
        config.save_api_key(valid_k, keyfile=kf)
        assert stat.S_IMODE(os.stat(kf).st_mode) == 0o600

        real_fchmod = os.fchmod
        monkeypatch.setattr(os, "fchmod", lambda fd, mode: real_fchmod(fd, 0o644))
        kf2 = tmp_path / "m2_key2" / "config.json"
        config.save_api_key(valid_k, keyfile=kf2)
        with pytest.raises(AssertionError):
            assert stat.S_IMODE(os.stat(kf2).st_mode) == 0o600

    # M2d: Disabling cursor indexing on Heun EDM sigmas breaks monotonic [0..6] step indexing
    heun_sigmas = (25.146, 4.496, 4.496, 1.214, 1.214, 0.029, 0.029, 0.0)
    mut_lib = FakeQFLib()
    mut_eng = QFEngineHandle.create(mut_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    try:
        mut_tracker = session.AutoSessionTracker(mut_eng, busy_retry_delay_s=0.0)
        mut_tracker.set_staged_schedule(
            session.StagedSchedule(
                sigmas=heun_sigmas,
                num_steps=7,
                timestep_scale=1000.0,
                use_cursor_indexing=False,  # MUTATION: cursor indexing disabled on EDM sigmas
            )
        )

        def _open(n: int, opts: dict[str, Any]) -> Any:
            return mut_eng.session_begin(512, 512, n, (1, 64, 128), QF_DTYPE_BF16, opts)

        mut_indices = []
        for idx, sig in enumerate((0.999, 0.666, 0.666, 0.333, 0.333, 0.0, 0.0)):
            mut_tracker.set_step_cursor(idx)
            mut_indices.append(mut_tracker.resolve_step_and_session(sig, ("g", 512), 16, {}, _open)[0])
        with pytest.raises(AssertionError):
            assert mut_indices == [0, 1, 2, 3, 4, 5, 6]
    finally:
        mut_eng.destroy()


# ── Test 36: SolConfig.from_dict / to_dict 18-field roundtrip & strict validation ──


def test_sol_from_dict_roundtrip_and_strict_validation() -> None:
    cfg = sol.SolConfig(
        enabled=True,
        tau=1.25,
        selector="meansim",
        window_enabled=True,
        start_percent=0.15,
        end_percent=0.85,
        sigma_start=0.9,
        sigma_end=0.1,
        min_sequence_length=8192,
        sink_conditioning="exact_kv",
        first_layers_dense=2,
        first_steps_dense=1,
        first_steps_dense_frac=0.25,
        plan_interval=3,
        auto_dense_guard=True,
        require_permute=False,
        top_p=0.92,
        cdf=0.96,
    )
    as_dict = cfg.to_dict()
    assert len(as_dict) == 18
    assert sol.SolConfig.from_dict(as_dict) == cfg
    assert sol.SolConfig.from_dict(json.loads(json.dumps(as_dict))) == cfg
    assert sol.SolConfig.from_dict({}) == sol.SolConfig()

    # Wire alias min_tokens -> min_sequence_length (alone or matching canonical)
    via_alias = sol.SolConfig.from_dict({"enabled": True, "min_tokens": 4096})
    assert via_alias.enabled is True
    assert via_alias.min_sequence_length == 4096
    matching = sol.SolConfig.from_dict({"min_tokens": 4096, "min_sequence_length": 4096})
    assert matching.min_sequence_length == 4096

    # Fail-closed refusals
    for bad in (
        {"min_tokens": 4096, "min_sequence_length": 8192},
        {"min_tokens": True},
        {"min_tokens": True, "min_sequence_length": 1},
        {"enabled": True, "foo": 1},
        {1: "bad_key"},
        {"enabled": True, "tau": float("nan")},
        {"enabled": True, "tau": float("inf")},
        {"enabled": True, "start_percent": 0.9, "end_percent": 0.2},
        {"enabled": True, "sigma_start": 0.8},
    ):
        with pytest.raises(ValueError):
            sol.SolConfig.from_dict(bad)  # type: ignore[arg-type]

    for non_mapping in (None, [("enabled", True)], "invalid", 42):
        with pytest.raises(ValueError):
            sol.SolConfig.from_dict(non_mapping)  # type: ignore[arg-type]


# ── Test 37: RuntimeOverrides pure-stdlib, UNSET vs None, aliases & refusals ──


def test_runtime_overrides_aliases_unset_and_refusals() -> None:
    # Pure-stdlib / zero-torch verification in isolated subprocess
    repo_root = Path(__file__).resolve().parent.parent
    probe = """
import sys
for mod in ("diffusers", "transformers", "accelerate", "huggingface_hub", "modelscope", "fastapi", "uvicorn", "torch"):
    sys.modules[mod] = None
from quantfunc.core.overrides import RuntimeOverrides, UNSET, VALID_SAMPLERS
from quantfunc.core.sol import SolConfig
assert VALID_SAMPLERS == ("res_multistep", "euler")
ov = RuntimeOverrides.from_dict({
    "fbcache": 0.05,
    "stepcache": 0.1,
    "sol": {"enabled": True, "min_tokens": 4096},
    "loras": [{"name": "style", "scale": 0.8}],
    "sampler": "euler",
})
assert ov.block_cache == 0.05 and ov.step_cache == 0.1
assert isinstance(ov.sol, SolConfig) and ov.sol.min_sequence_length == 4096
assert ov.sampler == "euler"
assert ov.to_dict()["sampler"] == "euler"
assert ov.quality_enhance is UNSET
"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # UNSET sentinel invariants
    assert bool(overrides.UNSET) is False
    assert repr(overrides.UNSET) == "UNSET"
    assert overrides.UNSET is not None
    assert copy.copy(overrides.UNSET) is overrides.UNSET
    assert copy.deepcopy(overrides.UNSET) is overrides.UNSET
    assert type(overrides.UNSET)() is overrides.UNSET

    # Empty & from_any behavior
    empty = overrides.RuntimeOverrides.from_dict({})
    assert empty.is_empty() is True
    assert empty.to_dict() == {}
    assert empty.sol is overrides.UNSET
    assert empty.loras is overrides.UNSET
    assert empty.sampler is overrides.UNSET
    assert overrides.RuntimeOverrides.from_any(None) == empty
    assert overrides.RuntimeOverrides.from_any(overrides.UNSET) == empty  # type: ignore[arg-type]
    assert overrides.RuntimeOverrides.from_any(empty) is empty

    # Explicit sol=None vs UNSET; explicit loras=[] vs UNSET
    sol_off = overrides.RuntimeOverrides.from_dict({"sol": None})
    assert sol_off.sol is None
    assert sol_off.sol is not overrides.UNSET
    assert sol_off.is_empty() is False
    assert sol_off.to_dict() == {"sol": None}
    assert overrides.RuntimeOverrides.from_dict(sol_off.to_dict()) == sol_off

    clear_loras = overrides.RuntimeOverrides.from_dict({"loras": []})
    assert clear_loras.loras == ()
    assert clear_loras.to_dict() == {"loras": []}

    # Aliases (fbcache/stepcache, matching canonical, attention aliases, LoRA formats)
    full = overrides.RuntimeOverrides.from_dict(
        {
            "quality_enhance": True,
            "fbcache": 0.05,
            "block_cache": 0.05,
            "stepcache": 0.12,
            "step_cache": 0.12,
            "sol": {"enabled": True, "tau": 1.3, "min_tokens": 2048},
            "attention_backend": "int4_pv8",
            "loras": [
                "a.safetensors",
                ["b.safetensors", 0.75],
                ("c.safetensors", -0.25),
                {"name": "style_alias", "scale": 0.6},
                {"path": "same", "name": "same", "scale": 1.0},
            ],
            "default_steps": 24,
            "sampler": "euler",
        }
    )
    assert full.quality_enhance is True
    assert full.block_cache == pytest.approx(0.05)
    assert full.step_cache == pytest.approx(0.12)
    assert isinstance(full.sol, sol.SolConfig) and full.sol.min_sequence_length == 2048
    assert full.attention_backend == "i4xi8"
    assert full.default_steps == 24
    assert full.sampler == "euler"
    assert full.loras == (
        lora.LoRAEntry("a.safetensors", 1.0),
        lora.LoRAEntry("b.safetensors", 0.75),
        lora.LoRAEntry("c.safetensors", -0.25),
        lora.LoRAEntry("style_alias", 0.6),
        lora.LoRAEntry("same", 1.0),
    )
    assert overrides.RuntimeOverrides.from_dict(full.to_dict()) == full
    for valid_sampler in ("euler", "res_multistep"):
        s_ov = overrides.RuntimeOverrides.from_dict({"sampler": valid_sampler})
        assert s_ov.is_empty() is False
        assert s_ov.to_dict() == {"sampler": valid_sampler}
        assert overrides.RuntimeOverrides.from_dict(s_ov.to_dict()) == s_ov
    assert overrides.RuntimeOverrides.from_dict({"attention_backend": "sage"}).attention_backend == "int8"
    assert overrides.RuntimeOverrides.from_dict({"attention_backend": "flash"}).attention_backend == "fp16"

    # Partial to_dict omits UNSET fields
    partial = overrides.RuntimeOverrides(quality_enhance=False, block_cache=0.04)
    assert partial.to_dict() == {"quality_enhance": False, "block_cache": 0.04}

    # Fail-closed refusals
    bad_payloads: list[Any] = [
        {"unknown_dial": 1},
        {1: "non_string_key"},
        {"fbcache": 0.05, "block_cache": 0.1},
        {"stepcache": 0.1, "step_cache": 0.2},
        {"fbcache": True, "block_cache": 1.0},
        {"quality_enhance": None},
        {"quality_enhance": 1},
        {"quality_enhance": "true"},
        {"block_cache": None},
        {"block_cache": -0.1},
        {"block_cache": 100.1},
        {"block_cache": True},
        {"block_cache": float("nan")},
        {"block_cache": float("inf")},
        {"step_cache": None},
        {"step_cache": -1.0},
        {"step_cache": False},
        {"step_cache": float("nan")},
        {"sol": True},
        {"sol": 1},
        {"sol": "enabled"},
        {"sol": ["bad"]},
        {"sol": {"unknown": 1}},
        {"attention_backend": None},
        {"attention_backend": ""},
        {"attention_backend": "   "},
        {"attention_backend": "int8\x00"},
        {"attention_backend": "nonexistent"},
        {"default_steps": None},
        {"default_steps": 0},
        {"default_steps": -5},
        {"default_steps": True},
        {"default_steps": 4.5},
        {"loras": None},
        {"loras": "a.safetensors"},
        {"loras": ""},
        {"loras": b"a.safetensors"},
        {"loras": {"path": "a.safetensors"}},
        {"loras": [123]},
        {"loras": [("only_one",)]},
        {"loras": [("a.safetensors", 1.0, "extra")]},
        {"loras": [("a.safetensors", True)]},
        {"loras": [("a.safetensors", float("nan"))]},
        {"loras": [("a.safetensors", float("inf"))]},
        {"loras": ["a\x00b.safetensors"]},
        {"loras": [{"scale": 0.5}]},
        {"loras": [{"path": "a", "name": "b"}]},
        {"loras": [{"path": "a", "extra": 1}]},
        {"loras": [{"name": "a\x00b"}]},
        {"loras": [{"path": "a", "scale": None}]},
        {"sampler": None},
        {"sampler": "bogus"},
        {"sampler": ""},
        {"sampler": True},
        {"sampler": 123},
    ]
    for bad in bad_payloads:
        with pytest.raises(ValueError, match="."):
            overrides.RuntimeOverrides.from_dict(bad)

    for non_map in (123, ["a"], "str"):
        with pytest.raises(ValueError):
            overrides.RuntimeOverrides.from_dict(non_map)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            overrides.RuntimeOverrides.from_any(non_map)  # type: ignore[arg-type]


# ── Test 38: LoRAManager O(1) stat-cached no-op, scale-only hit, invalidation & force ──


def test_lora_manager_o1_noop_invalidation_and_force(
    tmp_path: Path,
    fake_qf_lib: FakeQFLib,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    lora_a = _write_synthetic_safetensors(
        tmp_path / "lora_a.safetensors",
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw),
            "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [2, 2], raw),
        },
    )
    lora_link = tmp_path / "lora_a_symlink.safetensors"
    lora_link.symlink_to(lora_a)

    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    hdr_calls = {"n": 0}
    rebind_calls = {"n": 0}
    real_read_hdr = lora._read_safetensors_header

    def spy_read_hdr(path: str) -> dict[str, Any]:
        hdr_calls["n"] += 1
        return real_read_hdr(path)

    def spy_before_rebind() -> None:
        rebind_calls["n"] += 1
        eng.end_session_if_open()

    monkeypatch.setattr(lora, "_read_safetensors_header", spy_read_hdr)
    mgr = lora.LoRAManager(
        eng,
        cache_dir=tmp_path / "lora_cache",
        lora_registry={"alias_a": lora_a},
        on_before_rebind=spy_before_rebind,
    )

    try:
        # 1. Empty unload / set_loras([]) on already-empty manager is O(1) no-op
        eng.session_begin(512, 512, 4, (1, 16, 64), QF_DTYPE_BF16, {})
        mgr.unload_loras()
        mgr.set_loras([])
        assert eng.current_session is not None
        assert hdr_calls["n"] == 0
        assert rebind_calls["n"] == 0
        assert len(fake_qf_lib.loras_history) == 0

        # 2. First bind reads header, triggers on_before_rebind, calls qf_xfm_set_loras
        mgr.set_loras([(lora_a, 0.8)])
        assert eng.current_session is None
        assert hdr_calls["n"] == 1
        assert rebind_calls["n"] == 1
        assert len(fake_qf_lib.loras_history) == 1

        # 3. Identical stack via direct path, symlink, relative path, and registry alias -> O(1) no-op
        eng.session_begin(512, 512, 4, (1, 16, 64), QF_DTYPE_BF16, {})
        mgr.set_loras([(lora_a, 0.8)])
        mgr.set_loras([(lora_link, 0.8)])
        mgr.set_loras([{"name": "alias_a", "scale": 0.8}])
        monkeypatch.chdir(tmp_path)
        mgr.set_loras([(Path("lora_a.safetensors"), 0.8)])
        assert eng.current_session is not None  # Session never closed!
        assert hdr_calls["n"] == 1
        assert rebind_calls["n"] == 1
        assert len(fake_qf_lib.loras_history) == 1

        # 4. Scale-only change -> 0 header reads (hits _prepared_cache), triggers rebind + qf_xfm_set_loras
        mgr.set_loras([(lora_a, 0.5)])
        assert eng.current_session is None
        assert hdr_calls["n"] == 1
        assert rebind_calls["n"] == 2
        assert len(fake_qf_lib.loras_history) == 2
        assert fake_qf_lib.loras_history[-1][0]["scale"] == pytest.approx(0.5)

        # 5. In-place disk modification (mtime_ns / size change) invalidates cache & re-reads header
        raw2 = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        _write_synthetic_safetensors(
            lora_a,
            {
                "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 4], raw2),
                "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [4, 2], raw2),
            },
        )
        old_st = os.stat(lora_a)
        os.utime(lora_a, ns=(old_st.st_atime_ns, old_st.st_mtime_ns + 10_000_000))
        mgr.set_loras([(lora_a, 0.5)])
        assert hdr_calls["n"] == 2
        assert rebind_calls["n"] == 3
        assert len(fake_qf_lib.loras_history) == 3

        # 6. Atomic replacement via os.replace (new inode, even with identical size & mtime_ns!) invalidates
        st_before = os.stat(lora_a)
        replacement = _write_synthetic_safetensors(
            tmp_path / "lora_a_replacement.safetensors",
            {
                "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 4], raw2),
                "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [4, 2], raw2),
            },
        )
        os.utime(replacement, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))
        os.replace(replacement, lora_a)
        assert os.stat(lora_a).st_ino != st_before.st_ino
        mgr.set_loras([(lora_a, 0.5)])
        assert hdr_calls["n"] == 3
        assert rebind_calls["n"] == 4
        assert len(fake_qf_lib.loras_history) == 4

        # 7. force=True bypasses both stack cache and _prepared_cache
        mgr.set_loras([(lora_a, 0.5)], force=True)
        assert hdr_calls["n"] == 4
        assert rebind_calls["n"] == 5
        assert len(fake_qf_lib.loras_history) == 5

        # 8. Direct handle bypass (handle.set_loras([])) detected via applied_lora_sig cross-check
        eng.set_loras([])
        assert len(fake_qf_lib.loras_history) == 6
        mgr.set_loras([(lora_a, 0.5)])
        assert len(fake_qf_lib.loras_history) == 7

        # 9. unload_loras sends [] once, then becomes O(1) no-op; unload_loras(force=True) forces call
        mgr.unload_loras()
        assert len(fake_qf_lib.loras_history) == 8
        assert fake_qf_lib.loras_history[-1] == []
        mgr.unload_loras()
        mgr.set_loras([])
        assert len(fake_qf_lib.loras_history) == 8
        mgr.unload_loras(force=True)
        assert len(fake_qf_lib.loras_history) == 9

        # 10. Native qf_xfm_set_loras failure clears _bound_stack_sig so retry is NEVER skipped
        mgr.set_loras([(lora_a, 0.5)])
        assert len(fake_qf_lib.loras_history) == 10
        orig_set = fake_qf_lib.qf_xfm_set_loras._fn

        def _fail_once(xfm: Any, payload: Any) -> int:
            return fake_qf_lib._fail(QF_ERR_INTERNAL, "injected lora bind failure")

        fake_qf_lib.qf_xfm_set_loras._fn = _fail_once
        with pytest.raises(RuntimeError, match="injected lora bind failure"):
            mgr.set_loras([(lora_a, 0.9)])
        assert mgr._bound_stack_sig is None

        fake_qf_lib.qf_xfm_set_loras._fn = orig_set
        # Retrying previously bound scale (0.5) MUST still rebind because cache was cleared on error
        mgr.set_loras([(lora_a, 0.5)])
        assert len(fake_qf_lib.loras_history) == 11
    finally:
        eng.destroy()


# ── Test 39: LoRA registry & allowed_lora_roots LFI / path traversal defense ──


def test_lora_registry_and_allowed_roots_lfi_defense(
    tmp_path: Path,
    fake_qf_lib: FakeQFLib,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    root = tmp_path / "loras"
    sub = root / "sub"
    sub.mkdir(parents=True)
    sibling_evil = tmp_path / "loras_evil"
    sibling_evil.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    good_file = _write_synthetic_safetensors(
        sub / "good.safetensors",
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw),
            "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [2, 2], raw),
        },
    )
    evil_sibling_file = _write_synthetic_safetensors(
        sibling_evil / "evil.safetensors",
        {"transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw)},
    )
    outside_secret = _write_synthetic_safetensors(
        outside / "secret.safetensors",
        {"transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw)},
    )

    sym_inside_ok = root / "sym_good.safetensors"
    sym_inside_ok.symlink_to(good_file)
    sym_escape_existing = root / "sym_escape_existing.safetensors"
    sym_escape_existing.symlink_to(outside_secret)
    sym_escape_missing = root / "sym_escape_missing.safetensors"
    sym_escape_missing.symlink_to(outside / "does_not_exist.safetensors")

    # Positive resolution: alias, inside root, and symlink staying inside root
    assert (
        lora.resolve_lora_path(
            "style",
            lora_registry={"style": good_file},
            allowed_lora_roots=[root],
            strict_registry=True,
        )
        == os.path.realpath(good_file)
    )
    assert lora.resolve_lora_path(good_file, allowed_lora_roots=[root]) == os.path.realpath(good_file)
    assert lora.resolve_lora_path(sym_inside_ok, allowed_lora_roots=[root]) == os.path.realpath(good_file)

    # All 6 attack vectors fail closed BEFORE os.stat/open (zero existence oracle!)
    traversal_existing = str(root / ".." / "outside" / "secret.safetensors")
    traversal_missing = str(root / ".." / "outside" / "nonexistent.safetensors")
    sibling_missing = str(sibling_evil / "missing.safetensors")

    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    mgr = lora.LoRAManager(
        eng,
        cache_dir=tmp_path / "cache",
        lora_registry={"style": good_file, "bad_alias": outside_secret},
        allowed_lora_roots=[root],
        strict_registry=False,
    )
    stat_targets: list[str] = []
    hdr_reads = {"n": 0}
    real_stat = lora.os.stat
    real_hdr = lora._read_safetensors_header

    def spy_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        stat_targets.append(os.fspath(path))
        return real_stat(path, *args, **kwargs)

    def spy_hdr(path: str) -> dict[str, Any]:
        hdr_reads["n"] += 1
        return real_hdr(path)

    monkeypatch.setattr(lora.os, "stat", spy_stat)
    monkeypatch.setattr(lora, "_read_safetensors_header", spy_hdr)

    try:
        for attack in (
            traversal_existing,
            traversal_missing,
            str(evil_sibling_file),
            sibling_missing,
            str(sym_escape_existing),
            str(sym_escape_missing),
            str(root),
            str(root / "."),
            "bad_alias",
        ):
            stat_targets.clear()
            with pytest.raises(PermissionError):
                mgr.set_loras([attack])
            assert stat_targets == [], f"Existence oracle: os.stat called for rejected vector {attack!r}"
            assert hdr_reads["n"] == 0

        # NUL byte rejected with ValueError
        with pytest.raises(ValueError):
            lora.resolve_lora_path(f"{root}/bad\x00.safetensors", allowed_lora_roots=[root])

        # strict_registry=True rejects unregistered path even inside allowed_lora_roots
        mgr.configure_policy(strict_registry=True)
        stat_targets.clear()
        with pytest.raises(PermissionError):
            mgr.set_loras([str(good_file)])
        assert stat_targets == []
        # Registered alias still allowed
        mgr.set_loras(["style"])
        assert len(fake_qf_lib.loras_history) == 1

        # Inside-root nonexistent file and inside-root directory raise FileNotFoundError (404)
        mgr.configure_policy(strict_registry=False)
        with pytest.raises(FileNotFoundError):
            mgr.set_loras([str(root / "missing_inside.safetensors")])
        with pytest.raises(FileNotFoundError):
            mgr.set_loras([str(sub)])

        # Bare string allowed_lora_roots rejected (prevents char-by-char '/' root bug); empty list denies all
        with pytest.raises(ValueError):
            lora.resolve_lora_path(good_file, allowed_lora_roots=str(root))  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            lora.resolve_lora_path(good_file, allowed_lora_roots=[])
    finally:
        eng.destroy()


# ── Test 40 (Mutation Arm M7): L1 security, stat invalidation & strict-keys mutations ──


def test_mutation_m7_overrides_and_lora_security(
    tmp_path: Path,
    fake_qf_lib: FakeQFLib,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    root = tmp_path / "loras"
    root.mkdir()
    sibling = tmp_path / "loras_evil"
    sibling.mkdir()
    evil_file = _write_synthetic_safetensors(
        sibling / "x.safetensors",
        {"transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw)},
    )
    good_file = _write_synthetic_safetensors(
        root / "good.safetensors",
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw),
            "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [2, 2], raw),
        },
    )

    # M7a: Replacing commonpath with naive str.startswith breaks sibling-prefix LFI rejection
    with pytest.raises(PermissionError):
        lora.resolve_lora_path(evil_file, allowed_lora_roots=[root])

    real_commonpath = os.path.commonpath

    def _naive_startswith_commonpath(paths: list[str]) -> str:
        if len(paths) == 2 and paths[1].startswith(paths[0]):
            return paths[0]
        return real_commonpath(paths)

    monkeypatch.setattr(lora.os.path, "commonpath", _naive_startswith_commonpath)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(PermissionError):
            lora.resolve_lora_path(evil_file, allowed_lora_roots=[root])
    monkeypatch.undo()

    # M7b: Ignoring mtime_ns/size/ino (constant stat fingerprint) breaks disk-modification invalidation
    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    try:
        mgr = lora.LoRAManager(eng, cache_dir=tmp_path / "cache")
        mgr.set_loras([(good_file, 0.8)])
        assert len(fake_qf_lib.loras_history) == 1

        # Rewrite file on disk -> production code invalidates and rebinds
        raw2 = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        _write_synthetic_safetensors(
            good_file,
            {
                "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 4], raw2),
                "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [4, 2], raw2),
            },
        )
        st_cur = os.stat(good_file)
        os.utime(good_file, ns=(st_cur.st_atime_ns, st_cur.st_mtime_ns + 5_000_000))
        mgr.set_loras([(good_file, 0.8)])
        assert len(fake_qf_lib.loras_history) == 2

        # Mutate os.stat so (mtime_ns, size, ino) appear constant -> disk change fails to rebind!
        frozen_st = os.stat(good_file)
        monkeypatch.setattr(lora.os, "stat", lambda path: frozen_st)
        _write_synthetic_safetensors(
            good_file,
            {
                "transformer_blocks.0.attn.to_q.lora_A.weight": ("F32", [2, 2], raw),
                "transformer_blocks.0.attn.to_q.lora_B.weight": ("F32", [2, 2], raw),
            },
        )
        mgr.set_loras([(good_file, 0.8)])
        with pytest.raises(AssertionError):
            assert len(fake_qf_lib.loras_history) == 3
        monkeypatch.undo()
    finally:
        eng.destroy()

    # M7c: Dropping unknown-key rejection in RuntimeOverrides.from_dict turns refusal assertion red
    with pytest.raises(ValueError, match="Unknown"):
        overrides.RuntimeOverrides.from_dict({"unknown_dial": 1})

    orig_from_dict = overrides.RuntimeOverrides.from_dict.__func__

    def _permissive_from_dict(cls: type[overrides.RuntimeOverrides], data: dict[str, Any]) -> overrides.RuntimeOverrides:
        filtered = {k: v for k, v in data.items() if k in overrides._ALLOWED_KEYS}
        return orig_from_dict(cls, filtered)

    monkeypatch.setattr(overrides.RuntimeOverrides, "from_dict", classmethod(_permissive_from_dict))
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(ValueError):
            overrides.RuntimeOverrides.from_dict({"unknown_dial": 1})
    monkeypatch.undo()


# ── Test 41 (U-C03): producer_stream_mode default "torch_current" & cudaStreamLegacy (0x1) ──


def test_producer_stream_mode_default_and_legacy_sentinel(
    tmp_path: Path,
    fake_qf_lib: FakeQFLib,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from quantfunc.models import QuantFuncKrea2Transformer2DModel
    from quantfunc.native.handle import CREATE_CONFIG_ALLOWLIST

    # 1. EngineConfig default is "torch_current" and to_create_dict() stays strictly within 12-key allowlist
    ec_default = config.EngineConfig()
    assert ec_default.producer_stream_mode == "torch_current"
    cd_default = ec_default.to_create_dict()
    assert set(cd_default.keys()).issubset(CREATE_CONFIG_ALLOWLIST)
    assert "producer_stream_mode" not in cd_default
    assert len(CREATE_CONFIG_ALLOWLIST) == 12

    # 2. Explicit "sync" contract preserved; invalid producer_stream_mode rejected
    ec_sync = config.EngineConfig(producer_stream_mode="sync")
    assert ec_sync.producer_stream_mode == "sync"
    for bad_mode in ("", "cuda", "legacy", "ASYNC", None, 123, True):
        with pytest.raises(ValueError, match="producer_stream_mode"):
            config.EngineConfig(producer_stream_mode=bad_mode)  # type: ignore[arg-type]

    # 3. _resolve_producer_stream / _producer_stream_ptr across default stream (0->1), custom stream, CPU, and "sync"
    ckpt = _write_synthetic_safetensors(
        tmp_path / "krea2-stream-test.safetensors",
        {"transformer_blocks.0.attn.to_q.weight": ("F32", [1, 2], struct.pack("<2f", 1.0, 2.0))},
    )
    m_default = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=fake_qf_lib, torch_dtype=torch.float32)
    m_sync = QuantFuncKrea2Transformer2DModel.from_pretrained(
        ckpt, lib=fake_qf_lib, torch_dtype=torch.float32, producer_stream_mode="sync"
    )
    try:
        assert m_default._producer_stream_mode == "torch_current"
        assert m_sync._producer_stream_mode == "sync"

        # CPU tensor always resolves to 0
        cpu_t = torch.zeros(1, 2, dtype=torch.float32)
        assert m_default._resolve_producer_stream(cpu_t) == 0
        assert m_default._producer_stream_ptr(cpu_t) == 0
        assert m_sync._resolve_producer_stream(cpu_t) == 0

        class _FakeCudaTensor:
            is_cuda = True
            device = torch.device("cpu")

        class _FakeStream:
            def __init__(self, ptr: int) -> None:
                self.cuda_stream = ptr

        fake_cuda = _FakeCudaTensor()

        # Default PyTorch CUDA stream (raw == 0) -> cudaStreamLegacy sentinel 1 (0x1)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: _FakeStream(0))
        assert m_default._resolve_producer_stream(fake_cuda) == 1  # type: ignore[arg-type]
        assert m_default._producer_stream_ptr(fake_cuda) == 1  # type: ignore[arg-type]
        # Explicit "sync" mode still returns 0 even for CUDA tensor
        assert m_sync._resolve_producer_stream(fake_cuda) == 0  # type: ignore[arg-type]

        # Non-zero custom CUDA stream passes through verbatim
        custom_ptr = 0x7F12345600
        monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: _FakeStream(custom_ptr))
        assert m_default._resolve_producer_stream(fake_cuda) == custom_ptr  # type: ignore[arg-type]
        assert m_default._producer_stream_ptr(fake_cuda) == custom_ptr  # type: ignore[arg-type]
        assert m_sync._resolve_producer_stream(fake_cuda) == 0  # type: ignore[arg-type]
    finally:
        m_default.close()
        m_sync.close()


# ── Test 42 (U-C06): Depth-1 safetensors JSON-key byte scanner (0 json.loads) ──


def test_safetensors_depth1_key_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantfunc.models.ltx25 import _file_has_both_connectors

    raw = struct.pack("<2f", 1.0, 2.0)

    # 1. Valid wrapped connector keys + unicode/quote escapes, with zero json.loads calls
    good_st = _write_synthetic_safetensors(
        tmp_path / "ltx25_connectors_wrapped.safetensors",
        {
            "model.diffusion_model.video_embeddings_connector.blocks.0.weight": ("F32", [1, 2], raw),
            "model.diffusion_model.audio_embeddings_connector.blocks.0.weight": ("F32", [1, 2], raw),
            "other.weight": ("F32", [1, 2], raw),
        },
        metadata={"format": "pt", "desc": 'escaped \\"quote\\" and } { braces'},
    )

    # 2. Spoofed file: connector strings appear ONLY inside __metadata__ (including escaped quotes & fake JSON)
    #    and inside a tensor attribute value, NEVER as a depth-1 tensor key.
    spoof_hdr = (
        b'{"__metadata__":{'
        b'"fake":"video_embeddings_connector.blocks.0.weight",'
        b'"escaped":"foo \\"audio_embeddings_connector.weight\\" \\\\",'
        b'"injection":"} , \\"video_embeddings_connector.inject\\":{\\"dtype\\":\\"F32\\"}",'
        b'"nested":{"inner":["audio_embeddings_connector.nested",{"video_connector.deep":1}]}'
        b'},'
        b'"normal.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8],'
        b'"note":"audio_embeddings_connector.attr_spoof \\"video_embeddings_connector.attr\\""}}'
    )
    spoof_st = tmp_path / "spoof_metadata.safetensors"
    spoof_st.write_bytes(struct.pack("<Q", len(spoof_hdr)) + spoof_hdr + raw)

    # 3. Escaped depth-1 key (\u005f == '_')
    esc_hdr = (
        b'{"__metadata__":{"ignored":"audio_connector.x"},'
        b'"video_embeddings\\u005fconnector.proj.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}'
    )
    esc_st = tmp_path / "escaped_key.safetensors"
    esc_st.write_bytes(struct.pack("<Q", len(esc_hdr)) + esc_hdr + raw)

    # 4. Empty JSON object header
    empty_st = tmp_path / "empty_obj.safetensors"
    empty_st.write_bytes(struct.pack("<Q", 2) + b"{}")

    # Forbid json.loads across all scanner calls (must be 0 json.loads!)
    def _forbid_json_loads(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("json.loads must NOT be called by safetensors_header_has_key_markers")

    monkeypatch.setattr(json, "loads", _forbid_json_loads)
    monkeypatch.setattr(checkpoint.json, "loads", _forbid_json_loads)

    # Substring match hits wrapped connector keys
    res_sub = checkpoint.safetensors_header_has_key_markers(
        good_st,
        ["video_embeddings_connector.", "audio_embeddings_connector.", "missing_marker."],
        match="substring",
    )
    assert res_sub == {
        "video_embeddings_connector.": True,
        "audio_embeddings_connector.": True,
        "missing_marker.": False,
    }

    # Prefix match distinguishes true prefix from mid-key substring
    res_pfx = checkpoint.safetensors_header_has_key_markers(
        good_st,
        ["model.diffusion_model.", "video_embeddings_connector.", "other."],
        match="prefix",
    )
    assert res_pfx == {
        "model.diffusion_model.": True,
        "video_embeddings_connector.": False,
        "other.": True,
    }

    # Spoofed __metadata__ and tensor attribute strings MUST all return False
    res_spoof = checkpoint.safetensors_header_has_key_markers(
        spoof_st,
        ["video_embeddings_connector.", "audio_embeddings_connector.", "video_connector.", "normal."],
        match="substring",
    )
    assert res_spoof == {
        "video_embeddings_connector.": False,
        "audio_embeddings_connector.": False,
        "video_connector.": False,
        "normal.": True,
    }

    # Escaped depth-1 key decodes without json.loads while ignoring __metadata__
    res_esc = checkpoint.safetensors_header_has_key_markers(
        esc_st,
        ["video_embeddings_connector.", "audio_connector."],
        match="substring",
    )
    assert res_esc == {"video_embeddings_connector.": True, "audio_connector.": False}

    # Empty object & empty markers
    assert checkpoint.safetensors_header_has_key_markers(empty_st, ["a."], match="prefix") == {"a.": False}
    assert checkpoint.safetensors_header_has_key_markers(good_st, [], match="substring") == {}

    # ltx25._file_has_both_connectors integration (0 json.loads)
    assert _file_has_both_connectors(good_st) == (True, True, True)
    assert _file_has_both_connectors(spoof_st) == (False, False, False)
    assert _file_has_both_connectors(esc_st) == (False, True, False)

    # Negative / malformed / truncated / oversized header refusals
    trunc_len = tmp_path / "trunc_len.safetensors"
    trunc_len.write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError, match="Truncated"):
        checkpoint.safetensors_header_has_key_markers(trunc_len, ["a."])
    assert _file_has_both_connectors(trunc_len) == (False, False, False)

    zero_hdr = tmp_path / "zero_hdr.safetensors"
    zero_hdr.write_bytes(struct.pack("<Q", 0))
    with pytest.raises(ValueError, match="Implausible"):
        checkpoint.safetensors_header_has_key_markers(zero_hdr, ["a."])

    oversized = tmp_path / "oversized.safetensors"
    oversized.write_bytes(struct.pack("<Q", 1024) + b"{}")
    with pytest.raises(ValueError, match="Implausible"):
        checkpoint.safetensors_header_has_key_markers(oversized, ["a."], max_header_bytes=64)

    trunc_payload = tmp_path / "trunc_payload.safetensors"
    trunc_payload.write_bytes(struct.pack("<Q", 100) + b'{"a":1}')
    with pytest.raises(ValueError, match="Truncated"):
        checkpoint.safetensors_header_has_key_markers(trunc_payload, ["a"])

    non_obj = tmp_path / "non_obj.safetensors"
    non_obj.write_bytes(struct.pack("<Q", 7) + b"[1,2,3]")
    with pytest.raises(ValueError, match="not a JSON object"):
        checkpoint.safetensors_header_has_key_markers(non_obj, ["a"])

    with pytest.raises(ValueError, match="Unsupported match mode"):
        checkpoint.safetensors_header_has_key_markers(good_st, ["a"], match="regex")  # type: ignore[arg-type]

    # 5. Whitespace, simple escapes, UTF-16 surrogates, primitives, early-exit & all malformed forms
    ws_esc_hdr = (
        b' \t\r\n{\n'
        b'  "prim_str" : "hello \\"world\\"" ,\n'
        b'  "prim_num" : -12.5e3 ,\n'
        b'  "prim_bool" : true ,\n'
        b'  "prim_null" : null ,\n'
        b'  "esc_key_\\"\\\\\\/\\b\\f\\n\\r\\t_\\uD83D\\uDE00_\\uD83D\\u0041" : [ 1 , { "x" : 2 } ]\n'
        b'} \n'
    )
    ws_esc_st = tmp_path / "ws_esc.safetensors"
    ws_esc_st.write_bytes(struct.pack("<Q", len(ws_esc_hdr)) + ws_esc_hdr + raw)
    res_ws = checkpoint.safetensors_header_has_key_markers(
        ws_esc_st,
        ['esc_key_"\\/\b\f\n\r\t_\U0001f600_\ud83dA'],
        match="prefix",
    )
    assert res_ws == {'esc_key_"\\/\b\f\n\r\t_\U0001f600_\ud83dA': True}

    # Early exit when all pending markers match before trailing keys
    early_hdr = (
        b'{"video_connector.w":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},'
        b'"bad\\q":{"dtype":"F32"}}'
    )
    early_st = tmp_path / "early_exit.safetensors"
    early_st.write_bytes(struct.pack("<Q", len(early_hdr)) + early_hdr + raw)
    assert checkpoint.safetensors_header_has_key_markers(early_st, ["video_connector."]) == {
        "video_connector.": True
    }

    # Direct _unescape_json_string trailing backslash & strict RFC-8259 \uXXXX checks
    with pytest.raises(ValueError, match="Unterminated escape"):
        checkpoint._unescape_json_string(b"abc\\")

    for bad_u in (b"\\u+041", b"\\u 041", b"\\u 41", b"\\u0_41", b"\\uZZZZ", b"\\u12"):
        with pytest.raises(ValueError, match=r"\\u escape"):
            checkpoint._unescape_json_string(bad_u)
        # Parity with stdlib json.loads: RFC-8259 forbids sign/space/underscore in \uXXXX
        monkeypatch.undo()
        with pytest.raises(ValueError):
            json.loads('"' + bad_u.decode("ascii") + '"')
        monkeypatch.setattr(json, "loads", _forbid_json_loads)
        monkeypatch.setattr(checkpoint.json, "loads", _forbid_json_loads)

    for bad_surr in (b"\\uD83D\\u+C00", b"\\uD83D\\u 000", b"\\uD83D\\u0_00", b"\\uD83D\\uZZZZ"):
        with pytest.raises(ValueError, match="surrogate"):
            checkpoint._unescape_json_string(bad_surr)

    malformed_cases: list[tuple[bytes, str]] = [
        (b'{"bad\\q":1}', "Invalid escape"),
        (b'{"unterminated', "Unterminated JSON string"),
        (b'{"a":', "Unexpected end"),
        (b'{"a":[1,2', "Unterminated JSON container"),
        (b'{"a":[1,2}}', "Mismatched JSON container delimiter"),
        (b'{"a":{]}', "Mismatched JSON container delimiter"),
        (b'{"a":,}', "Malformed JSON primitive"),
        (b'{123:1}', "Malformed safetensors header JSON key"),
        (b'{"\xff\xfe":1}', "Malformed UTF-8"),
        (b'{"a" 1}', "missing ':'"),
        (b'{"a":1 ', "Truncated safetensors header JSON"),
        (b'{"a":1,', "Truncated safetensors header JSON"),
        (b'{"a":1,}', "Malformed safetensors header JSON key"),
        (b'{"a":1 "b":2}', "expected ',' or '}'"),
    ]
    for idx, (bad_hdr, pat) in enumerate(malformed_cases):
        bad_p = tmp_path / f"malformed_{idx}.safetensors"
        bad_p.write_bytes(struct.pack("<Q", len(bad_hdr)) + bad_hdr)
        with pytest.raises(ValueError, match=pat):
            checkpoint.safetensors_header_has_key_markers(bad_p, ["never_match"])


def test_session_keepalive_reset_and_factory_ordering(fake_qf_lib: FakeQFLib) -> None:
    from quantfunc.native.errors import OutOfMemoryError

    eng = QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "krea2"})
    try:
        events: list[str] = []
        tracker = session.AutoSessionTracker(
            eng,
            busy_retry_delay_s=0.0,
            on_session_reset=lambda: events.append("reset"),
        )

        def _open(num_steps: int, opts: dict[str, Any]) -> Any:
            events.append("open")
            return eng.session_begin(256, 256, num_steps, (1, 64, 128), QF_DTYPE_BF16, opts)

        # Round 1: open then sig_id_factory runs AFTER open
        _, _, k1 = tracker.resolve_step_and_session(
            0.9,
            ("g", 1),
            16,
            {},
            _open,
            sig_id_factory=lambda: (events.append("fp1"), ("ctx", 1))[1],
        )
        assert k1 == 1
        assert events == ["reset", "open", "fp1"]

        # Step 1 same session: no reset/open, computes fp1 -> same key 1
        events.clear()
        _, _, k1_b = tracker.resolve_step_and_session(
            0.5,
            ("g", 1),
            16,
            {},
            _open,
            sig_id_factory=lambda: (events.append("fp1_b"), ("ctx", 1))[1],
        )
        assert k1_b == 1
        assert events == ["fp1_b"]

        # Round 2 (sigma increases -> need_open=True): reset MUST fire BEFORE open & sig_id_factory!
        events.clear()
        _, _, k2 = tracker.resolve_step_and_session(
            0.95,
            ("g", 1),
            16,
            {},
            _open,
            sig_id_factory=lambda: (events.append("fp2"), ("ctx", 2))[1],
        )
        assert k2 == 1
        assert events == ["reset", "open", "fp2"]

        # Failing open_fn never invokes sig_id_factory and already triggered reset
        events.clear()
        fp_called = {"n": 0}

        def _boom_open(num_steps: int, opts: dict[str, Any]) -> Any:
            raise OutOfMemoryError("oom open")

        with pytest.raises(OutOfMemoryError):
            tracker.resolve_step_and_session(
                0.99,
                ("g", 1),
                16,
                {},
                _boom_open,
                sig_id_factory=lambda: fp_called.__setitem__("n", fp_called["n"] + 1) or "x",
            )
        assert events == ["reset"]
        assert fp_called["n"] == 0

        # SamplingSession on_session_end fires on __enter__, normal __exit__, error __exit__, and __enter__ failure
        end_calls = {"n": 0}
        with session.SamplingSession(
            eng,
            tracker=tracker,
            width=256,
            height=256,
            num_steps=3,
            max_context_dims=(1, 64, 128),
            cond_dtype=QF_DTYPE_BF16,
            options={},
            on_session_end=lambda: end_calls.__setitem__("n", end_calls["n"] + 1),
        ):
            assert end_calls["n"] >= 1
        after_norm = end_calls["n"]
        assert after_norm >= 2

        with pytest.raises(RuntimeError, match="boom"):
            with session.SamplingSession(
                eng,
                tracker=tracker,
                width=256,
                height=256,
                num_steps=3,
                max_context_dims=(1, 64, 128),
                cond_dtype=QF_DTYPE_BF16,
                options={},
                on_session_end=lambda: end_calls.__setitem__("n", end_calls["n"] + 1),
            ):
                raise RuntimeError("boom")
        assert end_calls["n"] > after_norm

        # Schedule-only SamplingSession (width=None -> skips session_begin, still invokes on_session_end)
        before_sched_only = end_calls["n"]
        with session.SamplingSession(
            eng,
            tracker=tracker,
            sigmas=(1.0, 0.5, 0.0),
            on_session_end=lambda: end_calls.__setitem__("n", end_calls["n"] + 1),
        ) as sess_only:
            assert sess_only.session_handle is None
        assert end_calls["n"] >= before_sched_only + 2

        orig_begin = eng.session_begin

        def _boom_begin(*args: Any, **kwargs: Any) -> Any:
            raise OutOfMemoryError("begin boom")

        eng.session_begin = _boom_begin  # type: ignore[method-assign]
        before_enter_fail = end_calls["n"]
        with pytest.raises(OutOfMemoryError, match="begin boom"):
            with session.SamplingSession(
                eng,
                tracker=tracker,
                width=256,
                height=256,
                num_steps=3,
                max_context_dims=(1, 64, 128),
                cond_dtype=QF_DTYPE_BF16,
                options={},
                on_session_end=lambda: end_calls.__setitem__("n", end_calls["n"] + 1),
            ):
                pass
        eng.session_begin = orig_begin  # type: ignore[method-assign]
        assert end_calls["n"] >= before_enter_fail + 2
    finally:
        eng.destroy()
