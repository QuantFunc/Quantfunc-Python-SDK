"""L1 unit & M2 mutation tests: Sol, Config/Auth, LoRA, 3-tier Session, Registry & Presets (pure CPU)."""

from __future__ import annotations

import json
import os
import stat
import struct
import warnings
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeQFLib

from quantfunc.core import checkpoint, config, lora, registry, session, sol
from quantfunc.native.capi import QF_DTYPE_BF16
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
