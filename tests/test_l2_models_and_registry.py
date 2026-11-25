"""L2 model adapter, registry/Auto factory, accelerate offload, and M3 mutation tests (Tests 19-25).

Pure-CPU only (CUDA_VISIBLE_DEVICES=""); exercises FakeQFLib end-to-end across all 4 models.
"""

from __future__ import annotations

import ctypes
import gc
import inspect
import json
import logging
import struct
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import FakeQFLib

from quantfunc.core.config import WIRE_QUALITY_RATIO_KEY
from quantfunc.core.registry import register_model, unregister_model
from quantfunc.core.session import StagedSchedule
from quantfunc.models import (
    AutoQuantFuncTransformer,
    QuantFuncBaseTransformer,
    QuantFuncKrea2Transformer2DModel,
    QuantFuncLTX25Transformer3DModel,
    QuantFuncMiniMaxH3Transformer3DModel,
    QuantFuncQwenImage21Transformer2DModel,
    Transformer2DModelOutput,
    restore_frozen_audio,
)
from quantfunc.native.capi import QF_MAX_STEP_REF_AREA, QF_MAX_STEP_REFS, QF_UINT64_MAX
from quantfunc.native.errors import InvalidArgError, OutOfMemoryError


def _write_safetensors(path: Path, tensors: dict[str, tuple[list[int], str, bytes]], metadata: dict[str, str] | None = None) -> Path:
    """Write a minimal valid safetensors file using stdlib only."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    payload = bytearray()
    offset = 0
    for name, (shape, dtype, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        fh.write(bytes(payload))
    return path


def _make_dummy_lora(path: Path) -> Path:
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    return _write_safetensors(
        path,
        {
            "transformer_blocks.0.attn.to_q.lora_A.weight": ([2, 2], "F32", raw),
            "transformer_blocks.0.attn.to_q.lora_B.weight": ([2, 2], "F32", raw),
        },
    )


def _make_dummy_checkpoint(path: Path, *, tensor_keys: list[str] | None = None, metadata: dict[str, str] | None = None) -> Path:
    keys = tensor_keys or ["transformer_blocks.0.attn.to_q.weight"]
    raw = struct.pack("<2f", 1.0, 2.0)
    return _write_safetensors(path, {k: ([1, 2], "F32", raw) for k in keys}, metadata=metadata)


# ── Test 19: QuantFuncBaseTransformer lifecycle, meta dtype-only, cross-GPU & Auto ──


def test_base_transformer_lifecycle_meta_dtype_and_cross_gpu(tmp_path: Path) -> None:
    lib = FakeQFLib()
    ckpt = _make_dummy_checkpoint(tmp_path / "krea2-quantfunc-int4.safetensors")
    lora_file = _make_dummy_lora(tmp_path / "style-lora.safetensors")

    m = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    m_single = QuantFuncKrea2Transformer2DModel.from_single_file(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        # 0-byte parameter anchor
        params = list(m.parameters())
        assert len(params) == 1
        assert params[0].numel() == 0
        assert m.dtype == torch.float32
        assert m_single.config.in_channels == m.config.in_channels

        # set_default_steps propagation & validation
        m.set_default_steps(16)
        assert m._session_tracker.default_steps == 16
        with pytest.raises(ValueError):
            m.set_default_steps(0)
        with pytest.raises(ValueError):
            m.set_default_steps(-4)
        with pytest.raises(ValueError):
            m.set_default_steps(True)  # type: ignore[arg-type]

        # estimate_vram & resident_vram_bytes
        est = m.estimate_vram()
        assert est.resident_weight_bytes == lib.estimate_resident_bytes
        assert m.resident_vram_bytes() == lib.initial_resident_bytes

        # load_lora_adapter must fail with clear NotImplementedError pointing to load_lora
        with pytest.raises(NotImplementedError, match="load_lora"):
            m.load_lora_adapter("some/repo")

        # Load a LoRA and verify step uses set_default_steps(16)
        m.load_lora(lora_file, scale=0.75)
        assert len(lib.loras_history) == 1
        assert lib.loras_history[-1][0]["scale"] == pytest.approx(0.75)

        x = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        ctx = torch.randn(1, 4, 2560, dtype=torch.float32)
        out = m(x, encoder_hidden_states=ctx, timestep=0.5)
        assert isinstance(out, Transformer2DModelOutput)
        assert lib.sessions[-1]["num_steps"] == 16
        assert not lib.sessions[-1]["ended"]

        # to("cpu") closes active session and calls unload_vram(QF_UINT64_MAX)
        m.to("cpu")
        assert lib.sessions[-1]["ended"] is True
        assert lib.unloads[-1] == QF_UINT64_MAX

        # to("cuda:1") destroys old handle and recreates on device_id=1 while re-applying active LoRAs
        old_hid = lib.created_configs[0]["handle_id"]
        m.to("cuda:1")
        assert old_hid in lib.destroyed
        assert lib.created_configs[-1]["device_id"] == 1
        assert len(lib.loras_history) >= 2
        assert lib.loras_history[-1][0]["scale"] == pytest.approx(0.75)
        assert m.device == torch.device("cuda:1")

        # Meta anchor + _hf_hook(execution_device="cuda:0"): dtype-only .to(torch.bfloat16) keeps meta
        class _FakeHook:
            execution_device = "cuda:0"

        m.to("meta")
        m._hf_hook = _FakeHook()
        assert m._qf_device_anchor.device.type == "meta"
        assert m.device == torch.device("cuda:0")

        m.to(torch.bfloat16)
        assert m._qf_device_anchor.device.type == "meta"
        assert m.dtype == torch.bfloat16
        assert m.device == torch.device("cuda:0")

        # AutoQuantFuncTransformer resolves built-in model + 5th user-registered custom model
        auto_krea = AutoQuantFuncTransformer.from_single_file(ckpt, lib=lib)
        assert isinstance(auto_krea, QuantFuncKrea2Transformer2DModel)
        auto_krea.close()

        custom_cfg_path = tmp_path / "custom_v5_config.json"
        custom_cfg_path.write_text(json.dumps({"in_channels": 32, "_class_name": "CustomV5Transformer"}), encoding="utf-8")
        custom_ckpt = _make_dummy_checkpoint(tmp_path / "custom_v5_model.safetensors")

        @register_model(
            arch="custom_v5",
            transformer_classes=("CustomV5Transformer",),
            file_hints=("*custom_v5*",),
            default_config_path=custom_cfg_path,
            engine_arch="auto",
        )
        class _CustomV5Transformer(QuantFuncBaseTransformer):
            model_arch = "custom_v5"

        try:
            custom_inst = AutoQuantFuncTransformer.from_pretrained(custom_ckpt, lib=lib)
            assert isinstance(custom_inst, _CustomV5Transformer)
            assert custom_inst.config.in_channels == 32
            assert lib.created_configs[-1]["model_arch"] == "auto"
            custom_inst.close()
        finally:
            unregister_model("custom_v5")
    finally:
        m.close()
        m_single.close()


def test_from_pretrained_default_auto_install_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sig = inspect.signature(QuantFuncBaseTransformer.from_pretrained)
    assert sig.parameters["auto_install"].default is True

    import quantfunc.models.base as base_mod

    lib = FakeQFLib()
    calls: list[dict[str, Any]] = []

    def _spy_load_lib(*, device_id: int = 0, so_path: str | None = None, auto_install: bool = True) -> FakeQFLib:
        calls.append({"device_id": device_id, "so_path": so_path, "auto_install": auto_install})
        return lib

    monkeypatch.setattr(base_mod, "load_lib", _spy_load_lib)
    ckpt = _make_dummy_checkpoint(tmp_path / "krea2-quantfunc-int4.safetensors")

    m_default = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, torch_dtype=torch.float32)
    m_single = QuantFuncKrea2Transformer2DModel.from_single_file(ckpt, torch_dtype=torch.float32)
    m_opt_out = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, auto_install=False, torch_dtype=torch.float32)
    try:
        assert [c["auto_install"] for c in calls] == [True, True, False]
    finally:
        m_default.close()
        m_single.close()
        m_opt_out.close()


def test_timestep_normalization_flowmatch_cursor_and_minimax_scale(tmp_path: Path) -> None:
    # 1. FlowMatchEulerDiscreteScheduler 4-step [1000.0, 667.0, 334.0, 1.0] -> [1.0, 0.667, 0.334, 0.001]
    #    with step_index = [0, 1, 2, 3] in BOTH Tier 1 and Tier 2, plus 2-step & single-step!
    lib = FakeQFLib()
    ckpt = _make_dummy_checkpoint(tmp_path / "krea2-quantfunc-int4.safetensors")
    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        x = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        ctx = torch.randn(1, 4, 2560, dtype=torch.float32)

        # Tier 1: FlowMatch 4-step schedule
        mk.set_staged_schedule(
            StagedSchedule(
                sigmas=(1.0, 0.667, 0.334, 0.001, 0.0),
                num_steps=4,
                timestep_scale=1000.0,
                use_cursor_indexing=False,
            )
        )
        for t_val in (1000.0, 667.0, 334.0, 1.0):
            mk(x, encoder_hidden_states=ctx, timestep=t_val)
        assert len(lib.sessions) == 1
        assert [s["sigma"] for s in lib.steps] == pytest.approx([1.0, 0.667, 0.334, 0.001])
        assert [s["step_index"] for s in lib.steps] == [0, 1, 2, 3]

        # Tier 1 single-step timestep=1.0 with sched_scale=1000.0 -> 0.001
        mk.end_session_if_open()
        lib.steps.clear()
        mk.set_staged_schedule(
            StagedSchedule(sigmas=(0.001, 0.0), num_steps=1, timestep_scale=1000.0)
        )
        mk(x, encoder_hidden_states=ctx, timestep=1.0)
        assert lib.steps[-1]["sigma"] == pytest.approx(0.001)
        assert lib.steps[-1]["step_index"] == 0

        # Tier 2 (no staged schedule): 4-step [1000.0, 667.0, 334.0, 1.0] & 2-step [1000.0, 1.0]
        mk.set_staged_schedule(None)
        mk.end_session_if_open()
        lib.sessions.clear()
        lib.steps.clear()
        for t_val in (1000.0, 667.0, 334.0, 1.0):
            mk(x, encoder_hidden_states=ctx, timestep=t_val)
        assert len(lib.sessions) == 1
        assert [s["sigma"] for s in lib.steps] == pytest.approx([1.0, 0.667, 0.334, 0.001])
        assert [s["step_index"] for s in lib.steps] == [0, 1, 2, 3]

        mk.end_session_if_open()
        lib.sessions.clear()
        lib.steps.clear()
        for t_val in (1000.0, 1.0):
            mk(x, encoder_hidden_states=ctx, timestep=t_val)
        assert len(lib.sessions) == 1
        assert [s["sigma"] for s in lib.steps] == pytest.approx([1.0, 0.001])
        assert [s["step_index"] for s in lib.steps] == [0, 1]

        # Tier 1 set_hook_step_cursor delegation for DDIM (empty sigmas) & Heun (EDM sigmas)
        mk.end_session_if_open()
        lib.sessions.clear()
        lib.steps.clear()
        mk.set_staged_schedule(
            StagedSchedule(
                sigmas=(25.146, 4.496, 4.496, 1.214, 1.214, 0.029, 0.029, 0.0),
                num_steps=7,
                timestep_scale=1000.0,
                use_cursor_indexing=True,
            )
        )
        for idx, t_val in enumerate((999.0, 666.0, 666.0, 333.0, 333.0, 0.0, 0.0)):
            mk.set_hook_step_cursor(idx)
            mk(x, encoder_hidden_states=ctx, timestep=t_val)
        assert len(lib.sessions) == 1
        assert [s["step_index"] for s in lib.steps] == [0, 1, 2, 3, 4, 5, 6]

        # Tier-3 SamplingSession exit resets scaled-timestep tracking for subsequent [0, 1] calls
        mk.set_staged_schedule(None)
        mk.end_session_if_open()
        lib.steps.clear()
        with mk.sampling_session(width=256, height=256, num_steps=2, max_context_dims=(1, 64, 2560)):
            mk(x, encoder_hidden_states=ctx, timestep=1000.0)
            mk(x, encoder_hidden_states=ctx, timestep=1.0)
        assert [s["sigma"] for s in lib.steps] == pytest.approx([1.0, 0.001])
        # After SamplingSession closes, fresh [0, 1] scale call with timestep=1.0 MUST stay 1.0
        lib.steps.clear()
        mk(x, encoder_hidden_states=ctx, timestep=1.0)
        assert lib.steps[-1]["sigma"] == pytest.approx(1.0)

        # Negative / non-finite sigma still rejected
        for bad_t in (-0.1, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                mk(x, encoder_hidden_states=ctx, timestep=bad_t)
    finally:
        mk.close()

    # 2. MiniMax-H3 [0, 1] timestep scale: t=1.0 MUST stay 1.0 (never divided to 0.001)!
    lib_h3 = FakeQFLib(echo_scale=1.0)
    ckpt_h3 = _make_dummy_checkpoint(tmp_path / "minimax-h3-scale.safetensors")
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h3, lib=lib_h3, torch_dtype=torch.float32)
    try:
        vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.float32)
        ch = torch.randn(1, 4, 5120, dtype=torch.float32)

        # Tier 2: data_direction=True -> t=1.0 stays 1.0 so engine sigma = 1.0 - 1.0 = 0.0
        mh(vh, encoder_hidden_states=ch, timestep=1.0)
        assert lib_h3.steps[-1]["sigma"] == pytest.approx(0.0)
        mh.end_session_if_open()

        # data_direction=False + StagedSchedule(timestep_scale=1.0): [1.0, 0.75, 0.5, 0.0] stays exact
        mh.set_staged_schedule(
            StagedSchedule(sigmas=(1.0, 0.75, 0.5, 0.0), num_steps=4, timestep_scale=1.0)
        )
        lib_h3.steps.clear()
        for tv in (1.0, 0.75, 0.5, 0.0):
            mh(vh, encoder_hidden_states=ch, timestep=tv, data_direction=False)
        assert [s["sigma"] for s in lib_h3.steps] == pytest.approx([1.0, 0.75, 0.5, 0.0])
        assert [s["step_index"] for s in lib_h3.steps] == [0, 1, 2, 3]
    finally:
        mh.close()


# ── Test 20: Real accelerate SequentialHook + AlignDevicesHook + meta .to() ──


def test_accelerate_sequential_offload_and_meta_to(tmp_path: Path) -> None:
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    SequentialHook = accelerate_hooks.SequentialHook
    AlignDevicesHook = accelerate_hooks.AlignDevicesHook
    add_hook_to_module = accelerate_hooks.add_hook_to_module

    lib = FakeQFLib()
    ckpt = _make_dummy_checkpoint(tmp_path / "krea2-quantfunc-int4.safetensors")
    m = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.bfloat16)
    try:
        seq_hook = SequentialHook(AlignDevicesHook(execution_device="cuda:0", offload=True))
        add_hook_to_module(m, seq_hook)

        # Anchor was moved to meta by AlignDevicesHook(offload=True), yet m.device resolves cuda:0
        assert m._qf_device_anchor.device.type == "meta"
        assert m.device == torch.device("cuda:0")

        # Pure dtype cast on meta keeps anchor on meta and avoids NotImplementedError
        m.to(torch.float16)
        assert m._qf_device_anchor.device.type == "meta"
        assert m.dtype == torch.float16
        assert m.device == torch.device("cuda:0")

        # Cycle-guard check on recursive hook traversal
        class _CyclicHook:
            def __init__(self) -> None:
                self.hooks: list[Any] = [self]

        orig_hook = m._hf_hook
        m._hf_hook = _CyclicHook()
        assert m.device.type != "meta"
        m._hf_hook = orig_hook

        # Transition from meta -> cpu -> cuda:0 without raising "Cannot copy out of meta tensor"
        m.to("cpu")
        assert m._qf_device_anchor.device.type == "cpu"
        assert m.device == torch.device("cpu")

        m.to("cuda:0")
        assert m.device == torch.device("cuda:0")
    finally:
        m.close()


# ── Test 21: Krea-2 4D RoPE, middle-padded mask compression & dynamic config ──


def test_krea2_4d_rope_middle_pad_mask_and_dynamic_config(tmp_path: Path) -> None:
    lib = FakeQFLib()
    captured_ctx_values: list[torch.Tensor] = []
    orig_step = lib.qf_xfm_step._fn

    def _spy_step(session: Any, params_ptr: Any) -> int:
        params = params_ptr._obj if hasattr(params_ptr, "_obj") else params_ptr.contents
        c_dims = [int(params.context_dims[i]) for i in range(3)]
        n_elems = c_dims[0] * c_dims[1] * c_dims[2]
        raw = ctypes.string_at(int(params.context), n_elems * 4)
        vals = torch.tensor(struct.unpack(f"<{n_elems}f", raw), dtype=torch.float32).reshape(*c_dims)
        captured_ctx_values.append(vals)
        return orig_step(session, params_ptr)

    lib.qf_xfm_step._fn = _spy_step

    ckpt = _make_dummy_checkpoint(tmp_path / "krea2-quantfunc-int4.safetensors")
    m = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        # 1. 3D [1, N=6, 64] + position_ids (grid 2x3, permuted) -> FakeQFLib MUST see 4D [1, 64, 2, 3]
        x3d = torch.arange(1 * 6 * 64, dtype=torch.float32).reshape(1, 6, 64)
        pos_ids = torch.tensor(
            [[0, 0, 0], [0, 0, 2], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2]],
            dtype=torch.long,
        )
        # 4D multi-tap context [1, S=5, L=12, D=2560] with middle-padded mask [1, 1, 0, 0, 1]
        ctx_4d = torch.zeros(1, 5, 12, 2560, dtype=torch.float32)
        for tok_idx in range(5):
            ctx_4d[:, tok_idx, :, :] = float(tok_idx + 10)
        mask = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.long)

        out_3d = m(
            x3d,
            encoder_hidden_states=ctx_4d,
            encoder_attention_mask=mask,
            position_ids=pos_ids,
            timestep=750.0,
        )
        assert isinstance(out_3d, Transformer2DModelOutput)
        assert tuple(out_3d.sample.shape) == (1, 6, 64)
        assert torch.allclose(out_3d.sample, x3d)

        last_step = lib.steps[-1]
        assert last_step["dims"] == [1, 64, 2, 3]  # Strictly 4D, never 3D
        assert last_step["context_dims"] == [1, 3, 12 * 2560]  # Compressed S_valid == 3
        assert last_step["sigma"] == pytest.approx(0.75)  # Normalized from 750.0
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.80)

        # Verify exact preserved tokens [0, 1, 4] (values 10.0, 11.0, 14.0)
        sent_ctx = captured_ctx_values[-1]
        assert tuple(sent_ctx.shape) == (1, 3, 12 * 2560)
        assert sent_ctx[0, 0, 0].item() == pytest.approx(10.0)
        assert sent_ctx[0, 1, 0].item() == pytest.approx(11.0)
        assert sent_ctx[0, 2, 0].item() == pytest.approx(14.0)

        # 2. 4D unpatched [1, 16, 4, 6] passthrough + quality_enhance=True -> 1.0
        m.set_quality_enhance(True)
        x4d_unpatched = torch.randn(1, 16, 4, 6, dtype=torch.float32)
        out_unpatched = m(x4d_unpatched, encoder_hidden_states=ctx_4d[:, :3], timestep=0.5, return_dict=False)
        assert isinstance(out_unpatched, tuple) and tuple(out_unpatched[0].shape) == (1, 16, 4, 6)
        assert lib.steps[-1]["dims"] == [1, 16, 4, 6]
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)

        # 3. Negative masks: all-zero mask & unequal batch valid counts fail closed
        with pytest.raises(ValueError, match="zero valid"):
            m(x4d_unpatched, encoder_hidden_states=ctx_4d[:, :3], encoder_attention_mask=torch.zeros(1, 3), timestep=0.5)

        x_b2 = torch.randn(2, 16, 4, 4, dtype=torch.float32)
        ctx_b2 = torch.randn(2, 3, 12, 2560, dtype=torch.float32)
        bad_b2_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="unequal valid"):
            m(x_b2, encoder_hidden_states=ctx_b2, encoder_attention_mask=bad_b2_mask, timestep=0.5)

        # 4. Reject audio/refs/ControlNet cleanly
        with pytest.raises(ValueError, match="audio_hidden_states"):
            m(x4d_unpatched, encoder_hidden_states=ctx_4d[:, :3], timestep=0.5, audio_hidden_states=torch.zeros(1, 2, 8))
        with pytest.raises(ValueError, match="ref_latents"):
            m(x4d_unpatched, encoder_hidden_states=ctx_4d[:, :3], timestep=0.5, ref_latents=[x4d_unpatched])
        with pytest.raises(ValueError, match="controlnet_block_samples"):
            m(x4d_unpatched, encoder_hidden_states=ctx_4d[:, :3], timestep=0.5, controlnet_block_samples=[x4d_unpatched])
    finally:
        m.close()

    # 5. Custom non-standard config geometry dynamically followed (zero hardcoded geometry)
    custom_cfg = {
        "in_channels": 32,
        "patch_size": 4,
        "num_text_layers": 4,
        "text_hidden_dim": 128,
    }
    m_dyn = QuantFuncKrea2Transformer2DModel.from_pretrained(
        ckpt, lib=lib, config=custom_cfg, torch_dtype=torch.float32
    )
    try:
        # unpatched channels = 32 // (4*4) = 2; 4D [1, 2, 8, 8]
        x_dyn = torch.randn(1, 2, 8, 8, dtype=torch.float32)
        ctx_dyn = torch.randn(1, 5, 4, 128, dtype=torch.float32)
        out_dyn = m_dyn(x_dyn, encoder_hidden_states=ctx_dyn, timestep=0.4)
        assert tuple(out_dyn.sample.shape) == (1, 2, 8, 8)
        assert lib.steps[-1]["dims"] == [1, 2, 8, 8]
        assert lib.steps[-1]["context_dims"] == [1, 5, 512]
    finally:
        m_dyn.close()


# ── Test 22: Qwen-Image-2.1 T2I + step_refs + dual QualityLaw + split-CFG reopen ──


def test_qwenimage21_refs_quality_law_and_split_cfg_reopen(tmp_path: Path) -> None:
    lib = FakeQFLib()
    ckpt = _make_dummy_checkpoint(tmp_path / "qwen-image-2.1-quantfunc-int4.safetensors")
    m = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        x = torch.randn(1, 64, 4, 4, dtype=torch.float32)
        ctx_pos = torch.ones(1, 8, 4096, dtype=torch.float32)
        ctx_neg = torch.full((1, 8, 4096), 2.0, dtype=torch.float32)

        # 1. Single-pass T2I (Tier 2) emits 0.85
        m(x, encoder_hidden_states=ctx_pos, timestep=1.0)
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.85)
        assert lib.steps[-1]["kind"] == "step"
        sess_1_id = lib.sessions[-1]["session_id"]

        # 2. Tier-2 sequential split-CFG: second branch at same sigma closes 0.85 session & reopens at 1.0!
        m(x, encoder_hidden_states=ctx_neg, timestep=1.0)
        assert len(lib.sessions) == 2
        assert lib.sessions[0]["session_id"] == sess_1_id
        assert lib.sessions[0]["ended"] is True
        assert lib.sessions[1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)

        # Next denoising step stays on the 1.0 session without reopening
        m(x, encoder_hidden_states=ctx_pos, timestep=0.5)
        assert len(lib.sessions) == 2
        assert lib.sessions[1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        m.end_session_if_open()

        # 3. Tier-1 StagedSchedule(do_cfg=True) emits 1.0 immediately on first session_begin
        m.set_staged_schedule(StagedSchedule(sigmas=(1.0, 0.5, 0.0), num_steps=2, guidance_scale=4.0, do_cfg=True))
        m(x, encoder_hidden_states=ctx_pos, timestep=1.0)
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        m.set_staged_schedule(None)
        m.end_session_if_open()

        # 4. batch > 1 emits 1.0 even with quality_enhance=False
        x_b2 = torch.randn(2, 64, 4, 4, dtype=torch.float32)
        ctx_b2 = torch.randn(2, 8, 4096, dtype=torch.float32)
        m(x_b2, encoder_hidden_states=ctx_b2, timestep=0.8)
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        m.end_session_if_open()

        # 5. Explicit ref_latents + image_slots -> qf_xfm_step_refs & ratio 1.0
        ref1 = torch.randn(1, 64, 2, 3, dtype=torch.float32)
        ref2 = torch.randn(1, 64, 3, 2, dtype=torch.float32)
        out_edit = m(x, encoder_hidden_states=ctx_pos, timestep=0.9, ref_latents=[ref1, ref2], image_slots=[2, 6])
        assert tuple(out_edit.sample.shape) == (1, 64, 4, 4)
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        assert lib.steps[-1]["kind"] == "step_refs"
        assert lib.steps[-1]["num_refs"] == 2
        assert lib.steps[-1]["ref_dims"] == [[1, 64, 2, 3], [1, 64, 3, 2]]
        assert lib.steps[-1]["image_slots"] == [2, 6]
        m.end_session_if_open()

        # 6. QwenImageEditPipeline concatenated 3D hidden_states + img_shapes
        target_3d = torch.randn(1, 16, 64, dtype=torch.float32)
        ref_3d = torch.randn(1, 6, 64, dtype=torch.float32)
        concat_in = torch.cat([target_3d, ref_3d], dim=1)
        out_concat = m(
            concat_in,
            encoder_hidden_states=ctx_pos,
            timestep=0.7,
            img_shapes=[[(1, 4, 4), (1, 2, 3)]],
        )
        assert tuple(out_concat.sample.shape) == (1, 22, 64)
        assert torch.allclose(out_concat.sample[:, :16, :], target_3d)
        assert torch.all(out_concat.sample[:, 16:, :] == 0)
        assert lib.steps[-1]["kind"] == "step_refs"
        assert lib.steps[-1]["ref_dims"] == [[1, 64, 2, 3]]
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)

        # 7. Fail-closed validations: >16 refs, area > 2^24, slot length mismatch, decreasing slots
        too_many_refs = [ref1] * (QF_MAX_STEP_REFS + 1)
        with pytest.raises(ValueError, match=str(QF_MAX_STEP_REFS)):
            m(x, encoder_hidden_states=ctx_pos, timestep=0.5, ref_latents=too_many_refs)

        # Zero-RAM strided view exceeding QF_MAX_STEP_REF_AREA (4097 * 4097 > 2^24)
        huge_ref = torch.zeros((1, 1, 1, 1), dtype=torch.float32).expand(1, 64, 4097, 4097)
        assert 4097 * 4097 > QF_MAX_STEP_REF_AREA
        with pytest.raises(ValueError, match="area"):
            m(x, encoder_hidden_states=ctx_pos, timestep=0.5, ref_latents=[huge_ref])

        with pytest.raises(ValueError, match="image_slots length"):
            m(x, encoder_hidden_states=ctx_pos, timestep=0.5, ref_latents=[ref1, ref2], image_slots=[1])

        with pytest.raises(ValueError, match="non-decreasing"):
            m(x, encoder_hidden_states=ctx_pos, timestep=0.5, ref_latents=[ref1, ref2], image_slots=[5, 2])
    finally:
        m.close()


# ── Test 23: LTX-2.5 connectors, dual ctx, sigma/t norm, I2V, ref_audio, frozen audio, STG ──


def test_ltx25_contracts_stg_guard_and_frozen_audio(tmp_path: Path) -> None:
    lib = FakeQFLib()

    # 1. Connectors auto-discovery: sibling *connector*.safetensors, missing & video-only fail-closed
    bare_dir = tmp_path / "bare"
    bare_ckpt = _make_dummy_checkpoint(bare_dir / "ltx25-quantfunc.safetensors")
    with pytest.raises(RuntimeError, match="connectors_path"):
        QuantFuncLTX25Transformer3DModel.from_pretrained(bare_ckpt, lib=lib, av_unprocessed_ctx=True)

    vid_only_dir = tmp_path / "vid_only"
    vid_only_ckpt = _make_dummy_checkpoint(
        vid_only_dir / "ltx25-quantfunc.safetensors",
        tensor_keys=["model.diffusion_model.video_embeddings_connector.weight"],
    )
    with pytest.raises(RuntimeError, match="audio connector"):
        QuantFuncLTX25Transformer3DModel.from_pretrained(vid_only_ckpt, lib=lib, av_unprocessed_ctx=True)

    sib_dir = tmp_path / "with_sib"
    sib_ckpt = _make_dummy_checkpoint(sib_dir / "ltx25-quantfunc.safetensors")
    sib_conn = _make_dummy_checkpoint(
        sib_dir / "ltx25_embeddings_connector.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.proj.weight",
            "model.diffusion_model.audio_embeddings_connector.proj.weight",
        ],
    )
    m = QuantFuncLTX25Transformer3DModel.from_pretrained(sib_ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        assert Path(lib.created_configs[-1]["config"]["connectors_path"]) == sib_conn.resolve()

        # 2. Zero-weight .rope / .audio_rope stubs & CacheMixin compatibility
        assert hasattr(m, "rope") and hasattr(m, "audio_rope")
        assert list(m.rope.parameters()) == []
        assert list(m.audio_rope.parameters()) == []
        assert len(list(m.parameters())) == 1 and next(m.parameters()).numel() == 0
        with m.cache_context("cond"):
            assert m._active_cache_context == "cond"
        assert m._active_cache_context is None

        # 3. Dual av_unprocessed_ctx=False concatenates video + audio embeddings; sigma=750 -> 0.75; ratio=0.80
        vid_5d = torch.randn(1, 128, 3, 2, 2, dtype=torch.float32)  # F=3, H=2, W=2 -> N_vid=12
        aud_4d = torch.randn(1, 8, 5, 16, dtype=torch.float32)  # La=5
        v_ctx = torch.randn(1, 6, 4096, dtype=torch.float32)
        a_ctx = torch.randn(1, 6, 2048, dtype=torch.float32)

        v_out, a_out = m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=v_ctx,
            audio_encoder_hidden_states=a_ctx,
            sigma=750.0,
            av_unprocessed_ctx=False,
        )
        assert tuple(v_out.shape) == (1, 128, 3, 2, 2)
        assert tuple(a_out.shape) == (1, 8, 5, 16)
        assert lib.steps[-1]["kind"] == "step_av"
        assert lib.steps[-1]["sigma"] == pytest.approx(0.75)
        assert lib.steps[-1]["context_dims"] == [1, 6, 4096 + 2048]
        assert lib.sessions[-1]["options"]["av_unprocessed_ctx"] is False
        assert lib.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.80)

        # Dual av_unprocessed_ctx=True passes single stream
        m.end_session_if_open()
        raw_ctx = torch.randn(1, 6, 4096, dtype=torch.float32)
        m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=raw_ctx,
            timestep=500.0,
            av_unprocessed_ctx=True,
        )
        assert lib.steps[-1]["sigma"] == pytest.approx(0.50)
        assert lib.steps[-1]["context_dims"] == [1, 6, 4096]
        assert lib.sessions[-1]["options"]["av_unprocessed_ctx"] is True
        m.end_session_if_open()

        # 4. 2D video_timestep -> frame_t_scales & 2D audio_timestep -> audio_frame_t_scales + variance refusals
        vid_t = torch.tensor([[0.0] * 4 + [1000.0] * 4 + [500.0] * 4], dtype=torch.float32)
        aud_t = torch.tensor([[0.0, 0.0, 1000.0, 1000.0, 500.0]], dtype=torch.float32)
        m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=v_ctx,
            audio_encoder_hidden_states=a_ctx,
            timestep=vid_t,
            audio_timestep=aud_t,
        )
        assert lib.sessions[-1]["options"]["frame_t_scales"] == pytest.approx([0.0, 1.0, 0.5])
        assert lib.sessions[-1]["options"]["audio_frame_t_scales"] == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.5])

        # Intra-frame video variance > 1e-4 rejected
        bad_vid_t = torch.tensor([[0.0, 500.0, 0.0, 0.0] + [1000.0] * 8], dtype=torch.float32)
        with pytest.raises(ValueError, match="within latent frame"):
            m(
                vid_5d,
                audio_hidden_states=aud_4d,
                encoder_hidden_states=v_ctx,
                audio_encoder_hidden_states=a_ctx,
                timestep=bad_vid_t,
            )

        # Intra-row audio variance > 1e-4 rejected
        bad_aud_t = torch.ones(1, 5, 4, dtype=torch.float32)
        bad_aud_t[0, 2, 0] = 0.0
        with pytest.raises(ValueError, match="within audio row"):
            m(
                vid_5d,
                audio_hidden_states=aud_4d,
                encoder_hidden_states=v_ctx,
                audio_encoder_hidden_states=a_ctx,
                timestep=0.5,
                audio_timestep=bad_aud_t,
            )

        # 5. ref_audio binding + mid-session mutation fail-closed
        m.end_session_if_open()
        ref_tok = torch.randn(1, 4, 128, dtype=torch.float32)
        m.set_reference_audio(ref_tok)
        m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=v_ctx,
            audio_encoder_hidden_states=a_ctx,
            timestep=0.8,
        )
        assert lib.sessions[-1]["options"]["av_conds"][0]["kind"] == "audio"
        assert lib.steps[-1]["audio_reference_active"] == 1
        assert m._handle.reference_audio_pending() is False

        # Mid-session ref_audio change fails closed via both setter and kwarg
        other_ref = torch.randn(1, 4, 128, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="ref_audio"):
            m.set_reference_audio(other_ref)
        with pytest.raises(RuntimeError, match="ref_audio"):
            m(
                vid_5d,
                audio_hidden_states=aud_4d,
                encoder_hidden_states=v_ctx,
                audio_encoder_hidden_states=a_ctx,
                timestep=0.6,
                ref_audio=other_ref,
            )
        m.end_session_if_open()
        m.set_reference_audio(None)

        # 6. restore_frozen_audio exact latent-level restoration (module fn + method)
        s1_aud = torch.full((1, 8, 5, 16), 3.25, dtype=torch.float32)
        s2_aud = torch.full((1, 8, 5, 16), -7.5, dtype=torch.float32)
        row_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        restored = restore_frozen_audio(s2_aud, s1_aud, row_mask)
        restored_method = m.restore_frozen_audio(s2_aud, s1_aud, row_mask)
        assert torch.equal(restored, restored_method)
        assert torch.all(restored[:, :, 0, :] == 3.25)
        assert torch.all(restored[:, :, 1, :] == 3.25)
        assert torch.all(restored[:, :, 2, :] == -7.5)
        assert torch.all(restored[:, :, 3, :] == -7.5)
        assert torch.all(restored[:, :, 4, :] == 3.25)

        # 7. STG / isolate_modalities mid-session guard (fail-closed default, safe reopen when opted in)
        m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=v_ctx,
            audio_encoder_hidden_states=a_ctx,
            timestep=0.9,
            spatio_temporal_guidance_blocks=[28],
        )
        assert lib.sessions[-1]["options"]["stg_blocks"] == [28]
        sessions_before = len(lib.sessions)

        with pytest.raises(RuntimeError, match="allow_stg_session_reset"):
            m(
                vid_5d,
                audio_hidden_states=aud_4d,
                encoder_hidden_states=v_ctx,
                audio_encoder_hidden_states=a_ctx,
                timestep=0.7,
                spatio_temporal_guidance_blocks=[29],
            )
        with pytest.raises(RuntimeError, match="allow_stg_session_reset"):
            m(
                vid_5d,
                audio_hidden_states=aud_4d,
                encoder_hidden_states=v_ctx,
                audio_encoder_hidden_states=a_ctx,
                timestep=0.7,
                spatio_temporal_guidance_blocks=[28],
                isolate_modalities=True,
            )

        m.allow_stg_session_reset = True
        m(
            vid_5d,
            audio_hidden_states=aud_4d,
            encoder_hidden_states=v_ctx,
            audio_encoder_hidden_states=a_ctx,
            timestep=0.7,
            spatio_temporal_guidance_blocks=[29],
            isolate_modalities=True,
        )
        assert len(lib.sessions) == sessions_before + 1
        assert lib.sessions[-2]["ended"] is True
        assert lib.sessions[-1]["options"]["stg_blocks"] == [29]
        assert lib.sessions[-1]["options"]["disable_av_cross_attn"] is True
    finally:
        m.close()


# ── Test 24: MiniMax-H3 1D t, packed rows, sign inversion, denoise_mask & audio_enhance ──


def test_minimax_h3_1d_t_packed_rows_sign_mask_and_conds(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    lib = FakeQFLib(echo_scale=1.0)
    ckpt = _make_dummy_checkpoint(tmp_path / "minimax-h3-quantfunc-int4.safetensors")
    m = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        # 1. Bind clean AV conditions (keyframes + refs + text_token_tags -> text_vision_row_ranges)
        kf_lat = torch.randn(1, 24, 1, 4, 4, dtype=torch.float32)
        ref_img = torch.randn(1, 24, 4, 4, dtype=torch.float32)
        ref_aud = torch.randn(1, 32, 2, 3, dtype=torch.float32)
        m.set_av_conditions(
            keyframes=[{"latent": kf_lat, "frame_index": 0}],
            refs=[{"kind": "image", "latent": ref_img}, {"kind": "audio", "audio_latent": ref_aud}],
            text_token_tags=[0, 0, 1, 1, 0, 1],
        )

        # 2. Packed (num_cond + num_gen) 3D video + 3D audio + 1D multi-value timestep
        # Frame grid: T_lat=2 -> num_frames=5; lat_h=4, lat_w=4, patch=(1,2,2) -> num_gen_vid = 2*2*2 = 8
        num_cond_vid = 4
        num_gen_vid = 8
        vid_cond = torch.full((1, num_cond_vid, 96), 9.0, dtype=torch.float32)
        vid_gen = torch.arange(1 * num_gen_vid * 96, dtype=torch.float32).reshape(1, num_gen_vid, 96) + 1.0
        vid_packed = torch.cat([vid_cond, vid_gen], dim=1)

        num_cond_aud = 2
        num_gen_aud = 6  # stereo=2 * audio_t=3
        aud_cond = torch.full((1, num_cond_aud, 32), 5.0, dtype=torch.float32)
        aud_gen = torch.arange(1 * num_gen_aud * 32, dtype=torch.float32).reshape(1, num_gen_aud, 32) + 2.0
        aud_packed = torch.cat([aud_cond, aud_gen], dim=1)

        ctx = torch.randn(1, 6, 5120, dtype=torch.float32)
        # 1D timestep slots: [0]=t_video=0.25, [1]=t_audio=0.40, [2]=t_cond=0.999, [3]=1.0
        timestep_1d = torch.tensor([0.25, 0.40, 0.999, 1.0], dtype=torch.float32)
        # Packed indices layout: [text(6), cond_v(4)->slot2, cond_a(2)->slot3, gen_a(6)->slot1, gen_v(8)->slot0]
        timestep_indices = torch.tensor([0] * 6 + [2] * 4 + [3] * 2 + [1] * 6 + [0] * 8, dtype=torch.long)
        video_indices = torch.tensor(list(range(6, 10)) + list(range(18, 26)), dtype=torch.long)
        audio_indices = torch.tensor(list(range(10, 12)) + list(range(12, 18)), dtype=torch.long)

        vel_v, vel_a = m(
            vid_packed,
            audio_hidden_states=aud_packed,
            encoder_hidden_states=ctx,
            timestep=timestep_1d,
            timestep_indices=timestep_indices,
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_frames=5,
            height=4,
            width=4,
        )

        # FakeQFLib received stripped generation rows ONLY, sigma = 1 - 0.25 = 0.75, exact audio_dims & conds
        step_rec = lib.steps[-1]
        sess_opts = lib.sessions[-1]["options"]
        assert step_rec["kind"] == "step_av"
        assert step_rec["dims"] == [1, num_gen_vid, 96]
        assert step_rec["audio_dims"] == [1, num_gen_aud, 32]
        assert step_rec["sigma"] == pytest.approx(0.75)
        assert sess_opts["audio_dims"] == [1, 32, 2, 3]
        assert sess_opts["num_frames"] == 5
        assert sess_opts["text_vision_row_ranges"] == [[0, 2], [4, 1]]
        assert len(sess_opts["av_conds"]) == 3
        assert sess_opts[WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.80)

        # Returned tensors zero-pad cond prefix and negate generation segment (data_direction=True)
        assert tuple(vel_v.shape) == (1, num_cond_vid + num_gen_vid, 96)
        assert tuple(vel_a.shape) == (1, num_cond_aud + num_gen_aud, 32)
        assert torch.all(vel_v[:, :num_cond_vid, :] == 0.0)
        assert torch.allclose(vel_v[:, num_cond_vid:, :], -vid_gen)
        assert torch.all(vel_a[:, :num_cond_aud, :] == 0.0)
        assert torch.allclose(vel_a[:, num_cond_aud:, :], -aud_gen)

        # 3. Fail-closed checks: av_sigma_shift <= 0, off-grid T, denoise_mask / inpaint_mask
        x5d = torch.randn(1, 24, 2, 4, 4, dtype=torch.float32)
        a4d = torch.randn(1, 32, 2, 3, dtype=torch.float32)

        with pytest.raises(ValueError, match="av_sigma_shift"):
            m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5, av_sigma_shift_audio=0.0)
        with pytest.raises(ValueError, match="av_sigma_shift"):
            m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5, av_sigma_shift_video=-1.0)
        with pytest.raises(ValueError, match="frame_grid"):
            m(torch.randn(1, 24, 3, 4, 4), audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5)
        with pytest.raises(ValueError, match="denoise_mask"):
            m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5, denoise_mask=torch.ones_like(x5d))

        # 4. Deprecated audio_enhance=True warns ONCE (DeprecationWarning + logger.warning) & NEVER enters options_json
        m.end_session_if_open()
        with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.set_audio_enhance(True)
            m.set_audio_enhance(True)  # Second call must NOT warn again
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) == 1
        assert sum("audio_enhance" in rec.message for rec in caplog.records) == 1

        m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5)
        assert "audio_enhance" not in lib.sessions[-1]["options"]
    finally:
        m.close()


def test_minimax_h3_av_conds_ptr_stability_and_3d_geometry(tmp_path: Path) -> None:
    lib = FakeQFLib(echo_scale=1.0)
    ckpt = _make_dummy_checkpoint(tmp_path / "minimax-h3-stability.safetensors")
    m = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    try:
        # 1. Non-contiguous CPU keyframes + refs across 2 steps with intermediate allocation -> ZERO reopen!
        kf_noncontig = torch.randn(1, 24, 1, 4, 8, dtype=torch.float32)[..., ::2]
        ref_noncontig = torch.randn(1, 24, 4, 8, dtype=torch.float32)[..., ::2]
        aud_noncontig = torch.randn(1, 32, 2, 6, dtype=torch.float32)[..., ::2]
        assert not kf_noncontig.is_contiguous()
        assert not ref_noncontig.is_contiguous()
        assert not aud_noncontig.is_contiguous()

        m.set_av_conditions(
            keyframes=[{"latent": kf_noncontig, "frame_index": 0}],
            refs=[{"kind": "video_audio", "latent": ref_noncontig, "audio_latent": aud_noncontig}],
        )

        x5d = torch.ones(1, 24, 2, 4, 4, dtype=torch.float32)
        a4d = torch.ones(1, 32, 2, 3, dtype=torch.float32)
        ctx = torch.randn(1, 6, 5120, dtype=torch.float32)

        conds_step0, _ = m._serialize_av_conditions(x5d.device)
        m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.1)
        # Hold freshly allocated buffers alive across steps to expose unstable temporary buffer pointers
        churn_hold = [torch.empty(1, 24, 1, 4, 4, dtype=torch.float32) for _ in range(8)]
        gc.collect()
        conds_step1, _ = m._serialize_av_conditions(x5d.device)
        m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.5)
        del churn_hold

        assert len(lib.sessions) == 1
        assert len(lib.steps) == 2
        assert conds_step0[0]["latent_ptr"] == conds_step1[0]["latent_ptr"]
        assert conds_step0[1]["latent_ptr"] == conds_step1[1]["latent_ptr"]
        assert conds_step0[1]["audio_latent_ptr"] == conds_step1[1]["audio_latent_ptr"]
        assert len(m._av_cond_device_cache) == 1

        # Updating av_conds clears cache and triggers clean reopen; to("cpu") clears cache
        m.set_av_conditions(keyframes=[{"latent": kf_noncontig.clone(), "frame_index": 0}])
        assert m._av_cond_device_cache == {}
        m(x5d, audio_hidden_states=a4d, encoder_hidden_states=ctx, timestep=0.6)
        assert len(lib.sessions) == 2
        assert len(m._av_cond_device_cache) == 1
        m.to("cpu")
        assert m._av_cond_device_cache == {}
        m.clear_av_conditions()

        # 2. 3D hidden_states [1, 512, 96] with latent dims num_frames=5, height=32, width=32
        #    MUST resolve to dims=[1, 512, 96], width=512, height=512 and non-zero output!
        lib.sessions.clear()
        lib.steps.clear()
        x3d_512 = torch.arange(1 * 512 * 96, dtype=torch.float32).reshape(1, 512, 96) + 1.0
        out_3d = m(
            x3d_512,
            encoder_hidden_states=ctx,
            timestep=0.25,
            num_frames=5,
            height=32,
            width=32,
        )
        assert isinstance(out_3d, Transformer2DModelOutput)
        assert lib.steps[-1]["dims"] == [1, 512, 96]
        assert lib.sessions[-1]["width"] == 512
        assert lib.sessions[-1]["height"] == 512
        assert torch.all(out_3d.sample != 0.0)
        assert torch.allclose(out_3d.sample, -x3d_512)

        # Pixel dims height=512, width=512 on the exact same [1, 512, 96] latent also resolves cleanly
        out_3d_px = m(
            x3d_512,
            encoder_hidden_states=ctx,
            timestep=0.5,
            num_frames=5,
            height=512,
            width=512,
        )
        assert lib.steps[-1]["dims"] == [1, 512, 96]
        assert lib.sessions[-1]["width"] == 512
        assert lib.sessions[-1]["height"] == 512
        assert torch.allclose(out_3d_px.sample, -x3d_512)

        # Latent dims height=16, width=16 (256x256 video) also disambiguates cleanly instead of dividing to (1, 1)
        x3d_128 = torch.ones(1, 128, 96, dtype=torch.float32) * 3.0
        out_3d_256 = m(
            x3d_128,
            encoder_hidden_states=ctx,
            timestep=0.55,
            num_frames=5,
            height=16,
            width=16,
        )
        assert lib.steps[-1]["dims"] == [1, 128, 96]
        assert lib.sessions[-1]["width"] == 256
        assert lib.sessions[-1]["height"] == 256
        assert torch.allclose(out_3d_256.sample, -x3d_128)

        # Explicit cond_prefix_len=8 with latent dims (32, 32) strips 8 prefix rows -> [1, 512, 96]
        x3d_520 = torch.cat([torch.full((1, 8, 96), 7.0), x3d_512], dim=1)
        out_pref = m(
            x3d_520,
            encoder_hidden_states=ctx,
            timestep=0.6,
            num_frames=5,
            height=32,
            width=32,
            cond_prefix_len=8,
        )
        assert lib.steps[-1]["dims"] == [1, 512, 96]
        assert torch.all(out_pref.sample[:, :8, :] == 0.0)
        assert torch.allclose(out_pref.sample[:, 8:, :], -x3d_512)

        # 3. Fail-closed negative check: mismatched 3D row count WITHOUT cond_prefix_len or av_conds -> ValueError!
        with pytest.raises(ValueError, match="cond_prefix_len|set_av_conditions"):
            m(
                x3d_520,
                encoder_hidden_states=ctx,
                timestep=0.7,
                num_frames=5,
                height=32,
                width=32,
            )
        with pytest.raises(ValueError, match="cond_prefix_len|set_av_conditions"):
            m(
                torch.ones(1, 100, 96, dtype=torch.float32),
                encoder_hidden_states=ctx,
                timestep=0.7,
                num_frames=5,
                height=4,
                width=4,
            )

        # When av_conds ARE bound, packed conditioning prefix inference succeeds
        m.set_av_conditions(keyframes=[{"latent": kf_noncontig, "frame_index": 0}])
        out_bound = m(
            x3d_520,
            encoder_hidden_states=ctx,
            timestep=0.7,
            num_frames=5,
            height=32,
            width=32,
        )
        assert lib.steps[-1]["dims"] == [1, 512, 96]
        assert torch.all(out_bound.sample[:, :8, :] == 0.0)
        assert torch.allclose(out_bound.sample[:, 8:, :], -x3d_512)
    finally:
        m.close()
        assert m._av_cond_device_cache == {}


# ── Test 25: Mutation Arm M3 (all 4 models must turn RED when production rules break) ──


def test_mutation_all_4_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # (a) Krea-2 mutations: (a1) passing 3D instead of 4D; (a2) rejecting middle-padded mask
    lib_k = FakeQFLib()
    ckpt_k = _make_dummy_checkpoint(tmp_path / "m3_krea2.safetensors")
    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt_k, lib=lib_k, torch_dtype=torch.float32)
    try:
        x3d = torch.randn(1, 4, 64, dtype=torch.float32)
        pos = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.long)
        ctx = torch.randn(1, 5, 12, 2560, dtype=torch.float32)
        mid_mask = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.long)

        # (a1) Broken _prepare_latent returning 3D instead of 4D -> 4D dims contract MUST fail
        monkeypatch.setattr(
            mk,
            "_prepare_latent",
            lambda hs, **kw: (hs.contiguous(), False, None, 1, int(hs.shape[1]), False),
        )
        mk(x3d, encoder_hidden_states=ctx, position_ids=pos, timestep=0.5)
        with pytest.raises(AssertionError):
            assert len(lib_k.steps[-1]["dims"]) == 4

        # (a2) Broken mask handler rejecting middle-padded mask -> positive contract MUST fail
        monkeypatch.undo()
        import quantfunc.models.krea2 as krea2_mod

        def _broken_mask(c: torch.Tensor, m_arg: torch.Tensor | None) -> torch.Tensor:
            if m_arg is not None:
                raise ValueError("mutated: middle-padded mask rejected")
            return c

        monkeypatch.setattr(krea2_mod, "_compress_middle_padded_context", _broken_mask)
        with pytest.raises(ValueError, match="mutated"):
            mk(x3d, encoder_hidden_states=ctx, encoder_attention_mask=mid_mask, position_ids=pos, timestep=0.5)
        monkeypatch.undo()
    finally:
        mk.close()

    # (b) Qwen-Image-2.1 mutations: (b1) Edit mis-emitting 0.85; (b2) split-CFG failing to reopen
    lib_q = FakeQFLib()
    ckpt_q = _make_dummy_checkpoint(tmp_path / "m3_qi21.safetensors")
    mq = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt_q, lib=lib_q, torch_dtype=torch.float32)
    try:
        xq = torch.randn(1, 64, 4, 4, dtype=torch.float32)
        refq = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        c1 = torch.ones(1, 6, 4096, dtype=torch.float32)
        c2 = torch.full((1, 6, 4096), 3.0, dtype=torch.float32)

        # (b1) Broken _effective_quality_ratio ignoring has_refs -> Edit contract MUST fail
        monkeypatch.setattr(mq, "_effective_quality_ratio", lambda **kw: 0.85)
        mq(xq, encoder_hidden_states=c1, timestep=0.8, ref_latents=[refq])
        with pytest.raises(AssertionError):
            assert lib_q.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        mq.end_session_if_open()
        monkeypatch.undo()

        # (b2) Broken split-CFG reopen (disable reopen threshold guard) -> split-CFG contract MUST fail
        import quantfunc.models.qwenimage21 as qi21_mod

        lib_q.sessions.clear()
        mq(xq, encoder_hidden_states=c1, timestep=1.0)
        assert lib_q.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.85)
        monkeypatch.setattr(qi21_mod, "_QUALITY_FULL_EPS", 0.0)
        mq(xq, encoder_hidden_states=c2, timestep=1.0)
        with pytest.raises(AssertionError):
            assert len(lib_q.sessions) == 2 and lib_q.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(1.0)
        monkeypatch.undo()
    finally:
        mq.close()

    # (c) LTX-2.5 mutations: (c1) skipping sigma > 1.0 normalization; (c2) silently allowing STG drift
    lib_l = FakeQFLib()
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "m3_ltx25.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.float32)
    try:
        vl = torch.randn(1, 128, 2, 2, 2, dtype=torch.float32)
        al = torch.randn(1, 8, 4, 16, dtype=torch.float32)
        vc = torch.randn(1, 4, 4096, dtype=torch.float32)
        ac = torch.randn(1, 4, 2048, dtype=torch.float32)

        # (c1) Broken sigma normalization passes raw 750.0 -> FakeQFLib rejects sigma > 100!
        monkeypatch.setattr(ml, "_normalize_scalar_sigma", lambda v: float(v))
        with pytest.raises(InvalidArgError):
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, sigma=750.0)
        monkeypatch.undo()

        # (c2) Mutated STG guard (allow_stg_session_reset flipped to True) -> fail-closed guard MUST fail
        ml(
            vl,
            audio_hidden_states=al,
            encoder_hidden_states=vc,
            audio_encoder_hidden_states=ac,
            timestep=0.8,
            spatio_temporal_guidance_blocks=[28],
        )
        monkeypatch.setattr(ml, "allow_stg_session_reset", True)
        with pytest.raises(AssertionError):
            try:
                ml(
                    vl,
                    audio_hidden_states=al,
                    encoder_hidden_states=vc,
                    audio_encoder_hidden_states=ac,
                    timestep=0.6,
                    spatio_temporal_guidance_blocks=[29],
                )
                raised = False
            except RuntimeError:
                raised = True
            assert raised, "STG mid-session drift must fail closed by default"
        monkeypatch.undo()
    finally:
        ml.close()

    # (d) MiniMax-H3 mutations: (d1) omitting velocity negation; (d2) injecting audio_enhance into options_json
    lib_h = FakeQFLib(echo_scale=1.0)
    ckpt_h = _make_dummy_checkpoint(tmp_path / "m3_h3.safetensors")
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.float32)
    try:
        vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.float32)
        ah = torch.ones(1, 32, 2, 3, dtype=torch.float32)
        ch = torch.randn(1, 4, 5120, dtype=torch.float32)

        # (d1) Broken sign inversion (data_direction disabled) -> data-direction negation assertion MUST fail
        monkeypatch.setattr(mh, "_data_direction", False)
        out_v, out_a = mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.3)
        with pytest.raises(AssertionError):
            assert torch.allclose(out_v, -vh) and torch.allclose(out_a, -ah)
        mh.end_session_if_open()
        monkeypatch.undo()

        # (d2) Injecting "audio_enhance" into session options -> L0/FakeQFLib allowlist MUST reject it!
        orig_build_opts = mh._build_base_session_options

        def _mutated_opts(**kw: Any) -> dict[str, Any]:
            d = orig_build_opts(**kw)
            d["audio_enhance"] = True
            return d

        monkeypatch.setattr(mh, "_build_base_session_options", _mutated_opts)
        with pytest.raises(InvalidArgError, match="audio_enhance"):
            mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.3)
        monkeypatch.undo()

        # (d3) Breaking _normalize_scalar_sigma back to fval > 1.0 only MUST break FlowMatch final-step 1.0 -> 0.001
        monkeypatch.setattr(
            mh,
            "_normalize_scalar_sigma",
            lambda v: float(v) / 1000.0 if float(v) > 1.0 else float(v),
        )
        mut_sigmas = [mh._normalize_scalar_sigma(t) for t in (1000.0, 667.0, 334.0, 1.0)]
        with pytest.raises(AssertionError):
            assert mut_sigmas == pytest.approx([1.0, 0.667, 0.334, 0.001])
        monkeypatch.undo()

        # (d4) Breaking _has_bound_av_conds to always True MUST break mismatched-row fail-closed ValueError
        mh.clear_av_conditions()
        monkeypatch.setattr(mh, "_has_bound_av_conds", lambda: True)
        with pytest.raises((AssertionError, pytest.fail.Exception)):
            with pytest.raises(ValueError):
                mh(
                    torch.ones(1, 520, 96, dtype=torch.float32),
                    encoder_hidden_states=ch,
                    timestep=0.3,
                    num_frames=5,
                    height=32,
                    width=32,
                )
        monkeypatch.undo()

        # (d5) Clearing _av_cond_device_cache between steps (old uncached behavior) MUST break single-session ptr stability
        mh.end_session_if_open()
        lib_h.sessions.clear()
        kf_nc = torch.randn(1, 24, 1, 4, 8, dtype=torch.float32)[..., ::2]
        assert not kf_nc.is_contiguous()
        mh.set_av_conditions(keyframes=[{"latent": kf_nc, "frame_index": 0}])
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.1)
        # Simulate old behavior: drop cached device buffer and allocate churn at freed address
        mh._av_cond_device_cache.clear()
        hold = [torch.empty(1, 24, 1, 4, 4, dtype=torch.float32) for _ in range(16)]
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.5)
        del hold
        with pytest.raises(AssertionError):
            assert len(lib_h.sessions) == 1
    finally:
        mh.close()


# ── Test 41: L2 reset_request_state() across all 4 models + O(1) LoRA & cross-GPU policy ──


def test_l2_reset_request_state_and_lora_policy_preservation(tmp_path: Path) -> None:
    lora_root = tmp_path / "allowed_loras"
    lora_root.mkdir()
    outside_dir = tmp_path / "outside_loras"
    outside_dir.mkdir()
    style_lora = _make_dummy_lora(lora_root / "style.safetensors")
    outside_lora = _make_dummy_lora(outside_dir / "escape.safetensors")

    # 1. Base / Krea-2: O(1) LoRA preserves active session, to("cuda:1") preserves policy & force-rebinds,
    #    and reset_request_state() clears transient state while preserving base weights, LoRAs & dials.
    lib_k = FakeQFLib()
    ckpt_k = _make_dummy_checkpoint(tmp_path / "krea2-r4.safetensors")
    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt_k, lib=lib_k, torch_dtype=torch.float32)
    try:
        mk.configure_lora_policy(
            lora_registry={"style": style_lora},
            allowed_lora_roots=[lora_root],
            strict_registry=True,
        )
        with pytest.raises(PermissionError):
            mk.set_loras([outside_lora])
        with pytest.raises(PermissionError):
            mk.set_loras([style_lora])  # unregistered raw path refused under strict_registry=True

        mk.set_loras([{"name": "style", "scale": 0.75}])
        assert len(lib_k.loras_history) == 1
        assert len(mk.active_loras) == 1

        mk.set_quality_enhance(True)
        mk.set_cache(block_cache=0.05)
        mk.set_staged_schedule(StagedSchedule(sigmas=(1.0, 0.5, 0.0), num_steps=2))
        mk._active_cache_context = "cond"

        xk = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        ck = torch.randn(1, 4, 2560, dtype=torch.float32)
        mk(xk, encoder_hidden_states=ck, timestep=0.5)
        assert mk._session_is_open() is True
        assert lib_k.sessions[-1]["ended"] is False

        # O(1) LoRA no-op MUST keep active session open and avoid qf_xfm_set_loras
        mk.set_loras([{"name": "style", "scale": 0.75}])
        assert mk._session_is_open() is True
        assert lib_k.sessions[-1]["ended"] is False
        assert len(lib_k.loras_history) == 1

        # Cross-GPU to("cuda:1") preserves lora_registry/allowed_lora_roots/strict_registry & force-rebinds
        mk.to("cuda:1")
        assert lib_k.created_configs[-1]["device_id"] == 1
        assert len(lib_k.loras_history) == 2
        assert lib_k.loras_history[-1][0]["scale"] == pytest.approx(0.75)
        with pytest.raises(PermissionError):
            mk.set_loras([outside_lora])
        with pytest.raises(PermissionError):
            mk.set_loras([style_lora])
        mk.set_loras([{"name": "style", "scale": 0.75}])  # still O(1) hit via preserved registry
        assert len(lib_k.loras_history) == 2

        # reset_request_state() clears session, schedule, sigmas, cache_context, scratch; keeps VRAM, LoRAs, dials
        mk(xk, encoder_hidden_states=ck, timestep=0.5)
        assert mk._session_is_open() is True
        assert len(mk._scratch) > 0
        vram_before_k = mk.resident_vram_bytes()
        loras_before_k = list(mk.active_loras)
        mk.reset_request_state()
        assert mk._session_is_open() is False
        assert mk._staged_schedule is None
        assert mk._sol_sigmas is None
        assert mk._active_cache_context is None
        assert mk._scratch == {}
        assert mk.resident_vram_bytes() == vram_before_k
        assert mk.active_loras == loras_before_k
        assert mk.quality_enhance is True
        assert mk.block_cache == pytest.approx(0.05)
    finally:
        mk.close()

    # 2. LTX-2.5: ref_audio + STG + isolate_modalities isolation + pending ref_audio OOM self-heal
    lib_l = FakeQFLib()
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "ltx25-r4.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.float32)
    try:
        vl = torch.randn(1, 128, 2, 2, 2, dtype=torch.float32)
        al = torch.randn(1, 8, 4, 16, dtype=torch.float32)
        vc = torch.randn(1, 4, 4096, dtype=torch.float32)
        ac = torch.randn(1, 4, 2048, dtype=torch.float32)
        ref_a1 = torch.randn(1, 4, 128, dtype=torch.float32)

        ml(
            vl,
            audio_hidden_states=al,
            encoder_hidden_states=vc,
            audio_encoder_hidden_states=ac,
            timestep=0.8,
            ref_audio=ref_a1,
            spatio_temporal_guidance_blocks=[28],
            isolate_modalities=True,
        )
        assert "av_conds" in lib_l.sessions[-1]["options"]
        assert lib_l.steps[-1]["audio_reference_active"] == 1
        assert lib_l.sessions[-1]["options"]["stg_blocks"] == [28]
        assert lib_l.sessions[-1]["options"]["disable_av_cross_attn"] is True

        vram_before_l = ml.resident_vram_bytes()
        ml.reset_request_state()
        assert ml._ref_audio_state is None
        assert ml._ref_audio_bound is False
        assert ml._bound_stg_blocks == ()
        assert ml._bound_disable_av_cross_attn is False
        assert ml.resident_vram_bytes() == vram_before_l

        # Request 2 without ref_audio / STG sees zero leakage
        ml(
            vl,
            audio_hidden_states=al,
            encoder_hidden_states=vc,
            audio_encoder_hidden_states=ac,
            timestep=0.8,
        )
        assert "av_conds" not in lib_l.sessions[-1]["options"]
        assert "stg_blocks" not in lib_l.sessions[-1]["options"]
        assert "disable_av_cross_attn" not in lib_l.sessions[-1]["options"]
        assert lib_l.steps[-1]["audio_reference_active"] == 0
        ml.reset_request_state()

        # Simulate session_begin failing with OutOfMemoryError AFTER retain_reference_audio -> stuck pending borrow
        orig_begin = ml._handle.session_begin

        def _boom_begin(*args: Any, **kwargs: Any) -> Any:
            raise OutOfMemoryError("simulated OOM during session_begin")

        ml._handle.session_begin = _boom_begin  # type: ignore[method-assign]
        ref_a2 = torch.randn(1, 4, 128, dtype=torch.float32)
        with pytest.raises(OutOfMemoryError, match="simulated OOM"):
            ml(
                vl,
                audio_hidden_states=al,
                encoder_hidden_states=vc,
                audio_encoder_hidden_states=ac,
                timestep=0.9,
                ref_audio=ref_a2,
            )
        assert ml._handle.reference_audio_pending() is True
        ml._handle.session_begin = orig_begin  # type: ignore[method-assign]

        # reset_request_state() self-heals via unload_sync() without freeing resident model weights
        unloads_before = len(lib_l.unloads)
        ml.reset_request_state()
        assert len(lib_l.unloads) == unloads_before + 1
        assert lib_l.unloads[-1] == 0
        assert ml._handle.reference_audio_pending() is False
        assert ml.resident_vram_bytes() == vram_before_l

        # Subsequent request works cleanly (no permanent pending-copy lockup)
        ml(
            vl,
            audio_hidden_states=al,
            encoder_hidden_states=vc,
            audio_encoder_hidden_states=ac,
            timestep=0.7,
        )
        assert lib_l.steps[-1]["audio_reference_active"] == 0
        assert "av_conds" not in lib_l.sessions[-1]["options"]
    finally:
        ml.close()

    # 3. MiniMax-H3: reset_request_state() clears _av_conditions & _av_cond_device_cache
    lib_h = FakeQFLib(echo_scale=1.0)
    ckpt_h = _make_dummy_checkpoint(tmp_path / "minimax-h3-r4.safetensors")
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.float32)
    try:
        kf = torch.randn(1, 24, 1, 4, 4, dtype=torch.float32)
        vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.float32)
        ah = torch.ones(1, 32, 2, 3, dtype=torch.float32)
        ch = torch.randn(1, 4, 5120, dtype=torch.float32)
        mh.set_av_conditions(keyframes=[{"latent": kf, "frame_index": 0}], text_token_tags=[0, 1, 0])
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.25)
        assert "av_conds" in lib_h.sessions[-1]["options"]
        assert len(mh._av_cond_device_cache) == 1

        vram_before_h = mh.resident_vram_bytes()
        mh.reset_request_state()
        assert mh._av_conditions is None
        assert mh._av_cond_device_cache == {}
        assert mh._session_is_open() is False
        assert mh.resident_vram_bytes() == vram_before_h

        # Request 2 has zero av_conds leakage
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.25)
        assert "av_conds" not in lib_h.sessions[-1]["options"]
        assert "text_vision_row_ranges" not in lib_h.sessions[-1]["options"]
    finally:
        mh.close()

    # 4. Qwen-Image-2.1: reset_request_state() clears _staged_schedule/_sol_sigmas/_split_cfg_forced_full
    lib_q = FakeQFLib()
    ckpt_q = _make_dummy_checkpoint(tmp_path / "qwen-image-2.1-r4.safetensors")
    mq = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt_q, lib=lib_q, torch_dtype=torch.float32)
    try:
        mq.set_loras([style_lora])
        xq = torch.randn(1, 64, 4, 4, dtype=torch.float32)
        cq1 = torch.ones(1, 6, 4096, dtype=torch.float32)
        cq2 = torch.full((1, 6, 4096), 2.0, dtype=torch.float32)
        # Trigger split-CFG escalation to ratio 1.0
        mq(xq, encoder_hidden_states=cq1, timestep=1.0)
        mq(xq, encoder_hidden_states=cq2, timestep=1.0)
        assert mq._split_cfg_forced_full is True
        mq.set_sol_sigmas([1.0, 0.5, 0.0])

        vram_before_q = mq.resident_vram_bytes()
        loras_before_q = list(mq.active_loras)
        mq.reset_request_state()
        assert mq._session_is_open() is False
        assert mq._staged_schedule is None
        assert mq._sol_sigmas is None
        assert mq._split_cfg_forced_full is False
        assert mq._session_opened_ratio is None
        assert mq._last_sigma is None
        assert mq._scratch == {}
        assert mq.resident_vram_bytes() == vram_before_q
        assert mq.active_loras == loras_before_q

        # Request 2 single-pass T2I starts fresh at 0.85 (no leaked split-CFG escalation)
        mq(xq, encoder_hidden_states=cq1, timestep=1.0)
        assert lib_q.sessions[-1]["options"][WIRE_QUALITY_RATIO_KEY] == pytest.approx(0.85)
    finally:
        mq.close()


# ── Test 42 (Mutation Arm M8): Breaking LTX-2.5 / MiniMax-H3 request isolation turns RED ──


def test_mutation_m8_l2_request_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # (a) LTX-2.5 mutations: (a1) skipping set_reference_audio(None); (a2) skipping unload_sync()
    lib_l = FakeQFLib()
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "m8_ltx25.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.float32)
    try:
        vl = torch.randn(1, 128, 2, 2, 2, dtype=torch.float32)
        al = torch.randn(1, 8, 4, 16, dtype=torch.float32)
        vc = torch.randn(1, 4, 4096, dtype=torch.float32)
        ac = torch.randn(1, 4, 2048, dtype=torch.float32)
        ref_a = torch.randn(1, 4, 128, dtype=torch.float32)

        # (a1) Skip set_reference_audio(None) -> Request 2 leaks audio_reference_active == 1
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.8, ref_audio=ref_a)
        monkeypatch.setattr(ml, "set_reference_audio", lambda x: None)
        ml.reset_request_state()
        monkeypatch.undo()
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.8)
        with pytest.raises(AssertionError):
            assert lib_l.steps[-1]["audio_reference_active"] == 0
        ml.reset_request_state()

        # (a2) Skip unload_sync() on pending ref_audio -> reset_request_state() fails self-heal!
        orig_begin = ml._handle.session_begin
        monkeypatch.setattr(
            ml._handle,
            "session_begin",
            lambda *a, **kw: (_ for _ in ()).throw(OutOfMemoryError("oom")),
        )
        with pytest.raises(OutOfMemoryError):
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.9, ref_audio=ref_a)
        assert ml._handle.reference_audio_pending() is True
        monkeypatch.setattr(ml._handle, "session_begin", orig_begin)
        monkeypatch.setattr(ml._handle, "unload_sync", lambda: 0)
        with pytest.raises(RuntimeError, match="pending copy completion"):
            ml.reset_request_state()
        monkeypatch.undo()
        ml.reset_request_state()
        assert ml._handle.reference_audio_pending() is False
    finally:
        ml.close()

    # (b) MiniMax-H3 mutation: skipping clear_av_conditions() leaks av_conds & GPU cache
    lib_h = FakeQFLib(echo_scale=1.0)
    ckpt_h = _make_dummy_checkpoint(tmp_path / "m8_h3.safetensors")
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.float32)
    try:
        kf = torch.randn(1, 24, 1, 4, 4, dtype=torch.float32)
        vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.float32)
        ah = torch.ones(1, 32, 2, 3, dtype=torch.float32)
        ch = torch.randn(1, 4, 5120, dtype=torch.float32)
        mh.set_av_conditions(keyframes=[{"latent": kf, "frame_index": 0}])
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.25)

        monkeypatch.setattr(mh, "clear_av_conditions", lambda: None)
        mh.reset_request_state()
        monkeypatch.undo()
        with pytest.raises(AssertionError):
            assert mh._av_cond_device_cache == {}
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.25)
        with pytest.raises(AssertionError):
            assert "av_conds" not in lib_h.sessions[-1]["options"]
    finally:
        mh.close()


# ── Test 43 (U-C01): All 4 models 0-sync _context_fingerprint, in-place, >16 ABA fallback ──


def test_context_fingerprint_zero_sync_all_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_item = torch.Tensor.item
    item_calls = {"n": 0}

    def _forbid_item(self: torch.Tensor) -> Any:
        item_calls["n"] += 1
        raise AssertionError("Tensor.item() must NOT be called on the 0-sync _context_fingerprint fast path")

    # Build all 4 models with torch_dtype=bfloat16 while passing float32 caller contexts
    # to force internal .to(dtype=act_dtype) allocations on every forward() call.
    lib_k, lib_q, lib_l, lib_h = FakeQFLib(), FakeQFLib(), FakeQFLib(), FakeQFLib(echo_scale=1.0)
    ckpt_k = _make_dummy_checkpoint(tmp_path / "fp_krea2.safetensors")
    ckpt_q = _make_dummy_checkpoint(tmp_path / "fp_qi21.safetensors")
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "fp_ltx25.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ckpt_h = _make_dummy_checkpoint(tmp_path / "fp_h3.safetensors")

    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt_k, lib=lib_k, torch_dtype=torch.bfloat16)
    mq = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt_q, lib=lib_q, torch_dtype=torch.bfloat16)
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.bfloat16)
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.bfloat16)
    try:
        # 1. Forbid Tensor.item() across all 4 models' steady-state + in-place mutation forward passes
        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)

        # (a) Krea-2: FP32 -> BF16 cast + middle-padded encoder_attention_mask (triggers torch.stack every step)
        xk = torch.randn(1, 64, 2, 2, dtype=torch.bfloat16)
        ck = torch.randn(1, 5, 12, 2560, dtype=torch.float32)
        mask_k = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.long)
        mk(xk, encoder_hidden_states=ck, encoder_attention_mask=mask_k, timestep=0.9)
        mk(xk, encoder_hidden_states=ck, encoder_attention_mask=mask_k, timestep=0.5)
        assert [s["cfg_context_key"] for s in lib_k.steps[-2:]] == [1, 1]
        # In-place mutation bumps _version -> MUST rotate cfg_context_key without calling .item()
        ck.mul_(2.0)
        mk(xk, encoder_hidden_states=ck, encoder_attention_mask=mask_k, timestep=0.4)
        assert lib_k.steps[-1]["cfg_context_key"] == 2
        # In-place mutation on encoder_attention_mask also rotates key
        mask_k.copy_(torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.long))
        mk(xk, encoder_hidden_states=ck, encoder_attention_mask=mask_k, timestep=0.3)
        assert lib_k.steps[-1]["cfg_context_key"] == 3

        # (b) Qwen-Image-2.1: FP32 -> BF16 cast stable key + in-place rotation
        xq = torch.randn(1, 64, 4, 4, dtype=torch.bfloat16)
        cq = torch.randn(1, 6, 4096, dtype=torch.float32)
        mq(xq, encoder_hidden_states=cq, timestep=0.9)
        mq(xq, encoder_hidden_states=cq, timestep=0.5)
        assert [s["cfg_context_key"] for s in lib_q.steps[-2:]] == [1, 1]
        cq.mul_(3.0)
        mq(xq, encoder_hidden_states=cq, timestep=0.5)
        # Note: split-CFG guard reopens session at ratio 1.0 when a 2nd key appears, proving key rotated!
        assert len(lib_q.sessions) == 2

        # (c) LTX-2.5: FP32 -> BF16 + (v_ctx, a_ctx) torch.cat every step -> stable key 1, in-place rotates!
        vl = torch.randn(1, 128, 2, 2, 2, dtype=torch.bfloat16)
        al = torch.randn(1, 8, 4, 16, dtype=torch.float32)
        vc = torch.randn(1, 4, 4096, dtype=torch.float32)
        ac = torch.randn(1, 4, 2048, dtype=torch.float32)
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.9)
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.5)
        assert [s["cfg_context_key"] for s in lib_l.steps[-2:]] == [1, 1]
        ac.mul_(2.0)
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.4)
        assert lib_l.steps[-1]["cfg_context_key"] == 2
        vc.mul_(0.5)
        ml(vl, audio_hidden_states=al, encoder_hidden_states=vc, audio_encoder_hidden_states=ac, timestep=0.3)
        assert lib_l.steps[-1]["cfg_context_key"] == 3

        # (d) MiniMax-H3: FP32 -> BF16 stable key + in-place rotation
        vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.bfloat16)
        ah = torch.ones(1, 32, 2, 3, dtype=torch.float32)
        ch = torch.randn(1, 4, 5120, dtype=torch.float32)
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.2)
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.5)
        assert [s["cfg_context_key"] for s in lib_h.steps[-2:]] == [1, 1]
        ch.mul_(2.0)
        mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch, timestep=0.6)
        assert lib_h.steps[-1]["cfg_context_key"] == 2

        # (e) cache_context override stays 0-sync
        with mk.cache_context("explicit_cond"):
            assert mk._context_fingerprint(ck, 0) == ("explicit_cond", 0)

        assert item_calls["n"] == 0

        # 2. end_session_if_open() clears _session_ctx_keepalive on all 4 models
        for model in (mk, mq, ml, mh):
            assert len(model._session_ctx_keepalive) > 0
            model.end_session_if_open()
            assert model._session_ctx_keepalive == {}

        # 3. Overflow (>16 distinct data_ptr) falls back to 3-scalar ("val", ...) WITHOUT evicting keepalive
        monkeypatch.undo()
        counted_items = {"n": 0}
        sum_calls = {"n": 0}
        real_sum = torch.Tensor.sum

        def _count_item(self: torch.Tensor) -> Any:
            counted_items["n"] += 1
            return real_item(self)

        def _count_sum(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            sum_calls["n"] += 1
            return real_sum(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "item", _count_item)
        monkeypatch.setattr(torch.Tensor, "sum", _count_sum)

        first_16 = [torch.full((1, 4, 16), float(idx + 1), dtype=torch.float32) for idx in range(16)]
        fps_16 = [mk._context_fingerprint(t, 0) for t in first_16]
        assert counted_items["n"] == 0
        assert sum_calls["n"] == 0
        assert len(mk._session_ctx_keepalive) == 16
        assert len(set(fps_16)) == 16
        assert all(fp[0] == "ptr" for fp in fps_16)
        kept_ptrs = set(mk._session_ctx_keepalive.keys())

        # 17th and 18th distinct tensors overflow -> no eviction, no ABA collision, 3 scalar items each, 0 sum()
        t17 = torch.tensor([[[100.0, 200.0, 300.0]]], dtype=torch.float32)
        t18 = torch.tensor([[[100.0, 250.0, 300.0]]], dtype=torch.float32)
        fp17 = mk._context_fingerprint(t17, 0)
        assert counted_items["n"] == 3
        assert sum_calls["n"] == 0
        assert fp17[0] == "val"
        assert set(mk._session_ctx_keepalive.keys()) == kept_ptrs  # zero eviction!
        assert fp17 not in set(fps_16)

        fp18 = mk._context_fingerprint(t18, 0)
        assert counted_items["n"] == 6
        assert sum_calls["n"] == 0
        assert fp18[0] == "val"
        assert fp18 != fp17  # mid element distinguishes them
        assert set(mk._session_ctx_keepalive.keys()) == kept_ptrs

        # Re-querying one of the first 16 still hits 0-sync "ptr" fast path with identical key
        assert mk._context_fingerprint(first_16[0], 0) == fps_16[0]
        assert counted_items["n"] == 6

        # _release_scratch() also clears keepalive
        mk._release_scratch()
        assert mk._session_ctx_keepalive == {}
    finally:
        mk.close()
        mq.close()
        ml.close()
        mh.close()


# ── Test 43b: torch.inference_mode() 0-sync _context_fingerprint across all 4 models ──


def test_context_fingerprint_inference_mode_all_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 0-sync _context_fingerprint across all 4 models under torch.inference_mode()."""
    item_calls = {"n": 0}

    def _forbid_item(self: torch.Tensor) -> Any:
        item_calls["n"] += 1
        raise AssertionError("Tensor.item() must NOT be called on the 0-sync _context_fingerprint fast path")

    lib_k, lib_q, lib_l, lib_h = FakeQFLib(), FakeQFLib(), FakeQFLib(), FakeQFLib(echo_scale=1.0)
    ckpt_k = _make_dummy_checkpoint(tmp_path / "inf_krea2.safetensors")
    ckpt_q = _make_dummy_checkpoint(tmp_path / "inf_qi21.safetensors")
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "inf_ltx25.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ckpt_h = _make_dummy_checkpoint(tmp_path / "inf_h3.safetensors")

    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt_k, lib=lib_k, torch_dtype=torch.bfloat16)
    mq = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt_q, lib=lib_q, torch_dtype=torch.bfloat16)
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.bfloat16)
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.bfloat16)
    try:
        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)

        with torch.inference_mode():
            # (a) Krea-2: inference_mode encoder_hidden_states + encoder_attention_mask
            xk = torch.randn(1, 64, 2, 2, dtype=torch.bfloat16)
            ck1 = torch.randn(1, 5, 12, 2560, dtype=torch.float32)
            ck2 = torch.randn(1, 5, 12, 2560, dtype=torch.float32)
            mask_k1 = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.long)
            mask_k2 = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.long)
            assert ck1.is_inference() and mask_k1.is_inference()
            mk(xk, encoder_hidden_states=ck1, encoder_attention_mask=mask_k1, timestep=0.9)
            mk(xk, encoder_hidden_states=ck1, encoder_attention_mask=mask_k1, timestep=0.5)
            assert [s["cfg_context_key"] for s in lib_k.steps[-2:]] == [1, 1]
            mk(xk, encoder_hidden_states=ck2, encoder_attention_mask=mask_k1, timestep=0.4)
            assert lib_k.steps[-1]["cfg_context_key"] == 2
            mk(xk, encoder_hidden_states=ck2, encoder_attention_mask=mask_k2, timestep=0.3)
            assert lib_k.steps[-1]["cfg_context_key"] == 3

            # (b) Qwen-Image-2.1: inference_mode encoder_hidden_states + split-CFG reopen + rotation
            xq = torch.randn(1, 64, 4, 4, dtype=torch.bfloat16)
            cq1 = torch.randn(1, 6, 4096, dtype=torch.float32)
            cq2 = torch.randn(1, 6, 4096, dtype=torch.float32)
            assert cq1.is_inference()
            mq(xq, encoder_hidden_states=cq1, timestep=0.9)
            mq(xq, encoder_hidden_states=cq1, timestep=0.5)
            assert [s["cfg_context_key"] for s in lib_q.steps[-2:]] == [1, 1]
            mq(xq, encoder_hidden_states=cq2, timestep=0.4)
            assert len(lib_q.sessions) == 2
            mq(xq, encoder_hidden_states=cq1, timestep=0.3)
            assert lib_q.steps[-1]["cfg_context_key"] == 2

            # (c) LTX-2.5: inference_mode encoder_hidden_states + audio_encoder_hidden_states
            vl = torch.randn(1, 128, 2, 2, 2, dtype=torch.bfloat16)
            al = torch.randn(1, 8, 4, 16, dtype=torch.float32)
            vc1 = torch.randn(1, 4, 4096, dtype=torch.float32)
            vc2 = torch.randn(1, 4, 4096, dtype=torch.float32)
            ac1 = torch.randn(1, 4, 2048, dtype=torch.float32)
            ac2 = torch.randn(1, 4, 2048, dtype=torch.float32)
            assert vc1.is_inference() and ac1.is_inference()
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc1, audio_encoder_hidden_states=ac1, timestep=0.9)
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc1, audio_encoder_hidden_states=ac1, timestep=0.5)
            assert [s["cfg_context_key"] for s in lib_l.steps[-2:]] == [1, 1]
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc1, audio_encoder_hidden_states=ac2, timestep=0.4)
            assert lib_l.steps[-1]["cfg_context_key"] == 2
            ml(vl, audio_hidden_states=al, encoder_hidden_states=vc2, audio_encoder_hidden_states=ac2, timestep=0.3)
            assert lib_l.steps[-1]["cfg_context_key"] == 3

            # (d) MiniMax-H3: inference_mode encoder_hidden_states
            vh = torch.ones(1, 24, 2, 4, 4, dtype=torch.bfloat16)
            ah = torch.ones(1, 32, 2, 3, dtype=torch.float32)
            ch1 = torch.randn(1, 4, 5120, dtype=torch.float32)
            ch2 = torch.randn(1, 4, 5120, dtype=torch.float32)
            assert ch1.is_inference()
            mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch1, timestep=0.2)
            mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch1, timestep=0.5)
            assert [s["cfg_context_key"] for s in lib_h.steps[-2:]] == [1, 1]
            mh(vh, audio_hidden_states=ah, encoder_hidden_states=ch2, timestep=0.6)
            assert lib_h.steps[-1]["cfg_context_key"] == 2

        # (e) Inference tensors passed into forward() outside torch.inference_mode() (TextEncoder -> caller pattern)
        mk(torch.randn(1, 64, 2, 2, dtype=torch.bfloat16), encoder_hidden_states=ck1, encoder_attention_mask=mask_k1, timestep=0.2)
        assert lib_k.steps[-1]["cfg_context_key"] == 1

        # (f) Explicit cache_context remains 0-sync for inference tensors
        with mk.cache_context("explicit_inf"):
            assert mk._context_fingerprint(ck1, 0) == ("explicit_inf", 0)

        # (g) Distinguish inference tensor (ver == -1) vs normal tensor (ver >= 0) at same data_ptr & shape
        shared_buf = bytearray(1 * 4 * 2560 * 4)
        t_norm = torch.frombuffer(shared_buf, dtype=torch.float32).reshape(1, 4, 2560)
        with torch.inference_mode():
            t_inf = torch.frombuffer(shared_buf, dtype=torch.float32).reshape(1, 4, 2560)
        assert t_norm.data_ptr() == t_inf.data_ptr()
        assert t_norm.shape == t_inf.shape
        assert t_norm.stride() == t_inf.stride()
        assert not t_norm.is_inference()
        assert t_inf.is_inference()

        mk.end_session_if_open()
        fp_norm = mk._context_fingerprint(t_norm, 0)
        fp_inf = mk._context_fingerprint(t_inf, 0)
        assert fp_norm[0] == "ptr" and fp_inf[0] == "ptr"
        assert fp_norm[1][0][1] >= 0
        assert fp_inf[1][0][1] == -1
        assert fp_norm != fp_inf

        # End-to-end forward() rotation between same-address/same-shape normal vs inference tensor
        mk.end_session_if_open()
        xk_shared = torch.randn(1, 64, 2, 2, dtype=torch.bfloat16)
        mk(xk_shared, encoder_hidden_states=t_norm, timestep=0.9)
        mk(xk_shared, encoder_hidden_states=t_inf, timestep=0.5)
        assert [s["cfg_context_key"] for s in lib_k.steps[-2:]] == [1, 2]
        assert item_calls["n"] == 0

        # (h) Overflow fallback (>16 distinct data_ptr) works cleanly on inference tensors
        monkeypatch.undo()
        mk.end_session_if_open()
        for idx in range(16):
            mk._context_fingerprint(torch.full((1, 2, 4), float(idx + 1)), 0)
        with torch.inference_mode():
            t_inf_overflow = torch.tensor([[[10.0, 20.0, 30.0]]], dtype=torch.float32)
        fp_overflow = mk._context_fingerprint(t_inf_overflow, 0)
        assert fp_overflow == ("val", (((1, 1, 3), torch.float32, (10.0, 20.0, 30.0)),), 0)
    finally:
        mk.close()
        mq.close()
        ml.close()
        mh.close()


# ── Test 44 (U-C04): Vectorized LTX-2.5 video & audio timestep validation (<=1 D2H) ──


def test_ltx25_timestep_vectorized_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = FakeQFLib()
    ckpt = _make_dummy_checkpoint(
        tmp_path / "ltx25_vec_ts.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)

    real_tolist = torch.Tensor.tolist
    tolist_calls = {"n": 0}

    def _forbid_item(self: torch.Tensor) -> Any:
        raise AssertionError("Tensor.item() must NOT be called in vectorized LTX-2.5 timestep validation")

    def _spy_tolist(self: torch.Tensor) -> Any:
        tolist_calls["n"] += 1
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    monkeypatch.setattr(torch.Tensor, "tolist", _spy_tolist)

    try:
        # 1. 0D & 1D video/audio timesteps: <=1 tolist D2H, 0 .item()
        tolist_calls["n"] = 0
        s0, sc0 = ml._extract_video_sigma_and_frame_scales(torch.tensor(500.0), None, batch=1, ppf=16, pph=2, ppw=2)
        assert tolist_calls["n"] <= 1
        assert s0 == pytest.approx([0.5]) and sc0 is None

        tolist_calls["n"] = 0
        s1, sc1 = ml._extract_video_sigma_and_frame_scales(
            torch.tensor([800.0, 400.0]), None, batch=2, ppf=16, pph=2, ppw=2
        )
        assert tolist_calls["n"] <= 1
        assert s1 == pytest.approx([0.8, 0.4]) and sc1 is None

        tolist_calls["n"] = 0
        a1 = ml._extract_audio_frame_scales(torch.tensor([0.5]), [0.5], batch=1, audio_rows=16)
        assert tolist_calls["n"] <= 1
        assert a1 is None

        # 2. Multi-dim I2V video (F=16, H=2, W=2) & audio (La=16): STRICTLY 1 D2H each, 0 .item()
        f_count = 16
        expected_vid_scales = [round(i / (f_count - 1), 4) for i in range(f_count)]
        vid_t_2d = torch.tensor(
            [[sc * 1000.0 for sc in expected_vid_scales for _ in range(4)]],
            dtype=torch.float32,
        )
        tolist_calls["n"] = 0
        sv, vid_scales = ml._extract_video_sigma_and_frame_scales(
            vid_t_2d, None, batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert tolist_calls["n"] == 1
        assert sv == pytest.approx([1.0])
        assert vid_scales == pytest.approx(expected_vid_scales, abs=1e-4)

        # 5D video timestep [B=1, C=1, F=16, H=2, W=2]
        vid_t_5d = vid_t_2d.reshape(1, 1, f_count, 2, 2)
        tolist_calls["n"] = 0
        sv5, vid_scales5 = ml._extract_video_sigma_and_frame_scales(
            vid_t_5d, None, batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert tolist_calls["n"] == 1
        assert sv5 == pytest.approx([1.0])
        assert vid_scales5 == pytest.approx(expected_vid_scales, abs=1e-4)

        # Multi-dim audio timestep [B=1, La=16, 4] & 4D [B=1, 8, La=16, 16]
        expected_aud_scales = [0.0 if i < 4 else 1.0 if i >= 12 else 0.5 for i in range(f_count)]
        aud_t_3d = torch.tensor(
            [[[sc * 1000.0] * 4 for sc in expected_aud_scales]],
            dtype=torch.float32,
        )
        tolist_calls["n"] = 0
        aud_scales = ml._extract_audio_frame_scales(aud_t_3d, [1.0], batch=1, audio_rows=f_count)
        assert tolist_calls["n"] == 1
        assert aud_scales == pytest.approx(expected_aud_scales)

        aud_t_4d = torch.zeros(1, 8, f_count, 16, dtype=torch.float32)
        for r, sc in enumerate(expected_aud_scales):
            aud_t_4d[:, :, r, :] = sc * 1000.0
        tolist_calls["n"] = 0
        aud_scales4 = ml._extract_audio_frame_scales(aud_t_4d, [1.0], batch=1, audio_rows=f_count)
        assert tolist_calls["n"] == 1
        assert aud_scales4 == pytest.approx(expected_aud_scales)

        # 3. All 5 negative categories on BOTH video and audio (NaN, +Inf, -Inf mixed, negative, intra-frame variance, batch mismatch)
        # (a) NaN
        bad_nan_v = vid_t_2d.clone()
        bad_nan_v[0, 5] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_video_sigma_and_frame_scales(bad_nan_v, None, batch=1, ppf=f_count, pph=2, ppw=2)

        bad_nan_a = aud_t_3d.clone()
        bad_nan_a[0, 3, 2] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_audio_frame_scales(bad_nan_a, [1.0], batch=1, audio_rows=f_count)

        # (b) +Inf and -Inf mixed with positive (lo=-inf, hi=positive specifically kills missing isfinite(lo) check!)
        bad_pinf_v = vid_t_2d.clone()
        bad_pinf_v[0, 4] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_video_sigma_and_frame_scales(bad_pinf_v, None, batch=1, ppf=f_count, pph=2, ppw=2)

        bad_ninf_v = torch.ones(1, f_count * 4, dtype=torch.float32)
        bad_ninf_v[0, 4] = float("-inf")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_video_sigma_and_frame_scales(bad_ninf_v, None, batch=1, ppf=f_count, pph=2, ppw=2)

        bad_pinf_a = aud_t_3d.clone()
        bad_pinf_a[0, 1, 0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_audio_frame_scales(bad_pinf_a, [1.0], batch=1, audio_rows=f_count)

        bad_ninf_a = torch.ones(1, f_count, 4, dtype=torch.float32)
        bad_ninf_a[0, 2, 1] = float("-inf")
        with pytest.raises(ValueError, match="finite"):
            ml._extract_audio_frame_scales(bad_ninf_a, [1.0], batch=1, audio_rows=f_count)

        # 1D NaN / Inf / negative
        with pytest.raises(ValueError, match="finite"):
            ml._extract_video_sigma_and_frame_scales(torch.tensor([float("nan")]), None, batch=1, ppf=4, pph=2, ppw=2)
        with pytest.raises(ValueError, match="non-negative"):
            ml._extract_video_sigma_and_frame_scales(torch.tensor([-0.1]), None, batch=1, ppf=4, pph=2, ppw=2)
        with pytest.raises(ValueError, match="finite"):
            ml._extract_audio_frame_scales(torch.tensor([float("inf")]), [0.5], batch=1, audio_rows=4)
        with pytest.raises(ValueError, match="non-negative"):
            ml._extract_audio_frame_scales(torch.tensor([-0.1]), [0.5], batch=1, audio_rows=4)

        # (c) Negative finite value
        bad_neg_v = vid_t_2d.clone()
        bad_neg_v[0, 0] = -0.25
        with pytest.raises(ValueError, match="non-negative"):
            ml._extract_video_sigma_and_frame_scales(bad_neg_v, None, batch=1, ppf=f_count, pph=2, ppw=2)

        bad_neg_a = aud_t_3d.clone()
        bad_neg_a[0, 0, 0] = -0.25
        with pytest.raises(ValueError, match="non-negative"):
            ml._extract_audio_frame_scales(bad_neg_a, [1.0], batch=1, audio_rows=f_count)

        # (d) Intra-frame / intra-row variance > 1e-4
        bad_var_v = vid_t_2d.clone()
        bad_var_v[0, 4] += 10.0  # frame 1 element 0 diverges from elements 1..3
        with pytest.raises(ValueError, match="within latent frame"):
            ml._extract_video_sigma_and_frame_scales(bad_var_v, None, batch=1, ppf=f_count, pph=2, ppw=2)

        bad_var_a = aud_t_3d.clone()
        bad_var_a[0, 5, 0] = 0.0  # row 5 diverges
        with pytest.raises(ValueError, match="within audio row"):
            ml._extract_audio_frame_scales(bad_var_a, [1.0], batch=1, audio_rows=f_count)

        # (e) Batch scale mismatch (batch=2)
        b2_vid = torch.cat([vid_t_2d, torch.ones_like(vid_t_2d) * 1000.0], dim=0)
        with pytest.raises(ValueError, match="must match across batch"):
            ml._extract_video_sigma_and_frame_scales(b2_vid, None, batch=2, ppf=f_count, pph=2, ppw=2)

        b2_aud = torch.cat([aud_t_3d, torch.ones_like(aud_t_3d) * 1000.0], dim=0)
        with pytest.raises(ValueError, match="must match across batch"):
            ml._extract_audio_frame_scales(b2_aud, [1.0, 1.0], batch=2, audio_rows=f_count)

        # 4. Additional branch coverage: 1D + sigma_override, empty 1D, 1D len < batch, 2D [B, F],
        #    multi-dim + tensor/scalar sigma_override, invalid 5D / non-5D, scalar audio, audio shapes,
        #    all-zero denom == 0.0, and full-denoise >= 0.9999 -> None.
        sv_ov1, sc_ov1 = ml._extract_video_sigma_and_frame_scales(
            torch.tensor([500.0]), torch.tensor([250.0]), batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert sv_ov1 == pytest.approx([0.25]) and sc_ov1 is None

        with pytest.raises(ValueError, match="non-empty"):
            ml._extract_video_sigma_and_frame_scales(torch.tensor([]), None, batch=1, ppf=f_count, pph=2, ppw=2)
        with pytest.raises(ValueError, match="does not match batch size"):
            ml._extract_video_sigma_and_frame_scales(
                torch.tensor([500.0, 400.0]), None, batch=3, ppf=f_count, pph=2, ppw=2
            )
        with pytest.raises(ValueError, match="multi-dim video_timestep batch"):
            ml._extract_video_sigma_and_frame_scales(
                torch.ones(2, f_count * 4), None, batch=1, ppf=f_count, pph=2, ppw=2
            )
        with pytest.raises(ValueError, match="per-item shape"):
            ml._extract_video_sigma_and_frame_scales(
                torch.ones(1, 1, 3, 2, 2), None, batch=1, ppf=5, pph=2, ppw=2
            )
        with pytest.raises(ValueError, match="per-item shape"):
            ml._extract_video_sigma_and_frame_scales(
                torch.ones(1, 3), None, batch=1, ppf=f_count, pph=2, ppw=2
            )

        # 2D [B, F] + scalar sigma_override and batch=2 tensor sigma_override
        vid_bf = torch.tensor([[sc * 1000.0 for sc in expected_vid_scales]], dtype=torch.float32)
        sv_bf, sc_bf = ml._extract_video_sigma_and_frame_scales(
            vid_bf, 800.0, batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert sv_bf == pytest.approx([0.8])
        assert sc_bf == pytest.approx([min(1.0, (sc * 1000.0) / 800.0) for sc in expected_vid_scales], abs=1e-4)

        vid_bf_b2 = vid_bf.repeat(2, 1)
        sv_bf_b2, sc_bf_b2 = ml._extract_video_sigma_and_frame_scales(
            vid_bf_b2, torch.tensor([1000.0, 1000.0]), batch=2, ppf=f_count, pph=2, ppw=2
        )
        assert sv_bf_b2 == pytest.approx([1.0, 1.0])
        assert sc_bf_b2 == pytest.approx(expected_vid_scales, abs=1e-4)

        # Video all-zero (denom == 0.0 -> 1.0 scales -> None) & near-1.0 (>= 0.9999 -> None)
        sv_zero, sc_zero = ml._extract_video_sigma_and_frame_scales(
            torch.zeros(1, f_count), None, batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert sv_zero == pytest.approx([0.0]) and sc_zero is None
        vid_full = torch.full((1, f_count), 1000.0, dtype=torch.float32)
        vid_full[0, 0] = 999.95
        sv_full, sc_full = ml._extract_video_sigma_and_frame_scales(
            vid_full, None, batch=1, ppf=f_count, pph=2, ppw=2
        )
        assert sv_full == pytest.approx([1.0]) and sc_full is None

        # Audio: scalar Python float, batch mismatch, [B, audio_rows], non-canonical % audio_rows == 0,
        # invalid shape, all-zero denom == 0.0, and full denoise >= 0.9999 -> None
        assert ml._extract_audio_frame_scales(0.5, [0.5], batch=1, audio_rows=f_count) is None
        with pytest.raises(ValueError, match="audio_timestep batch"):
            ml._extract_audio_frame_scales(aud_t_3d.repeat(2, 1, 1), [1.0], batch=1, audio_rows=f_count)
        assert ml._extract_audio_frame_scales(
            torch.tensor([[sc * 1000.0 for sc in expected_aud_scales]]), [1.0], batch=1, audio_rows=f_count
        ) == pytest.approx(expected_aud_scales)
        aud_noncanon = torch.tensor(
            [[sc * 1000.0 for sc in expected_aud_scales for _ in range(4)]],
            dtype=torch.float32,
        ).reshape(1, f_count, 2, 2)
        assert ml._extract_audio_frame_scales(aud_noncanon, [1.0], batch=1, audio_rows=f_count) == pytest.approx(
            expected_aud_scales
        )
        with pytest.raises(ValueError, match="does not align with audio_rows"):
            ml._extract_audio_frame_scales(torch.ones(1, 3), [1.0], batch=1, audio_rows=f_count)
        assert ml._extract_audio_frame_scales(
            torch.zeros(1, f_count, 4), [0.0], batch=1, audio_rows=f_count
        ) is None
        aud_full = torch.full((1, f_count, 4), 1000.0, dtype=torch.float32)
        aud_full[0, 0, :] = 999.95
        assert ml._extract_audio_frame_scales(aud_full, [1.0], batch=1, audio_rows=f_count) is None
    finally:
        ml.close()


# ── Test 45: Keepalive lifecycle (Tier-2 18-round, Tier-3, OOM/pre-step, meta), stride & batch=2 ──


def test_context_fingerprint_lifecycle_stride_and_error_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib_k, lib_q, lib_l, lib_h = FakeQFLib(), FakeQFLib(), FakeQFLib(), FakeQFLib(echo_scale=1.0)
    ckpt_k = _make_dummy_checkpoint(tmp_path / "lc_krea2.safetensors")
    ckpt_q = _make_dummy_checkpoint(tmp_path / "lc_qi21.safetensors")
    ckpt_l = _make_dummy_checkpoint(
        tmp_path / "lc_ltx25.safetensors",
        tensor_keys=[
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    ckpt_h = _make_dummy_checkpoint(tmp_path / "lc_h3.safetensors")

    mk = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt_k, lib=lib_k, torch_dtype=torch.float32)
    mq = QuantFuncQwenImage21Transformer2DModel.from_pretrained(ckpt_q, lib=lib_q, torch_dtype=torch.float32)
    ml = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_l, lib=lib_l, torch_dtype=torch.float32)
    mh = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h, lib=lib_h, torch_dtype=torch.float32)
    try:
        # 1. _extract_batch_sigmas: empty, batch > 1, len < batch
        with pytest.raises(ValueError, match="non-empty"):
            mk._extract_batch_sigmas(torch.tensor([]), 1)
        assert mk._extract_batch_sigmas(torch.tensor([0.9, 0.5]), 2) == pytest.approx([0.9, 0.5])
        with pytest.raises(ValueError, match="does not match batch size"):
            mk._extract_batch_sigmas(torch.tensor([0.9, 0.5]), 3)

        # 2. Overflow n == 0 empty tensor in _context_fingerprint
        for idx in range(16):
            mk._context_fingerprint(torch.full((1, 2, 4), float(idx + 1)), 0)
        assert mk._context_fingerprint(torch.empty(0, dtype=torch.float32), 0) == (
            "val",
            (((0,), torch.float32, (0.0, 0.0, 0.0)),),
            0,
        )
        mk._session_ctx_keepalive.clear()

        # 3. Transposed view m1 vs m1.t() non-collision (same data_ptr, _version, shape, dtype, device; diff stride!)
        m1 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        m2 = m1.t()
        assert m1.data_ptr() == m2.data_ptr()
        assert m1.shape == m2.shape
        assert m1.stride() != m2.stride()
        assert mk._context_fingerprint(m1, 0) != mk._context_fingerprint(m2, 0)
        mk._session_ctx_keepalive.clear()

        # End-to-end square 4D Krea-2 transposed prompt rotates cfg_context_key from 1 -> 2
        xk_sq = torch.randn(1, 64, 2, 2, dtype=torch.float32)
        ck_sq = torch.arange(1 * 12 * 12 * 2560, dtype=torch.float32).reshape(1, 12, 12, 2560)
        ck_sq_t = ck_sq.transpose(1, 2)
        assert ck_sq.data_ptr() == ck_sq_t.data_ptr()
        assert ck_sq.shape == ck_sq_t.shape
        assert ck_sq.stride() != ck_sq_t.stride()
        mk(xk_sq, encoder_hidden_states=ck_sq, timestep=0.9)
        mk(xk_sq, encoder_hidden_states=ck_sq_t, timestep=0.9)
        assert [s["cfg_context_key"] for s in lib_k.steps[-2:]] == [1, 2]
        mk.end_session_if_open()
        lib_k.sessions.clear()
        lib_k.steps.clear()

        # 4. batch=2 0-sync across all 4 models + optional second ctx None on krea2/ltx25
        item_calls = {"n": 0}

        def _forbid_item(self: torch.Tensor) -> Any:
            item_calls["n"] += 1
            raise AssertionError("Tensor.item() forbidden in 0-sync forward path")

        monkeypatch.setattr(torch.Tensor, "item", _forbid_item)

        # Krea-2 batch=2, encoder_attention_mask=None
        out_k2 = mk(
            torch.randn(2, 64, 2, 2),
            encoder_hidden_states=torch.randn(2, 4, 2560),
            encoder_attention_mask=None,
            timestep=torch.tensor([0.9, 0.8]),
        ).sample
        assert out_k2.shape == (2, 64, 2, 2)

        # Qwen-Image-2.1 batch=2
        out_q2 = mq(
            torch.randn(2, 64, 4, 4),
            encoder_hidden_states=torch.randn(2, 4, 4096),
            timestep=torch.tensor([0.9, 0.8]),
        ).sample
        assert out_q2.shape == (2, 64, 4, 4)

        # LTX-2.5 batch=2, audio_encoder_hidden_states=None & video-only (audio_hidden_states=None)
        out_l2 = ml(
            torch.randn(2, 128, 2, 2, 2),
            audio_hidden_states=torch.randn(2, 8, 4, 16),
            encoder_hidden_states=torch.randn(2, 4, 4096),
            audio_encoder_hidden_states=None,
            timestep=torch.tensor([0.9, 0.8]),
        )
        assert out_l2[0].shape == (2, 128, 2, 2, 2)
        out_l_vid_only = ml(
            torch.randn(1, 128, 2, 2, 2),
            audio_hidden_states=None,
            encoder_hidden_states=torch.randn(1, 4, 4096),
            audio_encoder_hidden_states=None,
            timestep=0.9,
        ).sample
        assert out_l_vid_only.shape == (1, 128, 2, 2, 2)

        # MiniMax-H3 batch=2
        out_h2 = mh(
            torch.ones(2, 24, 2, 4, 4),
            audio_hidden_states=torch.ones(2, 32, 2, 3),
            encoder_hidden_states=torch.randn(2, 4, 5120),
            timestep=0.25,
        )
        assert out_h2[0].shape == (2, 24, 2, 4, 4)
        assert item_calls["n"] == 0

        # 5. Tier-2 18-round auto-reopen keepalive reset: 0 .item() on round 17/18, consistent cfg_context_key=1
        mk.end_session_if_open()
        lib_k.sessions.clear()
        lib_k.steps.clear()
        xk1 = torch.randn(1, 64, 2, 2)
        for _rnd in range(18):
            prompt_r = torch.randn(1, 4, 2560)
            mk(xk1, encoder_hidden_states=prompt_r, timestep=0.9)
            assert len(mk._session_ctx_keepalive) == 1
            assert lib_k.steps[-1]["cfg_context_key"] == 1
            mk(xk1, encoder_hidden_states=prompt_r, timestep=0.5)
            assert len(mk._session_ctx_keepalive) == 1
            assert lib_k.steps[-1]["cfg_context_key"] == 1
        assert len(lib_k.sessions) == 18
        assert item_calls["n"] == 0
        monkeypatch.undo()

        # 6. Tier-3 SamplingSession clears prior keepalive on __enter__ & clears on normal/error __exit__
        assert len(mk._session_ctx_keepalive) == 1
        with mk.sampling_session(width=16, height=16, num_steps=2, max_context_dims=(1, 64, 2560)):
            assert len(mk._session_ctx_keepalive) == 0
            mk(xk1, encoder_hidden_states=torch.randn(1, 4, 2560), timestep=0.9)
            assert len(mk._session_ctx_keepalive) == 1
        assert len(mk._session_ctx_keepalive) == 0

        with pytest.raises(RuntimeError, match="tier3 abort"):
            with mk.sampling_session(width=16, height=16, num_steps=2, max_context_dims=(1, 64, 2560)):
                mk(xk1, encoder_hidden_states=torch.randn(1, 4, 2560), timestep=0.9)
                assert len(mk._session_ctx_keepalive) == 1
                raise RuntimeError("tier3 abort")
        assert len(mk._session_ctx_keepalive) == 0

        # 7. End-to-end forward() producer_stream passthrough to session_begin AND step
        mk.end_session_if_open()
        custom_stream = 0xCAFEBABE
        monkeypatch.setattr(mk, "_resolve_producer_stream", lambda t: custom_stream)
        mk(xk1, encoder_hidden_states=torch.randn(1, 4, 2560), timestep=0.9)
        assert lib_k.sessions[-1]["producer_stream"] == custom_stream
        assert lib_k.steps[-1]["producer_stream"] == custom_stream
        monkeypatch.undo()

        # 8. Exception zero-retention: session_begin OOM, pre-step prep OOM (qwenimage21/ltx25/minimax_h3), to("meta")
        # (a) session_begin OOM with previously open session
        assert len(mk._session_ctx_keepalive) == 1
        orig_k_begin = mk._handle.session_begin

        def _oom_begin(*args: Any, **kwargs: Any) -> Any:
            raise OutOfMemoryError("oom session_begin")

        mk._handle.session_begin = _oom_begin  # type: ignore[method-assign]
        with pytest.raises(OutOfMemoryError, match="oom session_begin"):
            mk(xk1, encoder_hidden_states=torch.randn(1, 4, 2560), timestep=0.95)
        mk._handle.session_begin = orig_k_begin  # type: ignore[method-assign]
        assert len(mk._session_ctx_keepalive) == 0
        assert not mk._session_is_open()

        # (b) pre-step prep OOM after _ensure_session_step: qwenimage21 ref_i prep
        xq1 = torch.randn(1, 64, 4, 4)
        cq1 = torch.randn(1, 8, 4096)
        bad_ref = torch.randn(1, 64, 4, 4)

        class _BoomTensor:
            shape = bad_ref.shape

            def to(self, *args: Any, **kwargs: Any) -> Any:
                raise OutOfMemoryError("oom ref prep")

        mq._session_ctx_keepalive.clear()
        orig_validate = mq._validate_refs_and_slots
        monkeypatch.setattr(mq, "_validate_refs_and_slots", lambda refs, slots, txt_len, batch: [0])
        with pytest.raises(OutOfMemoryError, match="oom ref prep"):
            mq(
                xq1,
                encoder_hidden_states=cq1,
                timestep=0.9,
                ref_latents=[_BoomTensor()],  # type: ignore[list-item]
            )
        monkeypatch.setattr(mq, "_validate_refs_and_slots", orig_validate)
        assert len(mq._session_ctx_keepalive) == 0
        assert not mq._session_is_open()

        # (c) pre-step prep OOM after _ensure_session_step: ltx25 & minimax_h3 audio scratch
        orig_l_scratch = ml._ensure_scratch

        def _oom_l_aud(key: str, *args: Any, **kwargs: Any) -> torch.Tensor:
            if key == "ltx25_aud_out":
                raise OutOfMemoryError("oom ltx25 aud")
            return orig_l_scratch(key, *args, **kwargs)

        monkeypatch.setattr(ml, "_ensure_scratch", _oom_l_aud)
        with pytest.raises(OutOfMemoryError, match="oom ltx25 aud"):
            ml(
                torch.randn(1, 128, 2, 2, 2),
                audio_hidden_states=torch.randn(1, 8, 4, 16),
                encoder_hidden_states=torch.randn(1, 4, 4096),
                audio_encoder_hidden_states=torch.randn(1, 4, 2048),
                timestep=0.9,
            )
        monkeypatch.undo()
        assert len(ml._session_ctx_keepalive) == 0
        assert not ml._session_is_open()

        orig_h_scratch = mh._ensure_scratch

        def _oom_h_aud(key: str, *args: Any, **kwargs: Any) -> torch.Tensor:
            if key == "h3_aud_out":
                raise OutOfMemoryError("oom h3 aud")
            return orig_h_scratch(key, *args, **kwargs)

        monkeypatch.setattr(mh, "_ensure_scratch", _oom_h_aud)
        with pytest.raises(OutOfMemoryError, match="oom h3 aud"):
            mh(
                torch.ones(1, 24, 2, 4, 4),
                audio_hidden_states=torch.ones(1, 32, 2, 3),
                encoder_hidden_states=torch.randn(1, 4, 5120),
                timestep=0.25,
            )
        monkeypatch.undo()
        assert len(mh._session_ctx_keepalive) == 0
        assert not mh._session_is_open()

        # (d) to("meta") clears _session_ctx_keepalive (both cpu->meta and meta->meta)
        mk(xk1, encoder_hidden_states=torch.randn(1, 4, 2560), timestep=0.9)
        assert len(mk._session_ctx_keepalive) == 1
        mk.to("meta")
        assert len(mk._session_ctx_keepalive) == 0
        mk._session_ctx_keepalive[999] = xk1
        mk.to("meta")
        assert len(mk._session_ctx_keepalive) == 0
    finally:
        mk.close()
        mq.close()
        ml.close()
        mh.close()
