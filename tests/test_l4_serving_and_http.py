"""L4 unit & mutation tests: ServingPipeline, HTTP/FastAPI example, and Mutation M10."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import struct
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from conftest import FakeQFLib

import quantfunc.accelerate.serving as acc_serving_mod
import quantfunc.core.lora as lora_mod
import quantfunc.pipelines.serving as serving_pipe_mod
from quantfunc import (
    UNSET,
    FlowMatchResMultistepScheduler,
    MultiStageRunner,
    ServingPipeline,
    SolConfig,
    StageSpec,
    load_krea2_pipeline,
    load_ltx25_pipeline,
    load_minimax_h3_pipeline,
    load_qwenimage21_pipeline,
)
from quantfunc.models.krea2 import QuantFuncKrea2Transformer2DModel
from quantfunc.models.ltx25 import QuantFuncLTX25Transformer3DModel
from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel
from quantfunc.pipelines._common import finalize_pipeline_load

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_dummy_ckpt(path: Path, keys: list[str] | None = None) -> Path:
    raw = struct.pack("<2f", 1.0, 2.0)
    tensor_keys = keys or ["transformer_blocks.0.attn.to_q.weight"]
    header = json.dumps(
        {k: {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, len(raw)]} for k in tensor_keys},
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(raw)
    return path


def _make_dummy_lora(path: Path, val: float = 1.0) -> Path:
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


class _DummySched:
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


class _WarmRecordPipe:
    def __init__(self, transformer: Any) -> None:
        self.transformer = transformer
        self.scheduler = _DummySched()
        self.calls: list[dict[str, Any]] = []
        self.on_call: Any = None

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if callable(self.on_call):
            self.on_call(kwargs)
        rec = {
            "kwargs": dict(kwargs),
            "quality_enhance": self.transformer.quality_enhance,
            "block_cache": self.transformer.block_cache,
            "step_cache": self.transformer.step_cache,
            "sol": self.transformer.sol_config,
            "attention_backend": self.transformer.attention_backend,
            "active_loras": [(e.path, e.scale) for e in self.transformer.active_loras],
        }
        self.calls.append(rec)
        if kwargs.get("output_type") == "latent":
            return {"latents": torch.ones(1, 4, 8, 8), "meta": rec}
        return rec


def _load_http_example() -> types.ModuleType:
    path = REPO_ROOT / "examples" / "12_http_fastapi_server.py"
    spec = importlib.util.spec_from_file_location("_example_12_http_fastapi_server", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _http_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return int(err.code), json.loads(err.read().decode("utf-8"))


# ── Test 46: ServingPipeline sync/async, stages, security, status & close ───


def test_serving_pipeline_sync_async_stages_status_and_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (0) Pipeline without QuantFunc transformer fails closed
    with pytest.raises(ValueError, match="QuantFunc transformer"):
        ServingPipeline(SimpleNamespace(scheduler=_DummySched()))

    lora_root = tmp_path / "secret_host_loras"
    outside_root = tmp_path / "outside_host_loras"
    style_lora = _make_dummy_lora(lora_root / "style.safetensors", val=1.0)
    unreg_lora = _make_dummy_lora(outside_root / "unreg.safetensors", val=2.0)

    ckpt = _make_dummy_ckpt(tmp_path / "krea2-serving.safetensors")
    lib = FakeQFLib()
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    pipe = _WarmRecordPipe(xfm)

    # Spy on _read_safetensors_header to prove 0 disk reads on O(1) hit AND on strict PermissionError
    header_reads = 0
    orig_read_hdr = lora_mod._read_safetensors_header

    def _spy_read_hdr(path: str) -> dict[str, Any]:
        nonlocal header_reads
        header_reads += 1
        return orig_read_hdr(path)

    monkeypatch.setattr(lora_mod, "_read_safetensors_header", _spy_read_hdr)

    # Configured with lora_registry ONLY (allowed_lora_roots=None) -> default strict_lora_registry=True!
    server = ServingPipeline(
        pipe,
        lora_registry={"style": str(style_lora)},
        sticky_loras=True,
        busy_timeout=2.0,
    )
    try:
        # (1) Single-stage <-> two-stage (Sequence[StageSpec] & MultiStageRunner) online switching + all conditions
        ref_lat = [torch.ones(1, 4, 8, 8)]
        r1 = server.generate(
            "alpine sunrise",
            negative_prompt="blurry",
            image=b"img_bytes",
            mask=b"mask_bytes",
            ref_latents=ref_lat,
            image_slots=[1],
            overrides={
                "loras": [{"name": "style", "scale": 0.8}],
                "fbcache": 0.05,
                "sol": {"enabled": True, "tau": 1.3},
                "quality_enhance": True,
                "attention_backend": "int8",
            },
        )
        assert r1["kwargs"]["prompt"] == "alpine sunrise"
        assert r1["kwargs"]["negative_prompt"] == "blurry"
        assert r1["kwargs"]["image"] == b"img_bytes"
        assert r1["kwargs"]["mask"] == b"mask_bytes"
        assert r1["kwargs"]["ref_latents"] is ref_lat
        assert r1["kwargs"]["image_slots"] == [1]
        assert r1["quality_enhance"] is True
        assert r1["block_cache"] == pytest.approx(0.05)
        assert r1["sol"].enabled is True
        assert r1["attention_backend"] == "int8"
        assert len(lib.loras_history) == 1
        assert header_reads == 1

        # Req 2: Same LoRA stack + two-stage Sequence[StageSpec] -> 0 extra native binds & 0 header reads!
        two_stages = [
            StageSpec(name="draft", width=256, height=256, num_inference_steps=4, output_type="latent"),
            StageSpec(
                name="refine",
                width=512,
                height=512,
                num_inference_steps=4,
                denoise=0.5,
                transformer_overrides={"quality_enhance": True},
                output_type="pil",
            ),
        ]
        r2_stages = server.generate(
            "two stage scene",
            overrides={"loras": [{"name": "style", "scale": 0.8}]},
            stages=two_stages,
            return_all_stages=True,
        )
        assert len(r2_stages) == 2
        assert len(lib.loras_history) == 1  # O(1) hit!
        assert header_reads == 1  # 0 extra disk reads!
        assert r2_stages[1].output["quality_enhance"] is True
        assert xfm.quality_enhance is False  # Restored cleanly!

        # MultiStageRunner instance also accepted directly
        r2b = server.generate("runner direct", stages=MultiStageRunner(two_stages))
        assert r2b["quality_enhance"] is True

        # Req 3: Omitting loras resets back to default_loras=() -> [] (prevents cross-tenant leak!)
        r3 = server.generate("clean prompt")
        assert r3["active_loras"] == []
        assert xfm.active_loras == []
        assert len(lib.loras_history) == 2

        # (2) StageSpec.transformer_overrides with unregistered LoRA MUST raise PermissionError with 0 header reads!
        reads_before = header_reads
        bad_stage = [
            StageSpec(
                name="evil_stage",
                num_inference_steps=2,
                transformer_overrides={"loras": [str(unreg_lora)]},
            )
        ]
        with pytest.raises(PermissionError):
            server.generate("stage attack", stages=bad_stage)
        assert header_reads == reads_before  # 0 safetensors header reads!

        # (3) Async agenerate() concurrency + non-blocking status() with zero absolute path leak
        async_entered = threading.Event()
        async_release = threading.Event()
        call_order: list[str] = []

        def _slow_hook(kw: dict[str, Any]) -> None:
            p = str(kw.get("prompt", ""))
            call_order.append(f"start:{p}")
            if p == "async1":
                async_entered.set()
                async_release.wait(timeout=5.0)
            call_order.append(f"end:{p}")

        pipe.on_call = _slow_hook

        async def _run_async() -> tuple[Any, Any, dict[str, Any]]:
            t1 = asyncio.create_task(server.agenerate("async1"))
            t2 = asyncio.create_task(server.agenerate("async2"))
            await asyncio.to_thread(async_entered.wait, 2.0)
            # Non-blocking status() while async1 is running on worker thread
            st_busy = server.status()
            async_release.set()
            res1, res2 = await asyncio.gather(t1, t2)
            return res1, res2, st_busy

        _, _, st_busy = asyncio.run(_run_async())
        pipe.on_call = None

        assert st_busy["busy"] is True
        assert st_busy["in_flight"] == 1
        assert st_busy["ready"] is True
        assert st_busy["lora_aliases"] == ["style"]
        # Absolute host root directories MUST NEVER leak in status()
        st_json = json.dumps(st_busy)
        assert str(tmp_path) not in st_json
        assert str(lora_root) not in st_json
        # Serialized execution: start1 -> end1 -> start2 -> end2
        assert call_order == ["start:async1", "end:async1", "start:async2", "end:async2"]

        # (4) Concurrent close() race guard: waiter blocked on self._lock MUST fail with RuntimeError
        #     BEFORE entering request_scope when close() completes!
        scope_entries = 0
        orig_scope = serving_pipe_mod.request_scope

        def _spy_scope(*args: Any, **kwargs: Any) -> Any:
            nonlocal scope_entries
            scope_entries += 1
            return orig_scope(*args, **kwargs)

        monkeypatch.setattr(serving_pipe_mod, "request_scope", _spy_scope)

        race_err: list[BaseException] = []
        worker_started = threading.Event()

        def _racing_generate() -> None:
            worker_started.set()
            try:
                server.generate("post_close_race")
            except BaseException as exc:
                race_err.append(exc)

        # Hold reentrant self._lock on main thread -> worker passes unlocked fast-check & blocks on self._lock
        with server.lock:
            t_race = threading.Thread(target=_racing_generate)
            t_race.start()
            assert worker_started.wait(timeout=2.0)
            time.sleep(0.05)  # allow worker to block on self._lock.acquire
            server.close()  # reenters self._lock, marks _closed=True, destroys handles
        t_race.join(timeout=5.0)

        assert len(race_err) == 1 and isinstance(race_err[0], RuntimeError)
        assert "closed" in str(race_err[0]).lower()
        assert scope_entries == 0  # request_scope was NEVER entered on destroyed handle!

        # close() is idempotent & post-close generate() raises RuntimeError
        server.close()
        assert server.closed is True
        assert server.status()["ready"] is False
        assert server.status()["closed"] is True
        with pytest.raises(RuntimeError, match="closed"):
            server.generate("already closed")
    finally:
        server.close()

    # (5) allowed_lora_roots enforcement on StageSpec.transformer_overrides & overrides (0 header reads)
    lib2 = FakeQFLib()
    xfm2 = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib2, torch_dtype=torch.float32)
    server2 = ServingPipeline(
        _WarmRecordPipe(xfm2),
        allowed_lora_roots=[str(lora_root)],
    )
    try:
        reads_before2 = header_reads
        with pytest.raises(PermissionError):
            server2.generate("root escape", overrides={"loras": [str(unreg_lora)]})
        with pytest.raises(PermissionError):
            server2.generate(
                "stage root escape",
                stages=[StageSpec(name="s", transformer_overrides={"loras": [str(unreg_lora)]})],
            )
        assert header_reads == reads_before2
    finally:
        server2.close()

    # (6) ref_audio & av_conditions pass-through and cleanup via ServingPipeline
    ckpt_ltx = _make_dummy_ckpt(
        tmp_path / "ltx25-srv.safetensors",
        [
            "model.diffusion_model.video_embeddings_connector.w",
            "model.diffusion_model.audio_embeddings_connector.w",
        ],
    )
    xfm_ltx = QuantFuncLTX25Transformer3DModel.from_pretrained(ckpt_ltx, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_ltx = ServingPipeline(_WarmRecordPipe(xfm_ltx))
    try:
        srv_ltx.generate("av prompt", ref_audio=torch.randn(1, 4, 128, dtype=torch.float32))
        assert xfm_ltx._ref_audio_state is None
    finally:
        srv_ltx.close()

    ckpt_h3 = _make_dummy_ckpt(tmp_path / "h3-srv.safetensors", ["blocks.0.w"])
    xfm_h3 = QuantFuncMiniMaxH3Transformer3DModel.from_pretrained(ckpt_h3, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_h3 = ServingPipeline(_WarmRecordPipe(xfm_h3))
    try:
        kf = [{"latent": torch.randn(1, 24, 1, 4, 4, dtype=torch.float32), "frame_index": 0}]
        srv_h3.generate("h3 prompt", av_conditions={"keyframes": kf})
        assert xfm_h3._av_conditions is None
    finally:
        srv_h3.close()


# ── Test 47: examples/12_http_fastapi_server.py E2E (stdlib + FastAPI + CLI) ─


def test_http_server_example_endpoints_pos_neg(tmp_path: Path) -> None:
    ex = _load_http_example()

    lora_dir = tmp_path / "loras"
    media_dir = tmp_path / "media"
    outside_dir = tmp_path / "outside"
    media_dir.mkdir(parents=True)
    outside_dir.mkdir(parents=True)

    style_lora = _make_dummy_lora(lora_dir / "style.safetensors", val=1.0)
    ok_media = media_dir / "ok.bin"
    ok_media.write_bytes(b"SAFEBYTES")
    secret_media = outside_dir / "secret.bin"
    secret_media.write_bytes(b"TOPSECRET")

    ckpt = _make_dummy_ckpt(tmp_path / "krea2-http.safetensors")
    lib = FakeQFLib()
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=lib, torch_dtype=torch.float32)
    default_sol = SolConfig(enabled=True, tau=1.25, start_percent=0.0, end_percent=0.9)
    xfm.set_sol(default_sol)

    class _HttpPipe:
        def __init__(self) -> None:
            self.transformer = xfm
            self.scheduler = _DummySched()

        def __call__(self, **kw: Any) -> dict[str, Any]:
            delay = float(kw.get("delay_s", 0.0) or 0.0)
            if delay > 0.0:
                time.sleep(delay)
            return {
                "prompt": kw.get("prompt"),
                "width": kw.get("width", 512),
                "has_image": kw.get("image") is not None,
                "quality_enhance": xfm.quality_enhance,
                "block_cache": xfm.block_cache,
                "sol_enabled": bool(xfm.sol_config and xfm.sol_config.enabled),
                "attention_backend": xfm.attention_backend,
                "active_loras": [(e.path, e.scale) for e in xfm.active_loras],
                "bind_count": len(lib.loras_history),
            }

    server = ServingPipeline(
        _HttpPipe(),
        lora_registry={"style": str(style_lora)},
        allowed_lora_roots=[str(lora_dir)],
        strict_lora_registry=True,
        sticky_loras=True,
        busy_timeout=2.0,
    )

    def _two_stage(payload: dict[str, Any]) -> list[StageSpec]:
        w = int(payload.get("width") or 512)
        return [
            StageSpec(name="draft", width=w, height=w, num_inference_steps=2, output_type="latent"),
            StageSpec(
                name="refine",
                width=w * 2,
                height=w * 2,
                num_inference_steps=2,
                denoise=0.5,
                transformer_overrides={"quality_enhance": True},
                output_type="pil",
            ),
        ]

    # (1) Always-run stdlib ThreadingHTTPServer over real localhost socket
    httpd = ex.build_stdlib_server(
        server,
        host="127.0.0.1",
        port=0,
        stage_builders={"two_stage": _two_stage},
        allowed_media_roots=[str(media_dir)],
        max_media_bytes=256,
    )
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        code, h = _http_json(base_url, "GET", "/health")
        assert code == 200 and h["ready"] is True

        code, st = _http_json(base_url, "GET", "/v1/status")
        assert code == 200 and st["lora_aliases"] == ["style"]
        assert str(tmp_path) not in json.dumps(st)

        # Req 1: full dials + LoRA alias + safe media path
        code, r1 = _http_json(
            base_url,
            "POST",
            "/v1/generate",
            {
                "prompt": "sun",
                "image": {"path": str(ok_media)},
                "loras": [{"name": "style", "scale": 0.8}],
                "fbcache": 0.05,
                "quality_enhance": True,
                "attention_backend": "int8",
            },
        )
        assert code == 200
        assert r1["result"]["bind_count"] == 1
        assert r1["result"]["sol_enabled"] is True  # omitted sol kept pipeline default!
        assert r1["result"]["has_image"] is True

        # Req 2: same LoRA O(1) hit + two-stage + explicit sol=None disables pipeline default Sol
        code, r2 = _http_json(
            base_url,
            "POST",
            "/v1/generate",
            {
                "prompt": "moon",
                "loras": [{"name": "style", "scale": 0.8}],
                "sol": None,
                "mode": "two_stage",
                "return_all_stages": True,
            },
        )
        assert code == 200 and len(r2["result"]) == 2
        assert r2["result"][0]["output"]["bind_count"] == 1  # O(1) hit!
        assert r2["result"][0]["output"]["sol_enabled"] is False  # explicit sol: null disabled Sol!
        assert r2["result"][1]["output"]["width"] == 1024
        assert xfm.sol_config == default_sol  # Restored!

        # Req 3: omit loras -> resets to [] + single-stage
        code, r3 = _http_json(base_url, "POST", "/v1/generate", {"prompt": "stars"})
        assert code == 200
        assert r3["result"]["active_loras"] == []
        assert r3["result"]["sol_enabled"] is True

        # Neg 1: out-of-root / traversal LoRA -> 403
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "loras": ["../../etc/passwd"]})
        assert code == 403

        # Neg 2: unknown key & conflicting fbcache != block_cache -> 400/422
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "unknown_field": 1})
        assert code in (400, 422)
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "fbcache": 0.05, "block_cache": 0.1})
        assert code == 400

        # Neg 3: concurrent busy -> 429
        slow_out: list[tuple[int, dict[str, Any]]] = []
        slow_th = threading.Thread(
            target=lambda: slow_out.append(
                _http_json(base_url, "POST", "/v1/generate", {"prompt": "slow", "extra_kwargs": {"delay_s": 0.35}})
            )
        )
        slow_th.start()
        time.sleep(0.08)
        busy_code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "busy", "busy_timeout": 0.02})
        slow_th.join(timeout=5.0)
        assert busy_code == 429
        assert slow_out and slow_out[0][0] == 200

        # Neg 4: out-of-root media (403), oversized base64 media (400), SSRF URL (400)
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "image": {"path": str(secret_media)}})
        assert code == 403
        big_b64 = base64.b64encode(b"A" * 512).decode("ascii")
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "image": {"base64": big_b64}})
        assert code == 400
        code, _ = _http_json(base_url, "POST", "/v1/generate", {"prompt": "x", "image": "https://example.invalid/a.png"})
        assert code == 400

        # Neg 5: extra_kwargs cannot hijack reserved control kwargs or bypass media sandboxing
        for bad_payload in (
            {"prompt": "x", "extra_kwargs": {"clear_conditions_on_exit": False}},
            {"prompt": "x", "extra_kwargs": {"image": "/etc/passwd"}},
            {"prompt": "x", "extra_kwargs": {"mask_image": "/etc/passwd"}},
            {"prompt": "x", "extra_kwargs": {"overrides": {"quality_enhance": True}}},
            {"prompt": "x", "stages": [{"name": "s1", "extra_kwargs": {"image": "/etc/passwd"}}]},
            {"prompt": "x", "stages": [{"name": "s1", "extra_kwargs": {"clear_conditions_on_exit": False}}]},
        ):
            code, body = _http_json(base_url, "POST", "/v1/generate", bad_payload)
            assert code == 400, f"Expected 400 for {bad_payload!r}, got {code}: {body}"

        # Recovery check after all negative requests
        code, r_ok = _http_json(base_url, "POST", "/v1/generate", {"prompt": "clean_recovery"})
        assert code == 200 and r_ok["ok"] is True

        # (2) Optional FastAPI TestClient path when fastapi + httpx are installed
        if ex.HAS_FASTAPI:
            try:
                from fastapi.testclient import TestClient

                app = ex.build_app(
                    server,
                    stage_builders={"two_stage": _two_stage},
                    allowed_media_roots=[str(media_dir)],
                    max_media_bytes=256,
                )
                client = TestClient(app)
                assert client.get("/health").status_code == 200
                assert client.get("/v1/status").status_code == 200
                # Explicit sol: null disables pipeline default Sol via model_dump(exclude_unset=True)
                resp_null = client.post("/v1/generate", json={"prompt": "fa_null", "sol": None})
                assert resp_null.status_code == 200
                assert resp_null.json()["result"]["sol_enabled"] is False
                # Omitted sol keeps pipeline default Sol
                resp_keep = client.post("/v1/generate", json={"prompt": "fa_keep"})
                assert resp_keep.status_code == 200
                assert resp_keep.json()["result"]["sol_enabled"] is True
                # Forbidden traversal LoRA & unknown key
                assert client.post("/v1/generate", json={"prompt": "x", "loras": ["../../etc/passwd"]}).status_code == 403
                assert client.post("/v1/generate", json={"prompt": "x", "bad_key": 1}).status_code in (400, 422)
            except ImportError:
                pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.close()

    # (3) Subprocess CLI --self-test check (pure CPU)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "12_http_fastapi_server.py"), "--self-test"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "HTTP SELF-TEST PASS" in proc.stdout


# ── Test 48 / Mutation Arm M10: ServingPipeline isolation & strict security ─


def test_mutation_m10_serving_pipeline_isolation_and_security(tmp_path: Path) -> None:
    lora_dir = tmp_path / "m10_loras"
    style_lora = _make_dummy_lora(lora_dir / "style.safetensors", val=1.0)
    unreg_lora = _make_dummy_lora(tmp_path / "m10_unreg.safetensors", val=2.0)
    ckpt = _make_dummy_ckpt(tmp_path / "krea2-m10.safetensors")

    # M10(a): Removing default_loras fallback (setting _default_loras=UNSET) MUST leak Req1 LoRA into Req3 -> RED!
    xfm_a = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_a = ServingPipeline(_WarmRecordPipe(xfm_a), lora_registry={"style": str(style_lora)}, sticky_loras=True)
    try:
        srv_a.generate("req1", overrides={"loras": [{"name": "style", "scale": 0.8}]})
        assert len(xfm_a.active_loras) == 1
        # Mutate: drop omitted-loras reset fallback
        srv_a._default_loras = UNSET  # type: ignore[assignment]
        r3_mut = srv_a.generate("req3_omitted")
        with pytest.raises(AssertionError):
            assert r3_mut["active_loras"] == []

        # Control: restore default_loras=() -> omitted loras clears cleanly to []
        srv_a._default_loras = ()
        r3_ok = srv_a.generate("req3_clean")
        assert r3_ok["active_loras"] == []
        assert xfm_a.active_loras == []
    finally:
        srv_a.close()

    # M10(b): Flipping default strict_lora_registry from True -> False when lora_registry set (no allowed_lora_roots)
    #         MUST allow unregistered absolute path (breaking PermissionError assertion -> RED!)
    xfm_b_mut = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_b_mut = ServingPipeline(
        _WarmRecordPipe(xfm_b_mut),
        lora_registry={"style": str(style_lora)},
        strict_lora_registry=False,  # MUTATION: weakens default strict_lora_registry=True
    )
    try:
        with pytest.raises((AssertionError, pytest.fail.Exception)):
            with pytest.raises(PermissionError):
                srv_b_mut.generate("unreg", overrides={"loras": [str(unreg_lora)]})
    finally:
        srv_b_mut.close()

    # Green control: default strict_lora_registry=None with lora_registry set MUST enforce PermissionError
    xfm_b_ok = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_b_ok = ServingPipeline(
        _WarmRecordPipe(xfm_b_ok),
        lora_registry={"style": str(style_lora)},
    )
    try:
        with pytest.raises(PermissionError):
            srv_b_ok.generate("unreg", overrides={"loras": [str(unreg_lora)]})
    finally:
        srv_b_ok.close()


# ── Tests 53-54: L4 default res_multistep loaders & online sampler switching ─


def test_finalize_pipeline_load_and_four_loaders_scheduler_defaults(tmp_path: Path) -> None:
    pytest.importorskip("diffusers")
    import warnings

    from diffusers import FlowMatchEulerDiscreteScheduler

    ckpt = _make_dummy_ckpt(tmp_path / "krea2-sched-def.safetensors")
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)
    try:
        # (1) finalize_pipeline_load default="res_multistep" converts BEFORE attach_session_hooks
        pipe_def = SimpleNamespace(
            transformer=xfm,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
        )
        finalize_pipeline_load(pipe_def)
        assert isinstance(pipe_def.scheduler, FlowMatchResMultistepScheduler)
        assert pipe_def.scheduler.sampler_mode == "res_multistep"
        assert pipe_def.scheduler.shift == pytest.approx(3.0)
        # Session hooks MUST be attached to the newly converted FlowMatchResMultistepScheduler!
        assert "set_timesteps" in pipe_def.scheduler.__dict__
        assert "step" in pipe_def.scheduler.__dict__

        # (2) scheduler="keep" and scheduler=None keep FlowMatchEulerDiscreteScheduler untouched
        for keep_val in ("keep", None):
            orig = FlowMatchEulerDiscreteScheduler(shift=2.0)
            p_keep = SimpleNamespace(transformer=xfm, scheduler=orig)
            finalize_pipeline_load(p_keep, scheduler=keep_val, attach_hooks=False)
            assert p_keep.scheduler is orig
            assert not isinstance(p_keep.scheduler, FlowMatchResMultistepScheduler)

        # (3) scheduler="euler" sets sampler_mode="euler" on FlowMatchResMultistepScheduler,
        #     and leaves FlowMatchEulerDiscreteScheduler untouched
        res_sched = FlowMatchResMultistepScheduler(shift=2.0)
        p_res_to_euler = SimpleNamespace(transformer=xfm, scheduler=res_sched)
        finalize_pipeline_load(p_res_to_euler, scheduler="euler", attach_hooks=False)
        assert p_res_to_euler.scheduler is res_sched
        assert res_sched.sampler_mode == "euler"

        orig_euler = FlowMatchEulerDiscreteScheduler(shift=2.0)
        p_euler = SimpleNamespace(transformer=xfm, scheduler=orig_euler)
        finalize_pipeline_load(p_euler, scheduler="euler", attach_hooks=False)
        assert p_euler.scheduler is orig_euler

        # (4) Non-FlowMatch dummy scheduler left untouched under default scheduler="res_multistep"
        dummy_sched = _DummySched()
        p_dummy = SimpleNamespace(transformer=xfm, scheduler=dummy_sched)
        finalize_pipeline_load(p_dummy, attach_hooks=False)
        assert p_dummy.scheduler is dummy_sched

        # (5) Invalid scheduler value rejected fail-closed
        for bad_sched in ("dpmpp", "", 123, True):
            with pytest.raises(ValueError, match="Unsupported scheduler"):
                finalize_pipeline_load(
                    SimpleNamespace(transformer=xfm, scheduler=FlowMatchEulerDiscreteScheduler()),
                    scheduler=bad_sched,  # type: ignore[arg-type]
                )

        # (6) All 4 model pipeline loaders default to FlowMatchResMultistepScheduler & respect scheduler=
        class _FakeFlowPipe:
            @classmethod
            def from_pretrained(cls, model_id: str, transformer: Any, **kw: Any) -> Any:
                del model_id, kw
                return SimpleNamespace(
                    transformer=transformer,
                    scheduler=FlowMatchEulerDiscreteScheduler(shift=2.5, stochastic_sampling=True),
                )

        for loader in (
            load_krea2_pipeline,
            load_qwenimage21_pipeline,
            load_ltx25_pipeline,
            load_minimax_h3_pipeline,
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_loaded = loader("org/model", transformer=xfm, pipeline_cls=_FakeFlowPipe)
            assert isinstance(p_loaded.scheduler, FlowMatchResMultistepScheduler)
            assert p_loaded.scheduler.shift == pytest.approx(2.5)
            assert p_loaded.scheduler.sampler_mode == "res_multistep"
            assert "set_timesteps" in p_loaded.scheduler.__dict__
            if loader is load_ltx25_pipeline:
                assert p_loaded.scheduler.config.stochastic_sampling is False

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_kept = loader("org/model", transformer=xfm, pipeline_cls=_FakeFlowPipe, scheduler="keep")
            assert not isinstance(p_kept.scheduler, FlowMatchResMultistepScheduler)
    finally:
        xfm.close()


def test_serving_pipeline_and_http_online_sampler_switching_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("diffusers")

    ckpt = _make_dummy_ckpt(tmp_path / "krea2-sampler-srv.safetensors")
    xfm = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)

    class _SamplerPipe:
        def __init__(self) -> None:
            self.transformer = xfm
            self.scheduler = FlowMatchResMultistepScheduler()
            self.scheduler.set_timesteps(4)
            self.raise_on_call: bool = False

        def __call__(self, **kw: Any) -> dict[str, Any]:
            # Populate multistep history during request to verify cleanup on exception
            self.scheduler.set_timesteps(4)
            self.scheduler.step(
                torch.ones(1, 2, 4, 4),
                self.scheduler.timesteps[0],
                torch.ones(1, 2, 4, 4),
                return_dict=False,
            )
            if self.raise_on_call:
                raise RuntimeError("gen_failed_mid_loop")
            return {
                "prompt": kw.get("prompt"),
                "sampler_mode": self.scheduler.sampler_mode,
            }

    pipe = _SamplerPipe()
    server = ServingPipeline(pipe, busy_timeout=1.0)
    try:
        # (1) Default mode is res_multistep; status() reports it
        assert server.status()["sampler_mode"] == "res_multistep"
        r_def = server.generate("default")
        assert r_def["sampler_mode"] == "res_multistep"
        assert pipe.scheduler.sampler_mode == "res_multistep"
        assert pipe.scheduler._old_denoised is None

        # (2) Per-request sampler="euler" via top-level arg & via overrides={"sampler": "euler"} + atomic restore
        r_euler = server.generate("fast", sampler="euler")
        assert r_euler["sampler_mode"] == "euler"
        assert pipe.scheduler.sampler_mode == "res_multistep"

        r_ov = server.generate("fast_ov", overrides={"sampler": "euler"})
        assert r_ov["sampler_mode"] == "euler"
        assert pipe.scheduler.sampler_mode == "res_multistep"

        # Async agenerate switching & restore
        r_async = asyncio.run(server.agenerate("async_euler", sampler="euler"))
        assert r_async["sampler_mode"] == "euler"
        assert pipe.scheduler.sampler_mode == "res_multistep"

        # (3) Exception rollback: sampler_mode restored to "res_multistep" AND history cleared!
        pipe.raise_on_call = True
        with pytest.raises(RuntimeError, match="gen_failed_mid_loop"):
            server.generate("boom", sampler="euler")
        pipe.raise_on_call = False
        assert pipe.scheduler.sampler_mode == "res_multistep"
        assert pipe.scheduler._old_denoised is None

        # (4) MultiStageRunner per-stage sampler override & restoration
        stage_modes: list[str] = []

        def _stage_rec(p: Any, _s: StageSpec, _b: Any, kw: dict[str, Any]) -> Any:
            stage_modes.append(p.scheduler.sampler_mode)
            return p(**kw)

        stages = [
            StageSpec(name="s1", num_inference_steps=2, output_type="latent", stage_call_adapter=_stage_rec),
            StageSpec(
                name="s2",
                num_inference_steps=2,
                transformer_overrides={"sampler": "euler"},
                stage_call_adapter=_stage_rec,
            ),
        ]
        server.generate("staged", stages=stages)
        assert stage_modes == ["res_multistep", "euler"]
        assert pipe.scheduler.sampler_mode == "res_multistep"

        # (5) Pre-lock validation: invalid sampler & conflicting sampler raise ValueError BEFORE acquiring lock
        held = threading.Event()
        release = threading.Event()

        def _hold() -> None:
            with server.lock:
                held.set()
                release.wait(timeout=5.0)

        t_hold = threading.Thread(target=_hold)
        t_hold.start()
        assert held.wait(timeout=2.0)
        try:
            with pytest.raises(ValueError, match="sampler"):
                server.generate("bad", sampler="dpmpp", busy_timeout=0.0)
            with pytest.raises(ValueError, match="Conflicting sampler"):
                server.generate(
                    "conflict",
                    sampler="euler",
                    overrides={"sampler": "res_multistep"},
                    busy_timeout=0.0,
                )
        finally:
            release.set()
            t_hold.join(timeout=5.0)

        # (6) HTTP controller + stdlib + FastAPI sampler wiring (pos & neg)
        ex = _load_http_example()
        httpd = ex.build_stdlib_server(server, host="127.0.0.1", port=0)
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            code, r1 = _http_json(base_url, "POST", "/v1/generate", {"prompt": "h1"})
            assert code == 200 and r1["result"]["sampler_mode"] == "res_multistep"

            code, r2 = _http_json(base_url, "POST", "/v1/generate", {"prompt": "h2", "sampler": "euler"})
            assert code == 200 and r2["result"]["sampler_mode"] == "euler"
            assert pipe.scheduler.sampler_mode == "res_multistep"

            code, r3 = _http_json(base_url, "POST", "/v1/generate", {"prompt": "h3", "sampler": None})
            assert code == 200 and r3["result"]["sampler_mode"] == "res_multistep"

            code, r4 = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "h4", "overrides": {"sampler": "euler"}},
            )
            assert code == 200 and r4["result"]["sampler_mode"] == "euler"

            # Negative controls: invalid sampler, conflicting sampler, extra_kwargs smuggling -> 400
            for bad_body in (
                {"prompt": "bad", "sampler": "bogus"},
                {"prompt": "bad", "sampler": "euler", "overrides": {"sampler": "res_multistep"}},
                {"prompt": "bad", "extra_kwargs": {"sampler": "euler"}},
            ):
                code, _ = _http_json(base_url, "POST", "/v1/generate", bad_body)
                assert code == 400, f"Expected 400 for {bad_body!r}, got {code}"

            if ex.HAS_FASTAPI:
                try:
                    from fastapi.testclient import TestClient

                    client = TestClient(ex.build_app(server))
                    assert client.post("/v1/generate", json={"prompt": "fa", "sampler": "euler"}).json()["result"]["sampler_mode"] == "euler"
                    assert pipe.scheduler.sampler_mode == "res_multistep"
                    assert client.post("/v1/generate", json={"prompt": "fa", "sampler": "bogus"}).status_code == 422
                except ImportError:
                    pass
        finally:
            httpd.shutdown()
            httpd.server_close()

        # (7) Mutation arm M11: skipping restore_runtime_overrides on exception MUST leak sampler_mode="euler" -> RED!
        pipe.raise_on_call = True
        with monkeypatch.context() as mp:
            mp.setattr(acc_serving_mod, "restore_runtime_overrides", lambda _saved: None)
            with pytest.raises(RuntimeError, match="gen_failed_mid_loop"):
                server.generate("mut_leak", sampler="euler")
            with pytest.raises(AssertionError):
                assert pipe.scheduler.sampler_mode == "res_multistep"
        # Restore baseline & verify green control
        pipe.scheduler.sampler_mode = "res_multistep"
        with pytest.raises(RuntimeError, match="gen_failed_mid_loop"):
            server.generate("ctrl_ok", sampler="euler")
        pipe.raise_on_call = False
        assert pipe.scheduler.sampler_mode == "res_multistep"
    finally:
        server.close()

    # (8) Scheduler without sampler_mode (plain _DummySched) handles sampler="euler" gracefully
    xfm2 = QuantFuncKrea2Transformer2DModel.from_pretrained(ckpt, lib=FakeQFLib(), torch_dtype=torch.float32)
    srv_plain = ServingPipeline(_WarmRecordPipe(xfm2))
    try:
        assert srv_plain.status()["sampler_mode"] is None
        res_plain = srv_plain.generate("plain_sched", sampler="euler")
        assert res_plain["kwargs"]["prompt"] == "plain_sched"
    finally:
        srv_plain.close()

