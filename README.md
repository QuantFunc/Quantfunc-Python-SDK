# QuantFunc Python SDK

Drop-in, layered Python acceleration runtime for image, video, and audio-video diffusion pipelines built on `torch` and `diffusers`.

- **Pure-Python wheel (`py3-none-any`)**: only requires `torch>=2.4`; optional extras for `diffusers`, `transformers`, HuggingFace Hub, and ModelScope.
- **1–3 line drop-in**: replace a pipeline's `transformer` with a `QuantFunc*Transformer` and keep the rest of your `diffusers` workflow intact.
- **Zero-boilerplate session management**: automatic Tier-1 scheduler hooks (`attach_session_hooks`) and Tier-2 fallback tracking, without mutating `pipe.__class__`.
- **Automatic native engine lifecycle**: zero-config install on first load, non-blocking background update checks with offline fallback, SHA-256 verification, and atomic rollback.
- **Model-agnostic multi-stage sampling**: generic `MultiStageRunner` for 2-stage and N-stage workflows, plus thin presets for Krea-2, Qwen-Image-2.1, LTX-2.5 AV, and MiniMax-H3 AV.

---

## Quickstart

### 1. Image T2I (Krea-2)

```python
import torch
from diffusers import Krea2Pipeline
from quantfunc import QuantFuncKrea2Transformer2DModel, attach_session_hooks, enable_offload

transformer = QuantFuncKrea2Transformer2DModel.from_pretrained(
    "/path/to/krea2-quantfunc.safetensors",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/krea-2",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
attach_session_hooks(pipe)
enable_offload(pipe, mode="model")

image = pipe("a cinematic mountain sunrise", num_inference_steps=8).images[0]
image.save("krea2.png")
```

### 2. Audio-Video Generation (LTX-2.5 AV)

```python
import torch
from diffusers import LTX2Pipeline
from quantfunc import (
    QuantFuncLTX25Transformer3DModel,
    SolConfig,
    apply_cache_on_pipe,
    apply_sol_on_pipe,
    ensure_deterministic_scheduler,
)

transformer = QuantFuncLTX25Transformer3DModel.from_pretrained(
    "/path/to/ltx25-quantfunc.safetensors",
    torch_dtype=torch.bfloat16,
)
pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2.5",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
ensure_deterministic_scheduler(pipe)
apply_cache_on_pipe(pipe, block_cache=0.05, step_cache=0.12)
apply_sol_on_pipe(pipe, SolConfig(enabled=True, tau=1.3, start_percent=0.0, end_percent=0.9))

out = pipe("ocean waves crashing at dusk", num_frames=121, num_inference_steps=20)
```

### 3. Automatic Architecture Discovery

```python
from quantfunc import AutoQuantFuncTransformer

transformer = AutoQuantFuncTransformer.from_pretrained("/path/to/checkpoint.safetensors")
```

---

## Installation & Authentication

```bash
# Core package + diffusers ecosystem
pip install "quantfunc[diffusers]"

# Or install all optional integrations (diffusers, transformers, hf, modelscope)
pip install "quantfunc[all]"

# Optional: pre-download native engine binaries into ~/.cache/quantfunc/engine
quantfunc install-engine

# Store your QuantFunc API key safely (interactive prompt or --stdin; never passed in argv)
quantfunc auth set-key
# Or via stdin:
# printf '%s\n' "$QUANTFUNC_API_KEY" | quantfunc auth set-key --stdin

# Verify environment and configuration
quantfunc doctor
```

API keys are resolved in strict precedence order: explicit `api_key=` argument > `QUANTFUNC_API_KEY` environment variable > `~/.config/quantfunc/config.json` (written atomically with POSIX `0o600` permissions).

---

## Native Engine Install & Update Lifecycle

The SDK aligns its native engine installation and update lifecycle with `ComfyUI-QuantFunc`:

- **User-writable cache**: engine binaries are stored in `~/.cache/quantfunc/engine` (or `$QF_ENGINE_DIR`), never inside `site-packages`.
- **Zero-config auto-install**: bare `import quantfunc` performs zero network I/O. On first native engine load (`load_lib(..., auto_install=True)`), missing engine binaries are downloaded and verified automatically.
- **Non-blocking background updates & offline fallback**: when binaries are already installed, `load_lib(..., check_updates=True)` checks remote release metadata asynchronously without blocking inference and falls back cleanly when offline.
- **Integrity & rollback**: every download validates HTTPS URLs, SHA-256 digests, ELF/PE host isolation, and toolchain compatibility, staging via atomic temporary files with rollback on failure and `.reinstalled` self-healing on recovery.
- **Manual checks & opt-out**:
  - Check or apply updates explicitly with `quantfunc update-engine --check`, `quantfunc update-engine`, or `quantfunc doctor --check-update`.
  - Disable automatic update checks with `QF_AUTO_UPDATE=0` or by placing a `.dev_lib_lock` marker inside the engine cache directory.

---

## Layered Architecture (`L0 -> L4`)

The SDK enforces a strict unidirectional 5-layer dependency DAG verified by AST tests:

```text
L4: quantfunc.pipelines / quantfunc.cli        Model loaders, two-stage presets, TE hooks, CLI
           |
           v
L3: quantfunc.accelerate / quantfunc.sampling  Instance-level pipe hooks, model-agnostic N-stage runner
           |
           v
L2: quantfunc.models                           Drop-in nn.Module transformers, AutoQuantFuncTransformer
           |
           v
L1: quantfunc.core                             SolConfig, EngineConfig, Sessions, LoRA, Presets, Registry
           |
           v
L0: quantfunc.native                           Frozen C-API bindings, secure installer/updater, RAII handle
```

| Layer | Package | Responsibility |
|---|---|---|
| **L0** | `quantfunc.native` | 25-symbol C-API, typed error hierarchy, HTTPS/SHA-256 installer & updater, RAII engine handle |
| **L1** | `quantfunc.core` | Model-agnostic `SolConfig`, `EngineConfig`, `RuntimeOverrides`, `AutoSessionTracker`, pure-stdlib LoRA converter & O(1) cache, preset metadata, `@register_model` |
| **L2** | `quantfunc.models` | `QuantFuncBaseTransformer` (meta/offload safe, per-request state reset) and Krea-2, Qwen-Image-2.1, LTX-2.5, MiniMax-H3 adapters |
| **L3** | `quantfunc.accelerate`, `quantfunc.sampling` | Instance-level scheduler hooks (`pipe.__class__` untouched), `request_scope` serving lock & override engine, generic `MultiStageRunner` |
| **L4** | `quantfunc.pipelines`, `quantfunc.cli` | Model-specific pipeline loaders, two-stage factories, model-agnostic `ServingPipeline`, Qwen3-VL pre-norm helper, `quantfunc` CLI |

---

## Recipes

### Instant Hot-Swappable LoRA

```python
# Automatic Kohya / ai-toolkit / PEFT / BFL conversion with caching (LyCORIS/DoRA/OFT fail closed)
transformer.load_lora("/path/to/lightning.safetensors", scale=1.0)

# Compose multiple LoRAs without rebuilding the model
transformer.set_loras([
    ("/path/to/style.safetensors", 0.7),
    ("/path/to/detail.safetensors", 0.4),
])

# Clear active LoRAs instantaneously
transformer.unload_loras()
```

### Sol Sparse Attention, Block Cache (FBCache), Step Cache & Offload

```python
from quantfunc import SolConfig, apply_cache_on_pipe, apply_sol_on_pipe, enable_offload

apply_cache_on_pipe(pipe, block_cache=0.05, step_cache=0.12)
apply_sol_on_pipe(
    pipe,
    SolConfig(
        enabled=True,
        tau=1.3,
        start_percent=0.0,
        end_percent=0.9,
        min_sequence_length=4096,
    ),
)
enable_offload(pipe, mode="low_vram")  # "model" or "low_vram" (unloads VRAM at end of loop)
```

### Generic Multi-Stage Sampling

```python
from quantfunc.sampling import (
    FlowMatchRenoise,
    InterpolateLatentUpscaler,
    MultiStageRunner,
    StageSpec,
)

runner = MultiStageRunner([
    StageSpec(name="base", width=1024, height=1024, num_inference_steps=8, output_type="latent"),
    StageSpec(
        name="refine",
        width=1536,
        height=1536,
        num_inference_steps=8,
        denoise=0.55,
        upscaler=InterpolateLatentUpscaler(scale_by=1.5, mode="bilinear", canvas_multiple=16),
        renoise=FlowMatchRenoise(),
        transformer_overrides={"quality_enhance": True},
        output_type="pil",
    ),
])
result = runner.run(pipe, prompt="a botanical greenhouse at golden hour")
```

### Online HTTP Serving & Per-Request Parameter Switching

Keep a single warm pipeline resident in GPU memory and switch prompts, multimodal conditions (`image`, `video`, `audio`, `mask`, `ref_audio`, `av_conditions`, `ref_latents`, `image_slots`), acceleration dials (`sol`, `loras`, `quality_enhance`, `fbcache`/`block_cache`, `stepcache`/`step_cache`, `attention_backend`, `default_steps`), and single-stage vs multi-stage workflows per request with zero base-weight reloads:

```python
from quantfunc import ServingPipeline

server = ServingPipeline(
    pipe,
    lora_registry={"style": "/srv/loras/style.safetensors"},
    allowed_lora_roots=["/srv/loras"],
    sticky_loras=True,   # O(1) zero-overhead LoRA reuse across consecutive identical requests
    busy_timeout=30.0,   # raises EngineBusyError when overloaded instead of corrupting state
)

# Request 1: single-stage with LoRA alias, block cache, Sol, and quality_enhance
out1 = server.generate(
    "alpine sunrise over misty peaks",
    overrides={
        "loras": [{"name": "style", "scale": 0.8}],
        "fbcache": 0.05,
        "sol": {"enabled": True, "tau": 1.3},
        "quality_enhance": True,
        "attention_backend": "int8",
    },
    width=1024,
    height=1024,
    num_inference_steps=8,
)

# Request 2: same LoRA stack hits O(1) cache (zero header reads / zero native rebind) + switches to two-stage
out2 = server.generate(
    "ocean waves at dusk",
    overrides={"loras": [{"name": "style", "scale": 0.8}]},
    stages=two_stage_specs,
)

# Request 3: omitting loras resets to default_loras (empty by default, preventing cross-request leakage)
out3 = server.generate("minimal desert dunes", overrides={"sol": None})

# Async serving & non-blocking health/status inspection (never exposes host root directories)
out4 = await server.agenerate("starry observatory")
snapshot = server.status()
server.close()
```

For lower-level control around an existing `diffusers` call, use `with request_scope(pipe, overrides, ...):`.

**Deployment model**: run one worker process per GPU with a single warm `ServingPipeline` instance (concurrent threads and `asyncio` tasks are safely serialized by the built-in per-pipeline reentrant lock). Scale across multiple GPUs horizontally by launching one worker process per `CUDA_VISIBLE_DEVICES` device.

Run the pure-CPU HTTP serving self-test (works with or without optional `fastapi`/`uvicorn`):

```bash
python3 examples/12_http_fastapi_server.py --self-test
```

### CLI Reference

| Command | Purpose |
|---|---|
| `quantfunc doctor [--check-update] [--json]` | Inspect SDK, PyTorch, engine cache, auth status, and optional remote update status |
| `quantfunc install-engine [--force] [--device-idx N]` | Download and verify native engine binaries into the user cache |
| `quantfunc update-engine [--check] [--force] [--device N]` | Check for or apply native engine updates |
| `quantfunc convert-lora --in SRC --out DST [--model BASE]` | Convert Kohya/PEFT/BFL LoRA checkpoints to QuantFunc runtime format |
| `quantfunc estimate-vram WEIGHTS [--vram-budget-mb N]` | Estimate model and working-set VRAM requirements |
| `quantfunc auth set-key [--stdin]` / `quantfunc auth status` | Securely store or inspect API key configuration (never accepts raw key in argv) |

---

## Examples

All 12 runnable example scripts live under [`examples/`](examples/) and read credentials from `QUANTFUNC_API_KEY` or `~/.config/quantfunc/config.json`:

| Example | Workflow & Features |
|---|---|
| [`examples/krea2_t2i.py`](examples/krea2_t2i.py) | Krea-2 single-stage T2I drop-in with `attach_session_hooks`, offload, attention backend, and optional Sol |
| [`examples/krea2_t2i_two_stage.py`](examples/krea2_t2i_two_stage.py) | Krea-2 1.5x two-stage sampling via `make_krea2_two_stage_stages` + `MultiStageRunner` |
| [`examples/qwen_image21_t2i.py`](examples/qwen_image21_t2i.py) | Qwen-Image-2.1 6-step T2I with Lightning LoRA hot-load, `quality_enhance`, and `encode_qwen3_vl_pre_norm` |
| [`examples/qwen_image21_transparent.py`](examples/qwen_image21_transparent.py) | Qwen-Image-2.1 RGBA generation preserving the alpha channel to PNG |
| [`examples/qwen_image21_edit.py`](examples/qwen_image21_edit.py) | Qwen-Image-2.1 single- and multi-reference image editing with `image_slots` |
| [`examples/ltx25_t2v_two_stage.py`](examples/ltx25_t2v_two_stage.py) | LTX-2.5 AV two-stage generation with `LTX2NeuralLatentUpscaler`, cache, Sol, and frozen audio restore |
| [`examples/minimax_h3_fl2va.py`](examples/minimax_h3_fl2va.py) | MiniMax-H3 first/last keyframe audio-video generation (FL2VA) |
| [`examples/minimax_h3_fl2va_two_stage.py`](examples/minimax_h3_fl2va_two_stage.py) | MiniMax-H3 FL2VA two-stage sampling with `split_sigmas`, `video_only_upscaler`, and conditioning reuse |
| [`examples/minimax_h3_ref2va.py`](examples/minimax_h3_ref2va.py) | MiniMax-H3 multimodal reference audio-video generation (Ref2VA) |
| [`examples/minimax_h3_ref2va_two_stage.py`](examples/minimax_h3_ref2va_two_stage.py) | MiniMax-H3 Ref2VA two-stage high-resolution refinement |
| [`examples/custom_multi_stage.py`](examples/custom_multi_stage.py) | Custom 3-stage workflow (`draft -> mid -> high-res refine`) with a custom `LatentUpscaler` and per-stage overrides |
| [`examples/12_http_fastapi_server.py`](examples/12_http_fastapi_server.py) | Online HTTP / FastAPI server (with stdlib fallback & pure-CPU `--self-test`) demonstrating per-request dial/LoRA/stage switching and LFI/SSRF protection |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
