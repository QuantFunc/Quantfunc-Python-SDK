"""Online HTTP / FastAPI serving example with per-request parameter switching.

Demonstrates how to keep a single warm pipeline resident in GPU memory via
``ServingPipeline`` and dynamically switch per-request conditioning inputs
(``prompt``, ``negative_prompt``, ``image``, ``video``, ``audio``, ``mask``,
``ref_audio``, ``av_conditions``, ``ref_latents``, ``image_slots``), engine
runtime dials (``sol``, ``loras`` via logical registry aliases,
``quality_enhance``, ``fbcache``/``block_cache``, ``stepcache``/``step_cache``,
``attention_backend``, ``default_steps``, ``sampler`` switching between default
``res_multistep`` aligned with ComfyUI workflows and ``euler``), and
single-stage vs two-stage workflows without reloading base weights.

FastAPI and Uvicorn are strictly optional; when absent, the script falls back
to a pure-stdlib ``ThreadingHTTPServer``.

Deployment model:
    Run one worker process per GPU with a single warm ``ServingPipeline``
    instance (concurrent threads / ``asyncio`` tasks are serialized safely by
    the per-pipeline lock). Scale across multiple GPUs by launching one worker
    process per ``CUDA_VISIBLE_DEVICES`` device.

Pure-CPU self-test (zero model weights, zero GPU, works with or without FastAPI):
    CUDA_VISIBLE_DEVICES="" python3 examples/12_http_fastapi_server.py --self-test
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import stat
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
if (_REPO_ROOT / "quantfunc").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quantfunc import (  # noqa: E402
    UNSET,
    EngineBusyError,
    FlowMatchResMultistepScheduler,
    InvalidArgError,
    LoRAEntry,
    LoRAManager,
    MultiStageRunner,
    OutOfMemoryError,
    RuntimeOverrides,
    ServingPipeline,
    SolConfig,
    StageSpec,
    UnsupportedError,
)
from quantfunc.accelerate.hooks import attach_session_hooks  # noqa: E402
from quantfunc.sampling.runner import StageResult  # noqa: E402

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    BaseModel = None  # type: ignore[assignment,misc]
    HAS_FASTAPI = False


if HAS_FASTAPI and BaseModel is not None:

    class GenerateRequest(BaseModel):  # type: ignore[valid-type,misc]
        """Pydantic request schema preserving omitted vs explicit null via ``exclude_unset=True``."""

        prompt: str | None = None
        negative_prompt: str | None = None
        width: int | None = None
        height: int | None = None
        num_frames: int | None = None
        num_inference_steps: int | None = None
        guidance_scale: float | None = None
        seed: int | None = None
        output_type: str | None = None
        extra_kwargs: dict[str, Any] | None = None
        image: Any = None
        video: Any = None
        audio: Any = None
        mask: Any = None
        ref_audio: Any = None
        av_conditions: dict[str, Any] | None = None
        ref_latents: list[Any] | None = None
        image_slots: list[int] | None = None
        overrides: dict[str, Any] | None = None
        quality_enhance: Any = None
        block_cache: Any = None
        fbcache: Any = None
        step_cache: Any = None
        stepcache: Any = None
        sol: Any = None
        attention_backend: Any = None
        loras: Any = None
        default_steps: Any = None
        sampler: Literal["res_multistep", "euler"] | None = None
        mode: str | None = "single"
        stages: list[dict[str, Any]] | None = None
        return_all_stages: bool = False
        busy_timeout: float | None = None
        sticky_loras: bool | None = None

        if hasattr(BaseModel, "model_dump"):
            model_config = {"extra": "forbid"}
        else:  # pragma: no cover - pydantic v1 fallback

            class Config:
                extra = "forbid"

    if hasattr(GenerateRequest, "model_rebuild"):
        GenerateRequest.model_rebuild(_types_namespace=globals())
else:  # pragma: no cover - optional dependency
    GenerateRequest = Any  # type: ignore[assignment,misc]


DEFAULT_MAX_MEDIA_BYTES: int = 16 * 1024 * 1024  # 16 MiB
DEFAULT_MAX_HTTP_BODY_BYTES: int = 32 * 1024 * 1024  # 32 MiB

_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "quality_enhance",
        "block_cache",
        "fbcache",
        "step_cache",
        "stepcache",
        "sol",
        "attention_backend",
        "loras",
        "default_steps",
    }
)

_MEDIA_KEYS: frozenset[str] = frozenset({"image", "video", "audio", "mask"})

_ALLOWED_GENERATE_KEYS: frozenset[str] = (
    frozenset(
        {
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "output_type",
            "extra_kwargs",
            "ref_audio",
            "av_conditions",
            "ref_latents",
            "image_slots",
            "overrides",
            "sampler",
            "mode",
            "stages",
            "return_all_stages",
            "busy_timeout",
            "sticky_loras",
        }
    )
    | _MEDIA_KEYS
    | _OVERRIDE_KEYS
)

_ALLOWED_STAGE_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "width",
        "height",
        "num_frames",
        "num_inference_steps",
        "guidance_scale",
        "denoise",
        "sigmas",
        "output_type",
        "transformer_overrides",
        "extra_kwargs",
    }
)

# Reserved control & sandboxed media keys that untrusted HTTP extra_kwargs
# (both top-level and per-stage) must NEVER be allowed to inject or shadow.
_FORBIDDEN_EXTRA_KWARGS: frozenset[str] = (
    frozenset(
        {
            "clear_conditions_on_exit",
            "overrides",
            "sampler",
            "stages",
            "return_all_stages",
            "ref_audio",
            "av_conditions",
            "busy_timeout",
            "sticky_loras",
            "prompt",
            "mask_image",
            "ref_images",
            "control_image",
        }
    )
    | _MEDIA_KEYS
    | _OVERRIDE_KEYS
)


def _validate_extra_kwargs(extra: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    validated: dict[str, Any] = {}
    for key, value in extra.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings, got {key!r}")
        if key in _FORBIDDEN_EXTRA_KWARGS:
            raise ValueError(
                f"{field_name} cannot contain reserved control/media key {key!r}; "
                "pass it through the top-level validated request field instead."
            )
        validated[key] = value
    return validated

_BLOCKED_URL_PREFIXES: tuple[str, ...] = (
    "http://",
    "https://",
    "file://",
    "ftp://",
    "ftps://",
    "gopher://",
    "ws://",
    "wss://",
)


def _decode_base64_bounded(b64_text: str, *, max_bytes: int, field_name: str) -> bytes:
    cleaned = "".join(b64_text.split())
    max_b64_len = ((max_bytes + 2) // 3) * 4 + 4
    if len(cleaned) > max_b64_len:
        raise ValueError(f"{field_name} base64 payload exceeds max_media_bytes ({max_bytes} bytes).")
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} contains invalid base64 data.") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"{field_name} decoded payload ({len(raw)} bytes) exceeds max_media_bytes ({max_bytes}).")
    return raw


def _read_sandboxed_media_file(
    raw_path: str,
    *,
    allowed_media_roots: Sequence[str | os.PathLike[str]] | None,
    max_bytes: int,
    field_name: str,
) -> bytes:
    if "\x00" in raw_path or not raw_path.strip():
        raise ValueError(f"{field_name} path must be a non-empty string without NUL bytes.")
    lower = raw_path.strip().lower()
    if any(lower.startswith(prefix) for prefix in _BLOCKED_URL_PREFIXES) or "://" in lower:
        raise ValueError(f"{field_name} remote URL fetching is disabled (SSRF protection).")

    if not allowed_media_roots:
        raise PermissionError(
            f"{field_name} local file access is denied because allowed_media_roots is not configured."
        )

    real_path = os.path.realpath(os.path.expanduser(raw_path))
    contained = False
    for root in allowed_media_roots:
        root_real = os.path.realpath(os.path.expanduser(os.fspath(root)))
        if real_path == root_real:
            continue
        try:
            if os.path.commonpath([root_real, real_path]) == root_real:
                contained = True
                break
        except ValueError:
            continue
    if not contained:
        raise PermissionError(f"{field_name} path is outside allowed_media_roots.")

    try:
        st = os.stat(real_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{field_name} file not found.") from exc
    if not stat.S_ISREG(st.st_mode):
        raise FileNotFoundError(f"{field_name} path is not a regular file.")
    if st.st_size > max_bytes:
        raise ValueError(f"{field_name} file size ({st.st_size} bytes) exceeds max_media_bytes ({max_bytes}).")

    with open(real_path, "rb") as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{field_name} file exceeded max_media_bytes ({max_bytes}) during read.")
    return data


def decode_media_input(
    value: Any,
    *,
    field_name: str,
    allowed_media_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
) -> bytes | None:
    """Decode bounded base64/data-URI or sandboxed local file media with zero URL SSRF."""
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) > max_media_bytes:
            raise ValueError(f"{field_name} exceeds max_media_bytes ({max_media_bytes}).")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} cannot be an empty string.")
        if stripped.lower().startswith("data:"):
            marker = ";base64,"
            idx = stripped.lower().find(marker)
            if idx < 0:
                raise ValueError(f"{field_name} data URI must use ';base64,' encoding.")
            b64_part = stripped[idx + len(marker) :]
            return _decode_base64_bounded(b64_part, max_bytes=max_media_bytes, field_name=field_name)
        return _read_sandboxed_media_file(
            stripped,
            allowed_media_roots=allowed_media_roots,
            max_bytes=max_media_bytes,
            field_name=field_name,
        )
    if isinstance(value, Mapping):
        allowed_keys = {"base64", "data_uri", "path"}
        unknown = sorted(set(value.keys()) - allowed_keys)
        if unknown:
            raise ValueError(f"{field_name} mapping has unknown key(s): {unknown}")
        present = [k for k in ("base64", "data_uri", "path") if k in value and value[k] is not None]
        if len(present) != 1:
            raise ValueError(f"{field_name} mapping must specify exactly one of 'base64', 'data_uri', or 'path'.")
        kind = present[0]
        raw_val = value[kind]
        if not isinstance(raw_val, str):
            raise ValueError(f"{field_name}[{kind!r}] must be a string.")
        if kind == "base64":
            return _decode_base64_bounded(raw_val, max_bytes=max_media_bytes, field_name=field_name)
        if kind == "data_uri":
            return decode_media_input(
                raw_val,
                field_name=field_name,
                allowed_media_roots=allowed_media_roots,
                max_media_bytes=max_media_bytes,
            )
        return _read_sandboxed_media_file(
            raw_val,
            allowed_media_roots=allowed_media_roots,
            max_bytes=max_media_bytes,
            field_name=field_name,
        )
    raise ValueError(f"Unsupported {field_name} input type: {type(value).__name__}")


def _build_stage_specs_from_payload(raw_stages: Any) -> list[StageSpec]:
    if isinstance(raw_stages, str | bytes) or not isinstance(raw_stages, Sequence) or not raw_stages:
        raise ValueError("stages must be a non-empty list of stage specification objects.")
    specs: list[StageSpec] = []
    for idx, item in enumerate(raw_stages):
        if isinstance(item, StageSpec):
            specs.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"stages[{idx}] must be an object, got {type(item).__name__}")
        unknown = sorted(k for k in item.keys() if k not in _ALLOWED_STAGE_KEYS)
        if unknown:
            raise ValueError(f"stages[{idx}] has unknown key(s): {unknown}")
        stage_kw = dict(item)
        if "extra_kwargs" in stage_kw and stage_kw["extra_kwargs"] is not None:
            stage_kw["extra_kwargs"] = _validate_extra_kwargs(
                stage_kw["extra_kwargs"],
                field_name=f"stages[{idx}].extra_kwargs",
            )
        if "transformer_overrides" in stage_kw and stage_kw["transformer_overrides"] is not None:
            stage_kw["transformer_overrides"] = RuntimeOverrides.from_any(stage_kw["transformer_overrides"])
        specs.append(StageSpec(**stage_kw))
    return specs


def _serialize_output(output: Any) -> Any:
    if output is None or isinstance(output, bool | int | float | str):
        return output
    if isinstance(output, bytes):
        return {"bytes_len": len(output)}
    if isinstance(output, StageResult):
        return {
            "name": output.name,
            "stage_index": output.stage_index,
            "output": _serialize_output(output.output),
        }
    if isinstance(output, Mapping):
        return {str(k): _serialize_output(v) for k, v in output.items()}
    if isinstance(output, list | tuple):
        return [_serialize_output(v) for v in output]
    if hasattr(output, "shape") and hasattr(output, "dtype"):
        return {"shape": [int(d) for d in output.shape], "dtype": str(output.dtype)}
    if hasattr(output, "size") and hasattr(output, "mode") and callable(getattr(output, "save", None)):
        return {"image_size": list(output.size), "mode": str(output.mode)}
    if hasattr(output, "__dict__"):
        return {
            str(k): _serialize_output(v)
            for k, v in output.__dict__.items()
            if not str(k).startswith("_")
        }
    return str(output)


def map_exception_to_http(exc: BaseException) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Deterministic SDK exception -> (HTTP status, JSON body, extra headers) mapping."""
    if isinstance(exc, PermissionError):
        return 403, {"error": "forbidden", "detail": str(exc)}, {}
    if isinstance(exc, FileNotFoundError):
        return 404, {"error": "not_found", "detail": str(exc)}, {}
    if isinstance(exc, EngineBusyError):
        return 429, {"error": "engine_busy", "detail": str(exc)}, {"Retry-After": "1"}
    if isinstance(exc, OutOfMemoryError):
        return 503, {"error": "out_of_memory", "detail": str(exc)}, {}
    if isinstance(exc, UnsupportedError):
        return 400, {"error": "unsupported", "detail": str(exc)}, {}
    if isinstance(exc, ValueError | InvalidArgError | TypeError):
        return 400, {"error": "invalid_request", "detail": str(exc)}, {}
    if isinstance(exc, RuntimeError):
        return 503, {"error": "service_unavailable", "detail": str(exc)}, {}
    return 500, {"error": "internal_error", "detail": type(exc).__name__}, {}


class ServingHTTPController:
    """Shared HTTP request dispatcher used by both FastAPI and stdlib HTTP backends."""

    def __init__(
        self,
        server: ServingPipeline,
        *,
        stage_builders: Mapping[str, Callable[[Mapping[str, Any]], MultiStageRunner | Sequence[StageSpec]]] | None = None,
        allowed_media_roots: Sequence[str | os.PathLike[str]] | None = None,
        max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
    ) -> None:
        self.server = server
        self.stage_builders: dict[str, Callable[[Mapping[str, Any]], MultiStageRunner | Sequence[StageSpec]]] = (
            dict(stage_builders) if stage_builders is not None else {}
        )
        self.allowed_media_roots = tuple(allowed_media_roots) if allowed_media_roots is not None else None
        self.max_media_bytes = int(max_media_bytes)

    def health(self) -> tuple[int, dict[str, Any], dict[str, str]]:
        st = self.server.status()
        code = 200 if st["ready"] else 503
        return code, {"status": "ok" if st["ready"] else "closed", "ready": st["ready"], "closed": st["closed"]}, {}

    def status(self) -> tuple[int, dict[str, Any], dict[str, str]]:
        return 200, self.server.status(), {}

    def _prepare_generate_args(self, payload: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError("Request body must be a JSON object.")
        unknown = sorted(k for k in payload.keys() if k not in _ALLOWED_GENERATE_KEYS)
        if unknown:
            raise ValueError(f"Unknown request field(s): {unknown}")

        # Merge nested overrides with optional flat override fields (fail-closed on conflict)
        merged_overrides: dict[str, Any] = {}
        if "overrides" in payload and payload["overrides"] is not None:
            raw_ov = payload["overrides"]
            if not isinstance(raw_ov, Mapping):
                raise ValueError("overrides must be a JSON object or null.")
            merged_overrides.update(raw_ov)

        for key in _OVERRIDE_KEYS:
            if key in payload:
                val = payload[key]
                if key in merged_overrides and (
                    isinstance(merged_overrides[key], bool) != isinstance(val, bool)
                    or merged_overrides[key] != val
                ):
                    raise ValueError(f"Conflicting values for {key!r} in top-level request and overrides.")
                merged_overrides[key] = val

        parsed_overrides = RuntimeOverrides.from_dict(merged_overrides) if merged_overrides else RuntimeOverrides()

        # Decode bounded/sandboxed multimodal inputs (zero URL SSRF)
        pipe_kwargs: dict[str, Any] = {}
        if "extra_kwargs" in payload and payload["extra_kwargs"] is not None:
            pipe_kwargs.update(_validate_extra_kwargs(payload["extra_kwargs"], field_name="extra_kwargs"))

        for key in (
            "negative_prompt",
            "width",
            "height",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "output_type",
            "ref_latents",
            "image_slots",
        ):
            if key in payload and payload[key] is not None:
                pipe_kwargs[key] = payload[key]

        for media_key in _MEDIA_KEYS:
            if media_key in payload and payload[media_key] is not None:
                pipe_kwargs[media_key] = decode_media_input(
                    payload[media_key],
                    field_name=media_key,
                    allowed_media_roots=self.allowed_media_roots,
                    max_media_bytes=self.max_media_bytes,
                )

        # Resolve single-stage vs multi-stage mode
        stages_arg: MultiStageRunner | Sequence[StageSpec] | None = None
        mode = payload.get("mode", "single")
        if "stages" in payload and payload["stages"] is not None:
            if mode not in ("single", None):
                raise ValueError("Specify either 'mode' or explicit 'stages', not both.")
            stages_arg = _build_stage_specs_from_payload(payload["stages"])
        elif mode is not None and mode != "single":
            if not isinstance(mode, str) or mode not in self.stage_builders:
                raise ValueError(
                    f"Unsupported mode {mode!r}; expected 'single' or one of {sorted(self.stage_builders.keys())}"
                )
            stages_arg = self.stage_builders[mode](payload)

        prompt_arg = payload["prompt"] if "prompt" in payload else UNSET
        sampler_arg = payload.get("sampler")
        if (
            sampler_arg is not None
            and "sampler" in merged_overrides
            and merged_overrides["sampler"] != sampler_arg
        ):
            raise ValueError(
                "Conflicting values for 'sampler' in top-level request and overrides."
            )
        ref_audio_arg = payload["ref_audio"] if "ref_audio" in payload else UNSET
        av_cond_arg = payload["av_conditions"] if "av_conditions" in payload else UNSET
        busy_timeout_arg = payload["busy_timeout"] if "busy_timeout" in payload else UNSET
        sticky_loras_arg = payload.get("sticky_loras")
        return_all_stages = bool(payload.get("return_all_stages", False))

        # Unpack pipe_kwargs BEFORE explicit control kwargs so reserved ServingPipeline
        # controls (including clear_conditions_on_exit=True) can never be shadowed.
        gen_kwargs: dict[str, Any] = {
            **pipe_kwargs,
            "overrides": parsed_overrides,
            "sampler": sampler_arg,
            "stages": stages_arg,
            "return_all_stages": return_all_stages,
            "ref_audio": ref_audio_arg,
            "av_conditions": av_cond_arg,
            "busy_timeout": busy_timeout_arg,
            "sticky_loras": sticky_loras_arg,
            "clear_conditions_on_exit": True,
        }
        return prompt_arg, gen_kwargs

    def generate(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        try:
            prompt_arg, gen_kwargs = self._prepare_generate_args(payload)
            result = self.server.generate(prompt_arg, **gen_kwargs)
            return 200, {"ok": True, "result": _serialize_output(result)}, {}
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            return map_exception_to_http(exc)

    async def agenerate(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        try:
            prompt_arg, gen_kwargs = self._prepare_generate_args(payload)
            result = await self.server.agenerate(prompt_arg, **gen_kwargs)
            return 200, {"ok": True, "result": _serialize_output(result)}, {}
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            return map_exception_to_http(exc)


def build_app(
    server: ServingPipeline,
    *,
    stage_builders: Mapping[str, Callable[[Mapping[str, Any]], MultiStageRunner | Sequence[StageSpec]]] | None = None,
    allowed_media_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
) -> Any:
    """Build an async FastAPI ASGI app backed by ``ServingPipeline`` (requires optional ``fastapi``)."""
    if not HAS_FASTAPI or FastAPI is None or BaseModel is None or JSONResponse is None:
        raise ImportError(
            "FastAPI is not installed. Install optional serving dependencies with "
            "`pip install fastapi uvicorn` or use `build_stdlib_server()`."
        )

    controller = ServingHTTPController(
        server,
        stage_builders=stage_builders,
        allowed_media_roots=allowed_media_roots,
        max_media_bytes=max_media_bytes,
    )

    app = FastAPI(title="QuantFunc Online Serving API", version="0.1.0")

    @app.get("/health")
    async def health_endpoint() -> Any:
        code, body, headers = controller.health()
        return JSONResponse(status_code=code, content=body, headers=headers)

    @app.get("/v1/status")
    async def status_endpoint() -> Any:
        code, body, headers = controller.status()
        return JSONResponse(status_code=code, content=body, headers=headers)

    @app.post("/v1/generate")
    async def generate_endpoint(req: GenerateRequest) -> Any:
        # Preserve omitted vs explicit null (e.g. sol: null) via exclude_unset=True.
        if hasattr(req, "model_dump"):
            payload = req.model_dump(exclude_unset=True)
        else:  # pragma: no cover - pydantic v1 fallback
            payload = req.dict(exclude_unset=True)
        code, body, headers = await controller.agenerate(payload)
        return JSONResponse(status_code=code, content=body, headers=headers)

    return app


def make_stdlib_handler(
    controller: ServingHTTPController,
    *,
    max_body_bytes: int = DEFAULT_MAX_HTTP_BODY_BYTES,
) -> type[BaseHTTPRequestHandler]:
    """Create a pure-stdlib ``BaseHTTPRequestHandler`` subclass backed by ``controller``."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _ = (format, args)

        def _send_json(self, code: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if headers:
                for hk, hv in headers.items():
                    self.send_header(hk, hv)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                code, body, headers = controller.health()
                self._send_json(code, body, headers)
                return
            if path == "/v1/status":
                code, body, headers = controller.status()
                self._send_json(code, body, headers)
                return
            self._send_json(404, {"error": "not_found", "detail": f"Unknown path: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/v1/generate":
                self._send_json(404, {"error": "not_found", "detail": f"Unknown path: {path}"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid_request", "detail": "Invalid Content-Length header."})
                return
            if content_length < 0 or content_length > max_body_bytes:
                self._send_json(
                    413,
                    {"error": "payload_too_large", "detail": f"Body exceeds {max_body_bytes} bytes."},
                )
                return
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
                return
            code, body, headers = controller.generate(parsed)
            self._send_json(code, body, headers)

    return _Handler


def build_stdlib_server(
    server: ServingPipeline,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    stage_builders: Mapping[str, Callable[[Mapping[str, Any]], MultiStageRunner | Sequence[StageSpec]]] | None = None,
    allowed_media_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
) -> ThreadingHTTPServer:
    """Create a multi-threaded stdlib HTTP server backed by ``ServingPipeline``."""
    controller = ServingHTTPController(
        server,
        stage_builders=stage_builders,
        allowed_media_roots=allowed_media_roots,
        max_media_bytes=max_media_bytes,
    )
    handler_cls = make_stdlib_handler(controller)
    return ThreadingHTTPServer((host, int(port)), handler_cls)


# ── Pure-CPU self-test harness (zero weights, zero GPU) ─────────────────────


def _write_dummy_lora_safetensors(path: Path) -> Path:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
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


class _SelfTestHandle:
    """Pure-CPU minimal engine handle tracking LoRA binds for ``--self-test``."""

    def __init__(self) -> None:
        self.loras_history: list[list[dict[str, Any]]] = []
        self.applied_lora_sig: str = "[]"
        self.session_open: bool = False
        self.closed: bool = False

    def set_loras(self, prepared: list[dict[str, Any]]) -> None:
        self.loras_history.append([dict(item) for item in prepared])
        self.applied_lora_sig = json.dumps(prepared, sort_keys=True, separators=(",", ":"))

    def end_session_if_open(self) -> tuple[bool, bool]:
        was = self.session_open
        self.session_open = False
        return (was, True)

    def resident_vram_bytes(self) -> int:
        return 0 if self.closed else 4 * (1 << 30)

    def destroy(self) -> None:
        self.closed = True


class _SelfTestTransformer:
    """Pure-CPU transformer backed by the real ``LoRAManager`` for ``--self-test``."""

    _is_quantfunc_transformer = True
    model_arch = "demo"

    def __init__(self, cache_dir: Path) -> None:
        self._handle = _SelfTestHandle()
        self._lora_manager = LoRAManager(
            self._handle,
            cache_dir=cache_dir,
            on_before_rebind=self.end_session_if_open,
        )
        self.quality_enhance: bool = False
        self.block_cache: float = 0.0
        self.step_cache: float = 0.0
        self.sol_config: SolConfig | None = None
        self.attention_backend: str = "auto"
        self._default_steps: int = 8
        self.reset_calls: int = 0
        self.closed: bool = False

    @property
    def active_loras(self) -> list[LoRAEntry]:
        return list(self._lora_manager.active_loras)

    def configure_lora_policy(
        self,
        *,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool = False,
    ) -> None:
        self._lora_manager.configure_policy(
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
        )

    def set_loras(
        self,
        loras: Sequence[Any],
        *,
        force: bool = False,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool | None = None,
    ) -> None:
        self._lora_manager.set_loras(
            loras,
            force=force,
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
        )

    def unload_loras(self, *, force: bool = False) -> None:
        self._lora_manager.unload_loras(force=force)

    def set_quality_enhance(self, enabled: bool) -> None:
        self.quality_enhance = bool(enabled)

    def set_cache(self, block_cache: float | None = None, step_cache: float | None = None) -> None:
        if block_cache is not None:
            self.block_cache = float(block_cache)
        if step_cache is not None:
            self.step_cache = float(step_cache)

    def set_sol(self, sol_config: SolConfig | None) -> None:
        self.sol_config = sol_config

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = str(backend)

    def set_default_steps(self, n: int) -> None:
        self._default_steps = int(n)

    def end_session_if_open(self) -> tuple[bool, bool]:
        return self._handle.end_session_if_open()

    def resident_vram_bytes(self) -> int:
        return self._handle.resident_vram_bytes()

    def reset_request_state(self) -> None:
        self.end_session_if_open()
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True
        self._handle.destroy()


class _SelfTestPipe:
    """Minimal warm pipeline recording active dials and supporting optional slow delay."""

    def __init__(self, transformer: _SelfTestTransformer) -> None:
        self.transformer = transformer
        self.scheduler = FlowMatchResMultistepScheduler()
        self.scheduler.set_timesteps(2)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        delay = float(kwargs.get("delay_s", 0.0) or 0.0)
        if delay > 0.0:
            time.sleep(delay)
        self.transformer._handle.session_open = True
        rec = {
            "prompt": kwargs.get("prompt"),
            "negative_prompt": kwargs.get("negative_prompt"),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "has_image": kwargs.get("image") is not None,
            "has_mask": kwargs.get("mask") is not None,
            "quality_enhance": self.transformer.quality_enhance,
            "block_cache": self.transformer.block_cache,
            "step_cache": self.transformer.step_cache,
            "sol_enabled": bool(self.transformer.sol_config and self.transformer.sol_config.enabled),
            "attention_backend": self.transformer.attention_backend,
            "sampler_mode": getattr(self.scheduler, "sampler_mode", None),
            "active_loras": [(e.path, e.scale) for e in self.transformer.active_loras],
            "lora_bind_count": len(self.transformer._handle.loras_history),
        }
        self.calls.append(rec)
        return rec


def _http_json(
    base_url: str,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return int(err.code), json.loads(err.read().decode("utf-8"))


def run_self_test() -> int:
    """Run a complete pure-CPU positive + negative HTTP self-test (stdlib + optional FastAPI)."""
    with tempfile.TemporaryDirectory(prefix="qf_http_selftest_") as td:
        root = Path(td)
        lora_dir = root / "loras"
        media_dir = root / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        style_lora = _write_dummy_lora_safetensors(lora_dir / "style.safetensors")
        sample_img = media_dir / "sample.bin"
        sample_img.write_bytes(b"PNGDATA")

        xfm = _SelfTestTransformer(cache_dir=root / "cache")
        pipe = _SelfTestPipe(xfm)
        attach_session_hooks(pipe)

        server = ServingPipeline(
            pipe,
            lora_registry={"style": str(style_lora)},
            allowed_lora_roots=[str(lora_dir)],
            strict_lora_registry=True,
            sticky_loras=True,
            busy_timeout=1.0,
        )

        def _build_two_stage(req_payload: Mapping[str, Any]) -> list[StageSpec]:
            w = int(req_payload.get("width") or 512)
            h = int(req_payload.get("height") or 512)
            return [
                StageSpec(name="draft", width=w, height=h, num_inference_steps=4, output_type="latent"),
                StageSpec(
                    name="refine",
                    width=int(w * 1.5),
                    height=int(h * 1.5),
                    num_inference_steps=4,
                    denoise=0.5,
                    transformer_overrides={"quality_enhance": True},
                    output_type="pil",
                ),
            ]

        httpd = build_stdlib_server(
            server,
            host="127.0.0.1",
            port=0,
            stage_builders={"two_stage": _build_two_stage},
            allowed_media_roots=[str(media_dir)],
            max_media_bytes=4096,
        )
        port = httpd.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        srv_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        srv_thread.start()

        try:
            # 1. /health and /v1/status
            code, health = _http_json(base_url, "GET", "/health")
            assert code == 200 and health["ready"] is True, f"health failed: {code} {health}"

            code, st = _http_json(base_url, "GET", "/v1/status")
            assert code == 200 and st["ready"] is True, f"status failed: {code} {st}"
            assert st["lora_aliases"] == ["style"]
            assert st["sampler_mode"] == "res_multistep"
            assert str(root) not in json.dumps(st), "status leaked absolute host directory!"

            # 2. Req 1: single-stage + default res_multistep + style:0.8 + fbcache + sol + quality_enhance + int8 + safe media
            code, r1 = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {
                    "prompt": "alpine sunrise",
                    "negative_prompt": "blurry",
                    "image": {"data_uri": "data:application/octet-stream;base64,UE5H"},
                    "mask": {"path": str(sample_img)},
                    "loras": [{"name": "style", "scale": 0.8}],
                    "fbcache": 0.05,
                    "sol": {"enabled": True, "tau": 1.3},
                    "quality_enhance": True,
                    "attention_backend": "int8",
                },
            )
            assert code == 200, f"Req1 failed: {code} {r1}"
            res1 = r1["result"]
            assert res1["quality_enhance"] is True
            assert abs(res1["block_cache"] - 0.05) < 1e-6
            assert res1["sol_enabled"] is True
            assert res1["attention_backend"] == "int8"
            assert res1["sampler_mode"] == "res_multistep"
            assert res1["has_image"] is True and res1["has_mask"] is True
            assert res1["lora_bind_count"] == 1

            # 3. Req 2: same LoRA (O(1) hit -> 0 extra binds!) + online sampler="euler" + two-stage switching
            code, r2 = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {
                    "prompt": "ocean sunset",
                    "loras": [{"name": "style", "scale": 0.8}],
                    "sampler": "euler",
                    "mode": "two_stage",
                    "return_all_stages": True,
                },
            )
            assert code == 200, f"Req2 failed: {code} {r2}"
            stages_out = r2["result"]
            assert len(stages_out) == 2
            assert stages_out[0]["output"]["lora_bind_count"] == 1  # O(1) hit: 0 extra binds!
            assert stages_out[0]["output"]["sampler_mode"] == "euler"
            assert stages_out[1]["output"]["sampler_mode"] == "euler"
            assert stages_out[1]["output"]["quality_enhance"] is True
            assert stages_out[1]["output"]["width"] == 768

            # 4. Req 3: omit loras & sampler -> resets to [] and restores default res_multistep + single-stage
            code, r3 = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "minimal desert", "mode": "single"},
            )
            assert code == 200, f"Req3 failed: {code} {r3}"
            assert r3["result"]["active_loras"] == []
            assert r3["result"]["quality_enhance"] is False
            assert r3["result"]["sol_enabled"] is False
            assert r3["result"]["sampler_mode"] == "res_multistep"

            # 5. Negative controls: 403 traversal, 400 unknown key, 400 invalid sampler, 400 alias conflict, 429 busy, 400 SSRF
            code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "bad_sampler", "sampler": "invalid_sampler"},
            )
            assert code == 400, f"Expected 400 for invalid sampler, got {code}"
            code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "evil", "loras": ["../../etc/passwd"]},
            )
            assert code == 403, f"Expected 403 for traversal LoRA, got {code}"

            code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "bad", "overrides": {"unknown_dial": 1}},
            )
            assert code in (400, 422), f"Expected 400/422 for unknown dial, got {code}"

            code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "bad", "fbcache": 0.05, "block_cache": 0.1},
            )
            assert code == 400, f"Expected 400 for conflicting alias, got {code}"

            code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "ssrf", "image": "http://169.254.169.254/latest"},
            )
            assert code == 400, f"Expected 400 for SSRF URL, got {code}"

            # Concurrent busy control -> 429
            slow_result: list[tuple[int, dict[str, Any]]] = []
            slow_t = threading.Thread(
                target=lambda: slow_result.append(
                    _http_json(
                        base_url,
                        "POST",
                        "/v1/generate",
                        {"prompt": "slow", "extra_kwargs": {"delay_s": 0.35}},
                    )
                )
            )
            slow_t.start()
            time.sleep(0.08)
            busy_code, _ = _http_json(
                base_url,
                "POST",
                "/v1/generate",
                {"prompt": "fast_fail", "busy_timeout": 0.02},
            )
            slow_t.join(timeout=5.0)
            assert busy_code == 429, f"Expected 429 when pipeline is busy, got {busy_code}"
            assert slow_result and slow_result[0][0] == 200

            # Recovery check: service remains healthy after negative requests
            code, r_ok = _http_json(base_url, "POST", "/v1/generate", {"prompt": "after errors"})
            assert code == 200 and r_ok["ok"] is True

            # Optional FastAPI TestClient verification when fastapi + httpx are available
            fastapi_checked = False
            if HAS_FASTAPI:
                try:
                    from fastapi.testclient import TestClient

                    app = build_app(
                        server,
                        stage_builders={"two_stage": _build_two_stage},
                        allowed_media_roots=[str(media_dir)],
                        max_media_bytes=4096,
                    )
                    client = TestClient(app)
                    assert client.get("/health").status_code == 200
                    assert client.get("/v1/status").status_code == 200
                    resp = client.post("/v1/generate", json={"prompt": "fastapi check", "sol": None, "sampler": "euler"})
                    assert resp.status_code == 200
                    assert resp.json()["result"]["sampler_mode"] == "euler"
                    resp_default = client.post("/v1/generate", json={"prompt": "fastapi default"})
                    assert resp_default.status_code == 200
                    assert resp_default.json()["result"]["sampler_mode"] == "res_multistep"
                    bad_sampler_resp = client.post("/v1/generate", json={"prompt": "bad", "sampler": "unknown"})
                    assert bad_sampler_resp.status_code in (400, 422)
                    bad_resp = client.post("/v1/generate", json={"prompt": "bad", "unknown_top_key": 1})
                    assert bad_resp.status_code in (400, 422)
                    fastapi_checked = True
                except ImportError:
                    pass

            print(
                f"HTTP SELF-TEST PASS (stdlib=OK, fastapi={fastapi_checked}, "
                f"lora_binds={len(xfm._handle.loras_history)}, total_requests={server.status()['total_requests']})"
            )
            return 0
        finally:
            httpd.shutdown()
            httpd.server_close()
            server.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantFunc online HTTP / FastAPI serving example.")
    parser.add_argument("--self-test", action="store_true", help="Run pure-CPU end-to-end self-test and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    print(
        "To serve a live model pipeline, construct your loaded pipeline with QuantFunc, "
        "wrap it in ServingPipeline(pipe, lora_registry=..., allowed_lora_roots=...), "
        "and pass it to build_app(server) (FastAPI + uvicorn) or build_stdlib_server(server). "
        "Run with --self-test for an immediate pure-CPU end-to-end demo.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
