"""Command-line interface for the QuantFunc Python SDK."""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from quantfunc import __version__
from quantfunc.core.config import (
    OFFLOAD_MODES,
    redact_api_key,
    resolve_auth,
    save_api_key,
    validate_api_key,
)
from quantfunc.core.lora import convert_lora_file
from quantfunc.core.registry import list_registered_models
from quantfunc.native.errors import console_safe
from quantfunc.native.handle import estimate_vram, make_create_params
from quantfunc.native.installer import (
    check_engine_update,
    default_engine_dir,
    engine_install_status,
    install_engine,
    resolve_so_path,
)
from quantfunc.native.loader import load_lib

_RAW_KEY_PATTERN = re.compile(r"qf_[0-9a-fA-F]{8,}")
_OPTIONAL_PACKAGES = (
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "modelscope",
)


def _sanitize_text(text: str, raw_secret: str | None = None) -> str:
    cleaned = str(text)
    if raw_secret:
        cleaned = cleaned.replace(raw_secret, redact_api_key(raw_secret))
    cleaned = _RAW_KEY_PATTERN.sub(lambda m: redact_api_key(m.group(0)), cleaned)
    return console_safe(cleaned)


def _serialize_update_info(info: Any) -> dict[str, Any]:
    if is_dataclass(info):
        return asdict(info)
    if hasattr(info, "_asdict"):
        return dict(info._asdict())
    if isinstance(info, dict):
        return dict(info)
    return {
        key: getattr(info, key)
        for key in (
            "installed_version",
            "latest_version",
            "update_available",
            "up_to_date",
            "status",
            "message",
            "error",
            "engine_dir",
            "pair_name",
        )
        if hasattr(info, key)
    }


def _probe_optional_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            versions[pkg] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            versions[pkg] = None
    return versions


def _cmd_doctor(args: argparse.Namespace) -> int:
    torch_version = "unavailable"
    torch_cuda = None
    try:
        torch_mod = importlib.import_module("torch")
        torch_version = str(getattr(torch_mod, "__version__", "unknown"))
        torch_cuda = getattr(getattr(torch_mod, "version", None), "cuda", None)
    except Exception:
        pass

    engine_dir = str(default_engine_dir())
    install_state = engine_install_status()
    auth_status = "unconfigured"
    auth_source = "none"
    auth_preview = None
    try:
        auth = resolve_auth(required=False)
        if auth is not None and getattr(auth, "api_key", None):
            auth_status = "configured"
            auth_source = str(getattr(auth, "source", "configured"))
            auth_preview = redact_api_key(auth.api_key)
    except Exception:
        auth_status = "invalid"

    update_payload: dict[str, Any] | None = None
    if args.check_update:
        try:
            update_payload = _serialize_update_info(check_engine_update())
        except Exception as exc:
            update_payload = {"error": _sanitize_text(str(exc))}

    try:
        importlib.import_module("quantfunc.models")
    except Exception:
        pass

    report: dict[str, Any] = {
        "sdk_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "torch_cuda_runtime": torch_cuda,
        "engine_dir": engine_dir,
        "engine_status": list(install_state) if isinstance(install_state, tuple) else install_state,
        "auth": {
            "status": auth_status,
            "source": auth_source,
            "key": auth_preview,
        },
        "registered_models": sorted(list_registered_models()),
        "optional_packages": _probe_optional_packages(),
    }
    if update_payload is not None:
        report["engine_update"] = update_payload

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"QuantFunc Python SDK {report['sdk_version']}")
        print(f"  Python:        {report['python']} ({report['platform']})")
        print(f"  PyTorch:       {report['torch']} (CUDA runtime: {report['torch_cuda_runtime'] or 'none'})")
        print(f"  Engine dir:    {report['engine_dir']}")
        print(f"  Engine status: {report['engine_status']}")
        print(
            f"  Auth:          {auth_status} (source={auth_source}, key={auth_preview or 'none'})"
        )
        print(f"  Models:        {', '.join(report['registered_models']) or 'none'}")
        if update_payload is not None:
            if "error" in update_payload and update_payload["error"]:
                print(f"  Engine update: error ({update_payload['error']})")
            else:
                inst = update_payload.get("installed_version", "unknown")
                latest = update_payload.get("latest_version", "unknown")
                avail = update_payload.get("update_available", False)
                print(
                    f"  Engine update: installed={inst}, latest={latest}, "
                    f"update_available={avail}"
                )
    return 0


def _resolve_installed_host_path(
    marker: Mapping[str, Any],
    *,
    device_idx: int | None,
    engine_dir: str | None,
) -> str | None:
    version = marker.get("version")
    gpu_set = marker.get("set")
    cuda = marker.get("cuda")
    host = marker.get("host")
    if version and gpu_set and cuda is not None and host:
        plat_sys = platform.system()
        subdir = "windows" if plat_sys == "Windows" else ("darwin" if plat_sys == "Darwin" else "linux")
        base = str(default_engine_dir(engine_dir) if engine_dir is not None else default_engine_dir())
        candidate = os.path.join(base, subdir, f"{version}-{gpu_set}-cu{cuda}", str(host))
        return os.path.realpath(candidate) if os.path.exists(candidate) else candidate
    try:
        return resolve_so_path(
            device_id=device_idx,
            engine_dir=engine_dir,
            trigger_self_heal=False,
        )
    except Exception:
        return str(host) if host else None


def _emit_install_result(
    res: Any,
    *,
    verb: str,
    as_json: bool,
    device_idx: int | None,
    engine_dir: str | None,
    force: bool | None = None,
) -> None:
    if res is None:
        raw_status = engine_install_status()
        if isinstance(raw_status, tuple) and len(raw_status) >= 2:
            state, detail = str(raw_status[0]), str(raw_status[1])
        else:
            state, detail = "skipped", str(raw_status)
        reason = _sanitize_text(detail or state or "local engine override active")
        payload: dict[str, Any] = {
            "status": "skipped",
            "installed": None,
            "version": None,
            "state": state,
            "reason": reason,
        }
        if force is not None:
            payload["force"] = bool(force)
        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Engine {verb} skipped ({state}): {reason}")
        return

    if isinstance(res, Mapping):
        marker = dict(res)
        version = marker.get("version")
        gpu_set = marker.get("set")
        cuda = marker.get("cuda")
        host_path = _resolve_installed_host_path(
            marker,
            device_idx=device_idx,
            engine_dir=engine_dir,
        )
        payload = {
            "status": "installed",
            "installed": marker,
            "version": version,
            "gpu_set": gpu_set,
            "cuda": cuda,
            "installed_path": host_path,
        }
        if force is not None:
            payload["force"] = bool(force)
        if as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            label = "Engine ready" if verb == "install" else "Engine updated"
            parts = []
            if version is not None:
                parts.append(f"version={version}")
            if gpu_set is not None:
                parts.append(f"set={gpu_set}")
            if cuda is not None:
                parts.append(f"cuda={cuda}")
            if host_path is not None:
                parts.append(f"path={host_path}")
            detail_str = ", ".join(parts) if parts else str(marker)
            print(f"{label}: {detail_str}")
        return

    path_str = os.fspath(res) if isinstance(res, str | os.PathLike) else str(res)
    payload = {
        "status": "installed",
        "installed": path_str,
        "version": None,
        "installed_path": path_str,
    }
    if force is not None:
        payload["force"] = bool(force)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        label = "Engine ready at" if verb == "install" else "Engine updated at"
        print(f"{label}: {path_str}")


def _cmd_install_engine(args: argparse.Namespace) -> int:
    try:
        installed = install_engine(
            device_idx=args.device_idx,
            force=args.force,
            engine_dir=args.engine_dir,
            base_url=args.base_url,
        )
    except Exception as exc:
        print(f"Failed to install engine: {_sanitize_text(str(exc))}", file=sys.stderr)
        return 1
    _emit_install_result(
        installed,
        verb="install",
        as_json=bool(getattr(args, "json", False)),
        device_idx=args.device_idx,
        engine_dir=args.engine_dir,
    )
    return 0


def _cmd_update_engine(args: argparse.Namespace) -> int:
    if args.check:
        try:
            info = check_engine_update(
                device_idx=args.device,
                engine_dir=args.engine_dir,
                base_url=args.base_url,
            )
        except Exception as exc:
            print(f"Failed to check engine update: {_sanitize_text(str(exc))}", file=sys.stderr)
            return 1
        payload = _serialize_update_info(info)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            installed = payload.get("installed_version", "none")
            latest = payload.get("latest_version", "unknown")
            avail = payload.get("update_available", False)
            status = "update available" if avail else "up to date"
            print(f"Engine version check: {status} (installed={installed}, latest={latest})")
        return 0

    try:
        installed_res = install_engine(
            device_idx=args.device,
            force=args.force,
            engine_dir=args.engine_dir,
            base_url=args.base_url,
        )
    except Exception as exc:
        print(f"Failed to update engine: {_sanitize_text(str(exc))}", file=sys.stderr)
        return 1

    _emit_install_result(
        installed_res,
        verb="update",
        as_json=bool(args.json),
        device_idx=args.device,
        engine_dir=args.engine_dir,
        force=bool(args.force),
    )
    return 0


def _cmd_convert_lora(args: argparse.Namespace) -> int:
    in_path = Path(args.in_path)
    if not in_path.is_file():
        print(f"Input LoRA file not found: {in_path}", file=sys.stderr)
        return 1
    try:
        result = convert_lora_file(
            str(in_path),
            str(args.out_path),
            model_path=args.model_path,
        )
    except Exception as exc:
        print(f"LoRA conversion failed: {_sanitize_text(str(exc))}", file=sys.stderr)
        return 1
    print(f"Converted LoRA written to: {result or args.out_path}")
    return 0


def _cmd_estimate_vram(args: argparse.Namespace) -> int:
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights path not found: {weights_path}", file=sys.stderr)
        return 1
    try:
        lib = load_lib(device_id=args.device_id, auto_install=False)
        params = make_create_params(
            weights_path=str(weights_path),
            device_id=args.device_id,
            config_json={
                "model_arch": args.arch,
                "offload_mode": args.offload_mode,
                "vram_budget_mb": args.vram_budget_mb,
            },
        )
        est = estimate_vram(lib, params)
    except Exception as exc:
        print(
            f"VRAM estimation failed: {_sanitize_text(str(exc))}. "
            "Run `quantfunc install-engine` if the native engine is not yet installed.",
            file=sys.stderr,
        )
        return 1

    if hasattr(est, "_asdict"):
        data = dict(est._asdict())
    elif is_dataclass(est):
        data = asdict(est)
    elif isinstance(est, dict):
        data = dict(est)
    else:
        data = {
            k: getattr(est, k)
            for k in (
                "model_bytes",
                "resident_bytes",
                "scratch_bytes",
                "total_bytes",
                "effective_model_bytes",
            )
            if hasattr(est, k)
        }

    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"VRAM estimate for {weights_path}:")
        for key, val in sorted(data.items()):
            if isinstance(val, int | float) and not isinstance(val, bool) and (key == "bytes" or key.endswith("_bytes")):
                mb = val / (1024 * 1024)
                print(f"  {key}: {val} bytes ({mb:.1f} MiB)")
            else:
                print(f"  {key}: {val}")
    return 0


def _cmd_auth_set_key(args: argparse.Namespace) -> int:
    raw_key = ""
    try:
        if args.stdin:
            raw_key = sys.stdin.readline().strip()
        else:
            raw_key = getpass.getpass("QuantFunc API key: ").strip()

        validate_api_key(raw_key)
        saved_path = save_api_key(raw_key, keyfile=args.keyfile)
    except Exception as exc:
        print(
            f"Failed to save API key: {_sanitize_text(str(exc), raw_secret=raw_key)}",
            file=sys.stderr,
        )
        return 1

    print(f"Saved API key ({redact_api_key(raw_key)}) to {saved_path}")
    return 0


def _cmd_auth_status(args: argparse.Namespace) -> int:
    keyfile = args.keyfile
    try:
        auth = resolve_auth(keyfile=keyfile, required=False)
    except Exception as exc:
        print(f"Auth status error: {_sanitize_text(str(exc))}", file=sys.stderr)
        return 1

    has_key = bool(auth is not None and getattr(auth, "api_key", None))
    payload = {
        "configured": has_key,
        "source": str(getattr(auth, "source", "none")) if has_key else "none",
        "key": redact_api_key(auth.api_key) if has_key else None,
        "server_url": getattr(auth, "server_url", None) if auth is not None else None,
        "env_var_set": bool(os.environ.get("QUANTFUNC_API_KEY")),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status_text = "configured" if has_key else "not configured"
        print(f"Auth status: {status_text}")
        print(f"  Source: {payload['source']}")
        print(f"  Key:    {payload['key'] or 'none'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``quantfunc`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="quantfunc",
        description="QuantFunc Python SDK command-line utilities.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Inspect SDK environment, native engine status, and authentication.",
    )
    p_doctor.add_argument(
        "--check-update",
        action="store_true",
        default=False,
        help="Query remote release metadata for a newer native engine version.",
    )
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_doctor.set_defaults(func=_cmd_doctor)

    # install-engine
    p_install = subparsers.add_parser(
        "install-engine",
        help="Download and verify native engine binaries into the user cache.",
    )
    p_install.add_argument("--device-idx", type=int, default=None, help="Target CUDA device index.")
    p_install.add_argument("--force", action="store_true", help="Re-download even if already installed.")
    p_install.add_argument("--engine-dir", default=None, help="Override engine cache directory.")
    p_install.add_argument("--base-url", default=None, help="Override HTTPS release base URL.")
    p_install.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_install.set_defaults(func=_cmd_install_engine)

    # update-engine
    p_update = subparsers.add_parser(
        "update-engine",
        help="Check for or apply native engine updates in the user cache.",
    )
    p_update.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Check for available engine updates without installing.",
    )
    p_update.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reinstall of the latest engine release.",
    )
    p_update.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target CUDA device index for engine selection (default: 0).",
    )
    p_update.add_argument("--engine-dir", default=None, help="Override engine cache directory.")
    p_update.add_argument("--base-url", default=None, help="Override HTTPS release base URL.")
    p_update.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_update.set_defaults(func=_cmd_update_engine)

    # convert-lora
    p_lora = subparsers.add_parser(
        "convert-lora",
        help="Convert a LoRA checkpoint into QuantFunc runtime format.",
    )
    p_lora.add_argument("--in", dest="in_path", required=True, help="Input LoRA .safetensors path.")
    p_lora.add_argument("--out", dest="out_path", required=True, help="Output .safetensors path.")
    p_lora.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Optional base model checkpoint path for key mapping.",
    )
    p_lora.set_defaults(func=_cmd_convert_lora)

    # estimate-vram
    p_vram = subparsers.add_parser(
        "estimate-vram",
        help="Estimate VRAM requirements for a QuantFunc checkpoint.",
    )
    p_vram.add_argument("weights", help="Path to checkpoint file or model directory.")
    p_vram.add_argument("--arch", default="auto", help="Model architecture override (default: auto).")
    p_vram.add_argument(
        "--vram-budget-mb",
        type=int,
        default=0,
        help="Optional VRAM budget cap in MiB (0 = automatic).",
    )
    p_vram.add_argument(
        "--offload-mode",
        choices=OFFLOAD_MODES,
        default="auto",
        help="Engine offload mode (default: auto).",
    )
    p_vram.add_argument("--device-id", type=int, default=0, help="Target device ID (default: 0).")
    p_vram.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_vram.set_defaults(func=_cmd_estimate_vram)

    # auth
    p_auth = subparsers.add_parser(
        "auth",
        help="Manage QuantFunc API key storage without exposing secrets in argv.",
    )
    auth_sub = p_auth.add_subparsers(dest="auth_command", required=True)

    p_auth_set = auth_sub.add_parser(
        "set-key",
        help="Store an API key securely from interactive prompt or --stdin.",
    )
    p_auth_set.add_argument(
        "--stdin",
        action="store_true",
        default=False,
        help="Read the API key from standard input instead of interactive prompt.",
    )
    p_auth_set.add_argument(
        "--keyfile",
        default=None,
        help="Optional path to config.json (defaults to ~/.config/quantfunc/config.json).",
    )
    p_auth_set.set_defaults(func=_cmd_auth_set_key)

    p_auth_status = auth_sub.add_parser(
        "status",
        help="Display current authentication status (key is always redacted).",
    )
    p_auth_status.add_argument(
        "--keyfile",
        default=None,
        help="Optional path to config.json.",
    )
    p_auth_status.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_auth_status.set_defaults(func=_cmd_auth_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning an integer process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
