"""Dynamic library loader, 25-symbol C-API binder, seamless auto-updater hook, Linux dlinfo CUDA guard, and Windows DLL directory registration."""

from __future__ import annotations

import ctypes
import glob
import os
import re
import sys
import threading
from typing import Any

from .capi import _CAPI_SIGNATURES, QFDeviceCaps
from .errors import console_safe, raise_for_status
from .installer import (
    _BIN_SUBDIR,
    _ENGINE_KERNEL_RE,
    _ENV_KERNEL_PLUGIN_DIR,
    _IS_WINDOWS,
    _auto_update_enabled,
    _engine_load_failed,
    _engine_load_ok,
    _engine_local_choice,
    _engine_platform,
    _select_kernel_plugin,
    _torch_cuda_major,
    assert_toolchain_compatible,
    elf_needed,
    has_ready_so,
    install_engine,
    resolve_so_path,
    start_engine_install,
)

_LIB: Any = None
_LIB_PATH: str | None = None
_LIB_LOCK = threading.Lock()
_LOG_LEVEL: int | None = None
_LOG_STDERR: int | None = None
_DLL_DIR_HANDLES: list[Any] = []

_ENV_LOG_LEVEL = "QF_LOG_LEVEL"
_DEFAULT_LOG_LEVEL = 3
_RTLD_DI_LINKMAP = 2
_CUDA_LIB_RE = re.compile(
    r"lib(?:cudart|cublas|cublasLt|cusolver|cusolverMg|cusparse|cufft|cufftw|curand|nvrtc|nvJitLink|cudnn\w*)\.so\.\d+"
)


class _LinkMap(ctypes.Structure):
    _fields_ = [("l_addr", ctypes.c_void_p), ("l_name", ctypes.c_char_p)]


def library_identity(lib: Any) -> tuple[str, Any]:
    """Return process-unique identity of a loaded library image (not just its filename)."""
    handle = getattr(lib, "_handle", None)
    return ("dso", int(handle)) if handle is not None else ("library", lib)


def loaded_so_path() -> str | None:
    """Return the loaded native host library path, or ``None`` before ``load_lib``."""
    return _LIB_PATH


def set_log_level(level: int) -> None:
    """Set process-wide engine log verbosity without triggering dynamic library loading."""
    global _LOG_LEVEL
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError(f"log level must be an integer, got {level!r}")
    _LOG_LEVEL = int(level)
    if _LIB is not None:
        _LIB.qf_set_log_level(_LOG_LEVEL)


def get_log_level() -> int:
    """Return active engine log level without triggering library loading."""
    if _LIB is not None:
        return int(_LIB.qf_get_log_level())
    if _LOG_LEVEL is not None:
        return int(_LOG_LEVEL)
    env_val = os.environ.get(_ENV_LOG_LEVEL, "").strip()
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return _DEFAULT_LOG_LEVEL


def set_log_stderr(level: int) -> None:
    """Configure stderr log mirroring without triggering library loading."""
    global _LOG_STDERR
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError(f"stderr log level must be an integer, got {level!r}")
    _LOG_STDERR = int(level)
    if _LIB is not None:
        _LIB.qf_set_log_stderr(_LOG_STDERR)


def last_err(lib: Any) -> str:
    """Return the engine's last error string, sanitized for the active console."""
    raw = lib.qf_last_error()
    if isinstance(raw, bytes):
        return console_safe(raw.decode("utf-8", "replace")) if raw else "(none)"
    return console_safe(str(raw)) if raw else "(none)"


def _bind(lib: Any) -> Any:
    """Bind exact 25 frozen C-API symbols; fail-closed if any symbol is absent."""
    missing = [name for name, _, _ in _CAPI_SIGNATURES if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(
            "QuantFunc native library ABI mismatch: missing required C-API symbol(s): "
            + ", ".join(missing[:8])
        )
    for name, restype, argtypes in _CAPI_SIGNATURES:
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


def _linker_path(soname: str) -> str | None:
    if _BIN_SUBDIR != "linux":
        return None
    try:
        lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        dlinfo = ctypes.CDLL(None).dlinfo
    except (OSError, AttributeError):
        return None
    dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dlinfo.restype = ctypes.c_int
    lm = ctypes.POINTER(_LinkMap)()
    if dlinfo(lib._handle, _RTLD_DI_LINKMAP, ctypes.byref(lm)) != 0 or not lm or not lm.contents.l_name:
        return None
    return os.fsdecode(lm.contents.l_name)


def _sidecar_preloads(so_path: str) -> list[str]:
    d = os.path.dirname(so_path)
    try:
        present = set(os.listdir(d))
    except OSError:
        return []
    never = set((_engine_platform() or {}).get("hosts", {}).values()) | {os.path.basename(so_path)}
    want: set[str] = set()
    todo = [os.path.basename(so_path)]
    while todo:
        for n in elf_needed(os.path.join(d, todo.pop())):
            if n in present and n not in want and n not in never and not _ENGINE_KERNEL_RE.fullmatch(n):
                want.add(n)
                todo.append(n)
    return sorted(want)


def _engine_cuda_needs(so_path: str) -> set[str]:
    seeds = [so_path]
    plugin = _select_kernel_plugin(so_path)
    if plugin is not None:
        seeds.append(plugin)
    seen = {os.path.realpath(p) for p in seeds}
    todo = list(seeds)
    cuda: set[str] = set()
    while todo:
        cur = todo.pop()
        cur_dir = os.path.dirname(cur)
        for n in elf_needed(cur):
            if _CUDA_LIB_RE.fullmatch(n):
                cuda.add(n)
            else:
                cand = os.path.join(cur_dir, n)
                rp = os.path.realpath(cand)
                if rp not in seen and os.path.isfile(cand):
                    seen.add(rp)
                    todo.append(cand)
    return cuda


def _torch_cuda_plan(needs: set[str], linker, major: int | None):
    if major is None:
        return set(), set(), []
    anchors = {
        os.path.dirname(p)
        for p in (linker(f"libcudart.so.{major}"), linker(f"libcublas.so.{major}"))
        if p
    }
    dirs = set(anchors)
    parents = [os.path.dirname(os.path.dirname(a)) for a in anchors if os.path.basename(a) == "lib"]
    for parent in {p for p in parents if parents.count(p) > 1}:
        dirs.update(os.path.realpath(d) for d in glob.glob(os.path.join(parent, "*", "lib")))
    provided: set[str] = set()
    preloads: list[str] = []
    for soname in sorted(needs):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) in dirs:
            provided.add(soname)
            continue
        path = next(
            (os.path.join(d, soname) for d in sorted(dirs) if os.path.exists(os.path.join(d, soname))),
            None,
        )
        if path:
            provided.add(soname)
            if not bound:
                preloads.append(path)
            dirs.add(os.path.dirname(os.path.realpath(path)))
    return dirs, provided, preloads


def _assert_torch_cuda_family(provided: set[str], dirs: set[str], linker) -> None:
    """Refuse loading if the process dynamic linker resolves a CUDA soname outside PyTorch's CUDA tree."""
    for soname in sorted(provided):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) not in dirs:
            raise RuntimeError(
                f"quantfunc: refusing engine — it links {soname}, and this process resolves it to "
                f"{os.path.realpath(bound)}, outside PyTorch's CUDA library set ({', '.join(sorted(dirs))}). "
                "One process must use a single CUDA/cuBLAS library family."
            )


def _register_windows_dll_dirs(so_path: str) -> None:
    """Register DLL search directories on Windows and retain handles for process lifetime."""
    add_dll_dir = getattr(os, "add_dll_directory", None)
    if not (_IS_WINDOWS or os.name == "nt") or add_dll_dir is None:
        return
    dirs = [os.path.dirname(os.path.abspath(so_path))]
    env_plugin_dir = os.environ.get(_ENV_KERNEL_PLUGIN_DIR, "").strip()
    if env_plugin_dir and os.path.isdir(env_plugin_dir):
        dirs.append(os.path.abspath(env_plugin_dir))
    for d in dirs:
        if d:
            _DLL_DIR_HANDLES.append(add_dll_dir(d))


def load_lib(
    device_id: int | None = None,
    so_path: str | os.PathLike[str] | None = None,
    auto_install: bool = True,
    check_updates: bool = True,
) -> Any:
    """Load and bind the 25-symbol QuantFunc native host library once per process.

    - Never runs on bare ``import quantfunc``.
    - If no intact pair is installed and ``auto_install=True``, installs synchronously.
    - If an intact pair is already installed, loads immediately with zero blocking latency
      and (when ``check_updates=True`` and ``QF_AUTO_UPDATE != 0``) stages background updates
      for the next launch.
    """
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        was_installed = False
        explicit_or_local = (so_path is not None) or (_engine_local_choice() is not None)
        if so_path is not None:
            resolved = os.fspath(so_path)
            if not os.path.isfile(resolved):
                raise RuntimeError(f"quantfunc: explicit so_path does not exist: {resolved!r}")
            resolved = os.path.realpath(resolved)
        else:
            if not explicit_or_local and has_ready_so(device_id=device_id):
                was_installed = True
                resolved = resolve_so_path(device_id=device_id)
            else:
                try:
                    resolved = resolve_so_path(device_id=device_id, trigger_self_heal=not auto_install)
                except RuntimeError:
                    if not auto_install or explicit_or_local:
                        raise
                    install_engine(device_idx=device_id)
                    resolved = resolve_so_path(device_id=device_id, trigger_self_heal=False)

        assert_toolchain_compatible(resolved)
        _register_windows_dll_dirs(resolved)

        cuda_dirs: set[str] = set()
        cuda_provided: set[str] = set()
        if not (_IS_WINDOWS or os.name == "nt"):
            torch_major = _torch_cuda_major() if "torch" in sys.modules else None
            cuda_dirs, cuda_provided, cuda_preloads = _torch_cuda_plan(
                _engine_cuda_needs(resolved),
                _linker_path,
                torch_major,
            )
            for p in cuda_preloads:
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
                except OSError as exc:
                    raise RuntimeError(f"quantfunc: could not preload PyTorch CUDA library {p}: {exc}") from exc
            so_dir = os.path.dirname(resolved)
            pending = [f for f in _sidecar_preloads(resolved) if f not in cuda_provided]
            for _ in range(max(1, len(pending))):
                still = []
                for f in pending:
                    try:
                        ctypes.CDLL(os.path.join(so_dir, f), mode=ctypes.RTLD_LOCAL)
                    except OSError:
                        still.append(f)
                if not still:
                    break
                pending = still

        cdll_kwargs = {} if (_IS_WINDOWS or os.name == "nt") else {"mode": ctypes.RTLD_GLOBAL}
        try:
            lib = ctypes.CDLL(resolved, **cdll_kwargs)
        except OSError as exc:
            msg = _engine_load_failed(resolved, exc)
            raise RuntimeError(f"quantfunc: {msg}") from exc
        _engine_load_ok(resolved)

        if not (_IS_WINDOWS or os.name == "nt"):
            _assert_torch_cuda_family(cuda_provided, cuda_dirs, _linker_path)

        bound = _bind(lib)
        _LIB_PATH = resolved
        _LIB = bound

        eff_level = _LOG_LEVEL
        if eff_level is None:
            env_lvl = os.environ.get(_ENV_LOG_LEVEL, "").strip()
            if env_lvl:
                try:
                    eff_level = int(env_lvl)
                except ValueError:
                    eff_level = None
        if eff_level is not None:
            _LIB.qf_set_log_level(int(eff_level))
        if _LOG_STDERR is not None:
            _LIB.qf_set_log_stderr(int(_LOG_STDERR))

        if was_installed and not explicit_or_local and check_updates and _auto_update_enabled():
            start_engine_install(device_idx=device_id)

        return _LIB


def query_device_caps(lib: Any, device_id: int = 0) -> QFDeviceCaps:
    """Query hardware capabilities via ``qf_device_query_caps``."""
    caps = QFDeviceCaps(struct_size=ctypes.sizeof(QFDeviceCaps))
    st = lib.qf_device_query_caps(int(device_id), ctypes.byref(caps))
    raise_for_status(st, context="qf_device_query_caps", detail=last_err(lib))
    return caps


__all__ = [
    "load_lib",
    "loaded_so_path",
    "library_identity",
    "set_log_level",
    "get_log_level",
    "set_log_stderr",
    "last_err",
    "query_device_caps",
    "_bind",
    "_linker_path",
    "_torch_cuda_plan",
    "_assert_torch_cuda_family",
    "_DLL_DIR_HANDLES",
]
