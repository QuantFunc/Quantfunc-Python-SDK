"""Secure engine binary installer, auto-updater, cache resolver, and ELF/PE toolchain validator.

Pure stdlib at import time; lazy ``torch`` import occurs strictly inside CUDA probe helpers.
Engine binaries are installed exclusively into a user-writable cache directory
(``~/.cache/quantfunc/engine`` or ``QF_ENGINE_DIR``) and never into ``site-packages``.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import mmap
import os
import platform
import re
import secrets
import shutil
import stat
import struct
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .errors import EngineNotInstallable, console_safe

ENGINE_COMPAT_VERSION = "0.0.08"

_IS_WINDOWS = platform.system() == "Windows"
_IS_DARWIN = platform.system() == "Darwin"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else ("darwin" if _IS_DARWIN else "linux")
_LIB_BASENAME = (
    "quantfunc.dll" if _IS_WINDOWS else ("libquantfunc.dylib" if _IS_DARWIN else "libquantfunc.so")
)

_ENV_ENGINE_DIR = "QF_ENGINE_DIR"
_ENV_SO_OVERRIDE = "QF_NATIVE_SO_PATH"
_ENV_KERNEL_PLUGIN_DIR = "QF_KERNEL_PLUGIN_DIR"
_ENV_ENGINE_BASE_URL = "QF_ENGINE_BASE_URL"
_ENV_ENGINE_COMPAT_VERSION = "QF_ENGINE_COMPAT_VERSION"
_ENV_ALLOW_UNVERIFIED_TOOLCHAIN = "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN"
_ENV_AUTO_UPDATE = "QF_AUTO_UPDATE"

_DEFAULT_ENGINE_BASE_URL = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master"
_ENGINE_PLATFORMS: dict[str, dict[str, Any]] = {
    "linux": {
        "key": "linux",
        "arches": ("x86_64",),
        "kernel": True,
        "hosts": {13: "libquantfunc.so", 12: "libquantfunc-12.so"},
    },
    "windows": {
        "key": "win32",
        "arches": ("AMD64",),
        "kernel": True,
        "hosts": {13: "quantfunc.dll", 12: "quantfunc-12.dll"},
    },
}

_CONSUMER_SMS = (75, 86, 89, 120)

_ENGINE_KERNEL_RE = re.compile(r"libquantfunc_kernels(?:-12)?\.so")
_ENGINE_KERNEL_WIN_RE = re.compile(r"libquantfunc_kernels(?:-12)?\.dll", re.IGNORECASE)
_ENGINE_ADDON_RE = re.compile(r"[A-Za-z0-9][-A-Za-z0-9_.]{0,63}\.(?:so|dll|dylib)")
_ENGINE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_ENGINE_SET_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ENGINE_SETS_SCHEMA = 2
_ENGINE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ENGINE_PAIR_ID_RE = re.compile(r"[0-9a-f]{32}")
_ENGINE_MARKER_RE = re.compile(r"\.engine-([a-z][a-z0-9_]{0,31})-cu(\d+)\.json")
_ENGINE_PAIR_RE = re.compile(r"(\d+\.\d+\.\d+)-([a-z][a-z0-9_]{0,31})-cu(\d+)")
_ENGINE_STALE_RE = re.compile(r"\..+\.stale-\d+-[0-9a-f]+")
_ENGINE_LOCAL_BUILD_LOCK = ".dev_lib_lock"
_ENGINE_INSTALL_LOCKFILE = ".engine-install.lock"
_ENGINE_REINSTALLED = ".reinstalled"
_ENGINE_VERIFY_SCHEMA_MAX = 1
_ENGINE_HTTP_TIMEOUT_S = 120

# System V ELF constants for pure-stdlib DT_NEEDED parsing.
_SHT_DYNAMIC = 6
_DT_NULL = 0
_DT_NEEDED = 1

_HOST_FORBIDDEN_CUDA_RE = re.compile(r"lib(?:cudart|cublas|cublasLt)\.so\.\d+")
_HOST_FORBIDDEN_CUDA_PE_RE = re.compile(r"(?:cudart|cublas(?:lt)?)64_\d+\.dll", re.IGNORECASE)

_ENGINE_STATUS: dict[str, str] = {"state": "idle", "detail": ""}
_ENGINE_STATUS_LOCK = threading.Lock()
_ENGINE_DEVICE = 0


@dataclass(frozen=True)
class UpdateInfo:
    """Read-only engine update availability report."""

    installed_version: str | None
    latest_version: str | None
    update_available: bool
    gpu_set: str | None = None
    cuda_major: int | None = None
    detail: str = ""


def _engine_status(state: str, detail: str = "") -> None:
    with _ENGINE_STATUS_LOCK:
        _ENGINE_STATUS.update(state=state, detail=detail)


def engine_install_status() -> tuple[str, str]:
    """Return ``(state, detail)``: idle | checking | downloading | installed | offline | unavailable | failed | local."""
    with _ENGINE_STATUS_LOCK:
        return _ENGINE_STATUS["state"], _ENGINE_STATUS["detail"]


def default_engine_dir(engine_dir: str | os.PathLike[str] | None = None) -> str:
    """Resolve user-writable engine cache root (never inside ``site-packages``)."""
    if engine_dir is not None and str(engine_dir).strip():
        return os.path.abspath(os.path.expanduser(os.fspath(engine_dir)))
    env_dir = os.environ.get(_ENV_ENGINE_DIR, "").strip()
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    if os.name == "nt" or _IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA", "").strip() or os.path.expanduser("~")
        return os.path.abspath(os.path.join(base, "quantfunc", "engine"))
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = xdg if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.abspath(os.path.join(base, "quantfunc", "engine"))


def _engine_bin_dir(engine_dir: str | os.PathLike[str] | None = None) -> str:
    return os.path.join(default_engine_dir(engine_dir), _BIN_SUBDIR)


def _engine_platform() -> dict[str, Any] | None:
    return _ENGINE_PLATFORMS.get(_BIN_SUBDIR)


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _engine_compat_version() -> str:
    val = os.environ.get(_ENV_ENGINE_COMPAT_VERSION, "").strip() or ENGINE_COMPAT_VERSION
    if not _ENGINE_VERSION_RE.fullmatch(val):
        raise ValueError(f"Malformed engine compatibility version: {val!r}")
    return val


def _allow_unverified_toolchain() -> bool:
    return os.environ.get(_ENV_ALLOW_UNVERIFIED_TOOLCHAIN, "").strip().lower() in ("1", "true", "yes")


def _auto_update_enabled() -> bool:
    return os.environ.get(_ENV_AUTO_UPDATE, "1").strip().lower() not in ("0", "false", "no", "off")


def _has_local_build_lock(engine_dir: str | os.PathLike[str] | None = None) -> bool:
    root_dir = default_engine_dir(engine_dir)
    bin_dir = _engine_bin_dir(engine_dir)
    return os.path.exists(os.path.join(bin_dir, _ENGINE_LOCAL_BUILD_LOCK)) or os.path.exists(
        os.path.join(root_dir, _ENGINE_LOCAL_BUILD_LOCK)
    )


def _engine_local_choice(engine_dir: str | os.PathLike[str] | None = None) -> str | None:
    if os.environ.get(_ENV_SO_OVERRIDE, "").strip():
        return f"{_ENV_SO_OVERRIDE} names the engine library"
    if _has_local_build_lock(engine_dir):
        return f"{_ENGINE_LOCAL_BUILD_LOCK} keeps the local build"
    return None


# ── Strict HTTPS URL & redirect validation ───────────────────────────────────


def validate_https_url(url: str) -> str:
    """Validate an HTTPS URL strictly: require ``https://`` and hostname; reject userinfo, query, fragment."""
    if not isinstance(url, str) or not url or url != url.strip() or any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError(f"Invalid HTTPS URL: {url!r}")
    if "?" in url or "#" in url:
        raise ValueError(f"Refusing URL with query or fragment: {url!r}")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS URL: {url!r}")
    if not parts.hostname:
        raise ValueError(f"HTTPS URL missing hostname: {url!r}")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError(f"Refusing HTTPS URL containing userinfo: {url!r}")
    if parts.query or parts.fragment:
        raise ValueError(f"Refusing HTTPS URL with query or fragment: {url!r}")
    try:
        _ = parts.port
    except ValueError as exc:
        raise ValueError(f"Invalid port in HTTPS URL: {url!r}") from exc
    return url


def _validate_redirect_url(url: str) -> str:
    """Validate an HTTPS redirect hop (allows signed CDN query strings, refuses downgrade/userinfo/fragment)."""
    if not isinstance(url, str) or not url or any(c.isspace() or ord(c) < 32 for c in url):
        raise RuntimeError(f"Refusing malformed redirect URL: {url!r}")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https" or not parts.hostname:
        raise RuntimeError(f"Refusing redirect to non-HTTPS engine URL: {url}")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise RuntimeError(f"Refusing redirect URL with userinfo: {url}")
    if parts.fragment or "#" in url:
        raise RuntimeError(f"Refusing redirect URL with fragment: {url}")
    return url


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse any redirect hop that downgrades from HTTPS or carries credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        _validate_redirect_url(resolved)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _engine_http_open(url: str):
    validate_https_url(url)
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
    resp = opener.open(url, timeout=_ENGINE_HTTP_TIMEOUT_S)
    final_url = resp.geturl()
    try:
        _validate_redirect_url(final_url)
    except BaseException:
        resp.close()
        raise
    return resp


def _engine_http_get(url: str) -> bytes:
    with _engine_http_open(url) as resp:
        return resp.read()


# ── Cross-platform atomic file operations & permission guards ────────────────


def _atomic_open_flags() -> int:
    return (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )


def _harden_posix_mode(path: str, mode: int = 0o755) -> None:
    """Set POSIX mode on a non-symlink file/dir without an islink->chmod TOCTOU window."""
    if os.name != "posix" or not hasattr(os, "fchmod"):
        return
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fchmod(fd, mode)
    except OSError:
        pass
    finally:
        os.close(fd)


def _refuse_symlink_dir(path: str, label: str) -> None:
    if os.path.islink(path):
        raise RuntimeError(f"quantfunc: refusing symlinked {label} directory: {path}")


def _make_part_path(directory: str, name: str) -> str:
    return os.path.join(directory, f".{name}.part-{os.getpid()}-{secrets.token_hex(8)}")


def _engine_write_file(path: str, data: bytes, *, posix_mode: int = 0o644) -> None:
    """Cross-platform atomic write using O_CREAT|O_EXCL + random suffix + fsync + os.replace."""
    parent = os.path.dirname(os.path.abspath(path))
    tmp = _make_part_path(parent, os.path.basename(path))
    fd = os.open(tmp, _atomic_open_flags(), 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
            if os.name == "posix" and hasattr(os, "fchmod"):
                os.fchmod(fh.fileno(), posix_mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _engine_fetch_to(url: str, dest: str, label: str) -> str:
    """Stream an HTTPS artifact into a newly created exclusive temp file and return its SHA-256."""
    h = hashlib.sha256()
    done = 0
    fd = os.open(dest, _atomic_open_flags(), 0o600)
    try:
        with _engine_http_open(url) as resp, os.fdopen(fd, "wb") as fh:
            fd = -1
            total = int(resp.headers.get("Content-Length") or 0)
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                fh.write(chunk)
                h.update(chunk)
                done += len(chunk)
                _engine_status("downloading", f"{label}: {done >> 20} of {total >> 20 if total else '?'} MB")
            fh.flush()
            os.fsync(fh.fileno())
            if os.name == "posix" and hasattr(os, "fchmod"):
                os.fchmod(fh.fileno(), 0o755)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(dest)
        except OSError:
            pass
        raise
    return h.hexdigest()


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Lazy CUDA / GPU probes (torch imported ONLY inside these helpers) ────────


def _torch_cuda_major() -> int | None:
    try:
        import torch

        return int(str(torch.version.cuda).split(".")[0]) if torch.version.cuda else None
    except Exception:  # noqa: BLE001
        return None


def _driver_cuda_major() -> int | None:
    try:
        cuda = ctypes.CDLL("nvcuda.dll" if _BIN_SUBDIR == "windows" else "libcuda.so.1")
        v = ctypes.c_int(0)
        if cuda.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
        return v.value // 1000
    except Exception:  # noqa: BLE001
        return None


def _gpu_sm(device_idx: int = 0) -> int | None:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(int(device_idx))
        return major * 10 + minor
    except Exception:  # noqa: BLE001
        return None


def engine_choice(device_idx: int = 0) -> tuple[int, int]:
    """Return ``(cuda_major, sm)`` for this machine or raise ``EngineNotInstallable``."""
    plat = _engine_platform()
    if plat is None:
        raise EngineNotInstallable(
            f"QuantFunc engines are published for Linux and Windows only (current platform: {platform.system()})"
        )
    torch_major, driver_major = _torch_cuda_major(), _driver_cuda_major()
    if torch_major is None:
        raise EngineNotInstallable(
            "PyTorch reports no CUDA version (CPU-only PyTorch?); the QuantFunc engine requires a CUDA build of PyTorch"
        )
    hosts = plat["hosts"]
    if torch_major not in hosts:
        raise EngineNotInstallable(
            f"No QuantFunc engine is published for CUDA {torch_major} "
            f"(torch CUDA {torch_major}, driver CUDA {driver_major}); published: CUDA {sorted(hosts)}"
        )
    if driver_major is not None and driver_major < torch_major:
        raise EngineNotInstallable(
            f"NVIDIA driver supports up to CUDA {driver_major}, but torch uses CUDA {torch_major}: update the driver"
        )
    sm = _gpu_sm(device_idx)
    if sm is None:
        raise EngineNotInstallable("No CUDA GPU is visible to PyTorch; no engine was installed")
    if sm not in _CONSUMER_SMS:
        supported = ", ".join(f"{s // 10}.{s % 10}" for s in _CONSUMER_SMS)
        raise EngineNotInstallable(
            f"No QuantFunc PC Engine is published for GPU architecture SM {sm // 10}.{sm % 10}; "
            f"supported consumer GPUs: SM {supported}"
        )
    return torch_major, sm


# ── Pure-stdlib ELF / PE readers & zero-CUDA-host toolchain guard ────────────


def _cstr(data: bytes | mmap.mmap, off: int) -> str:
    z = data.find(b"\x00", off)
    if z < 0:
        raise ValueError("unterminated string table entry")
    return data[off:z].decode("latin-1")


def _mmap_read(path: str, read_fn):
    try:
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return read_fn(data)
    except Exception:  # noqa: BLE001
        return None


def _elf_sections(data: bytes | mmap.mmap):
    if data[:4] != b"\x7fELF":
        return []
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3A)
    else:
        shoff = struct.unpack_from(end + "I", data, 0x20)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2E)
    raw = []
    for i in range(shnum):
        b = shoff + i * shentsize
        if is64:
            raw.append(
                struct.unpack_from(end + "II", data, b)
                + struct.unpack_from(end + "QQ", data, b + 0x18)
                + struct.unpack_from(end + "I", data, b + 0x28)
            )
        else:
            raw.append(
                struct.unpack_from(end + "II", data, b)
                + struct.unpack_from(end + "III", data, b + 0x10)
            )
    names = raw[shstrndx][2] if shstrndx < len(raw) else None
    return [("" if names is None else _cstr(data, names + n), t, o, s, lnk) for (n, t, o, s, lnk) in raw]


def elf_needed(path: str | os.PathLike[str]) -> list[str]:
    """Return ``DT_NEEDED`` sonames of an ELF binary (pure stdlib; ``[]`` on parse error)."""

    def read(data):
        secs = _elf_sections(data)
        dyn = next(((o, s, lnk) for (_, t, o, s, lnk) in secs if t == _SHT_DYNAMIC), None)
        if not dyn or dyn[2] >= len(secs):
            return []
        dyn_off, dyn_sz, dyn_link = dyn
        dynstr_off = secs[dyn_link][2]
        is64 = data[4] == 2
        end = "<" if data[5] == 1 else ">"
        needed = []
        for off in range(dyn_off, dyn_off + dyn_sz, 16 if is64 else 8):
            tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
            if tag == _DT_NULL:
                break
            if tag == _DT_NEEDED:
                needed.append(_cstr(data, dynstr_off + val))
        return needed

    return _mmap_read(os.fspath(path), read) or []


def elf_pair_id(path: str | os.PathLike[str]) -> str | None:
    """Read the 32-hex ``.qf_pair_id`` build pairing tag from an ELF binary, or ``None``."""

    def read(data):
        sec = next(((o, s) for (n, _, o, s, _) in _elf_sections(data) if n == ".qf_pair_id"), None)
        pid = data[sec[0] : sec[0] + sec[1]].split(b"\x00")[0].decode("ascii") if sec else ""
        return pid if _ENGINE_PAIR_ID_RE.fullmatch(pid) else None

    return _mmap_read(os.fspath(path), read)


def pe_imports(path: str | os.PathLike[str]) -> list[str]:
    """Return lowercased DLL names imported by a Windows PE image (import + delay-load tables)."""

    def read(data):
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe : pe + 4] != b"PE\0\0":
            return []
        nsec, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        dirs = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)
        secs = [struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8) for i in range(nsec)]

        def at(rva: int) -> int:
            return next(ptr + rva - va for vsize, va, raw, ptr in secs if va <= rva < va + max(vsize, raw))

        names = []
        for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
            rva = struct.unpack_from("<I", data, dirs + 8 * index)[0]
            if not rva:
                continue
            off = at(rva)
            while True:
                name_rva = struct.unpack_from("<I", data, off + name_at)[0]
                if not name_rva:
                    break
                names.append(_cstr(data, at(name_rva)).lower())
                off += size
        return names

    return _mmap_read(os.fspath(path), read) or []


def _is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _is_pe(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"MZ"
    except OSError:
        return False


def _host_forbidden_cuda(so_path: str) -> list[str]:
    return [n for n in elf_needed(so_path) if _HOST_FORBIDDEN_CUDA_RE.fullmatch(n)]


def _host_forbidden_cuda_pe(dll_path: str) -> list[str]:
    return [n for n in pe_imports(dll_path) if _HOST_FORBIDDEN_CUDA_PE_RE.fullmatch(n)]


def _kernel_regex(is_pe: bool | None = None):
    use_win = (_BIN_SUBDIR == "windows") if is_pe is None else bool(is_pe)
    return _ENGINE_KERNEL_WIN_RE if use_win else _ENGINE_KERNEL_RE


def _kernel_filename(major: int = 13, is_pe: bool | None = None) -> str:
    use_win = (_BIN_SUBDIR == "windows") if is_pe is None else bool(is_pe)
    suffix = "-12" if major == 12 else ""
    ext = "dll" if use_win else "so"
    return f"libquantfunc_kernels{suffix}.{ext}"


def _host_cuda_major(so_path: str, is_pe: bool | None = None) -> int:
    use_win = _is_pe(so_path) if is_pe is None else bool(is_pe)
    stem = os.path.splitext(os.path.basename(so_path))[0]
    return 12 if (stem.lower() if use_win else stem).endswith("-12") else 13


def _so_cuda_major(bin_path: str) -> int | None:
    """Return CUDA major dynamically linked by an ELF kernel plugin or Windows PE DLL, or None."""
    for lib in elf_needed(bin_path):
        m = re.match(r"libcudart\.so\.(\d+)", lib)
        if m:
            return int(m.group(1))
    imports = pe_imports(bin_path)
    for pat in (r"cudart64_(\d+)\.dll", r"cublas(?:lt)?64_(\d+)\.dll"):
        majors = {int(m.group(1)) for m in (re.fullmatch(pat, n) for n in imports) if m}
        if majors:
            return majors.pop() if len(majors) == 1 else None
    return None


def _check_posix_safe_entry(path: str, *, expect_dir: bool, label: str) -> None:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"quantfunc: cannot stat {label} ({path}): {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise RuntimeError(f"quantfunc: refusing symlinked {label}: {path}")
    if expect_dir and not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"quantfunc: {label} is not a directory: {path}")
    if not expect_dir and not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"quantfunc: {label} is not a regular file: {path}")
    if (st.st_mode & 0o022) != 0:
        raise RuntimeError(
            f"quantfunc: refusing group/world-writable {label} ({path}, mode={oct(stat.S_IMODE(st.st_mode))})"
        )
    allowed_uids = {0}
    if hasattr(os, "geteuid"):
        allowed_uids.add(os.geteuid())
    if st.st_uid not in allowed_uids:
        raise RuntimeError(
            f"quantfunc: refusing {label} ({path}) owned by untrusted uid={st.st_uid}"
        )


def validate_plugin_dir(path: str | os.PathLike[str]) -> str:
    """Cross-platform safety check for ``QF_KERNEL_PLUGIN_DIR`` before any ``dlopen``/``LoadLibrary``.

    - POSIX: verifies directory and every ``.so`` file via ``os.lstat`` (no symlinks,
      no group/world write bits ``0o022``, owner is ``euid`` or ``root``).
    - Windows: refuses symlinked directory and ``.dll`` files, rejects non-kernel host DLLs
      importing CUDA, and verifies kernel plugin DLLs are valid PE images with safe CUDA imports.
    """
    dir_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.name == "posix":
        _check_posix_safe_entry(dir_path, expect_dir=True, label="QF_KERNEL_PLUGIN_DIR")
        for entry in sorted(os.listdir(dir_path)):
            if entry.endswith(".so") or ".so." in entry:
                _check_posix_safe_entry(
                    os.path.join(dir_path, entry),
                    expect_dir=False,
                    label=f"kernel plugin file {entry}",
                )
        return dir_path

    # Windows / non-POSIX validation
    if os.path.islink(dir_path):
        raise RuntimeError(f"quantfunc: refusing symlinked QF_KERNEL_PLUGIN_DIR: {dir_path}")
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"quantfunc: QF_KERNEL_PLUGIN_DIR is not a directory: {dir_path}")
    torch_major = _torch_cuda_major()
    allow_unverified = _allow_unverified_toolchain()
    for entry in sorted(os.listdir(dir_path)):
        if not entry.lower().endswith(".dll"):
            continue
        entry_path = os.path.join(dir_path, entry)
        if os.path.islink(entry_path):
            raise RuntimeError(f"quantfunc: refusing symlinked plugin DLL: {entry_path}")
        if not os.path.isfile(entry_path):
            continue
        if _ENGINE_KERNEL_WIN_RE.fullmatch(entry):
            if not _is_pe(entry_path):
                raise RuntimeError(f"quantfunc: kernel plugin DLL is not a valid PE binary: {entry_path}")
            if not allow_unverified:
                dll_major = _so_cuda_major(entry_path)
                if dll_major is None:
                    raise RuntimeError(
                        f"quantfunc: kernel plugin DLL ({entry_path}) has missing or conflicting CUDA PE imports"
                    )
                if torch_major is not None and dll_major != torch_major:
                    raise RuntimeError(
                        f"quantfunc: kernel plugin DLL ({entry_path}) links CUDA {dll_major}, "
                        f"mismatched with torch CUDA {torch_major}"
                    )
        elif _is_pe(entry_path):
            leaked = _host_forbidden_cuda_pe(entry_path)
            if leaked:
                raise RuntimeError(
                    f"quantfunc: non-kernel DLL ({entry_path}) in QF_KERNEL_PLUGIN_DIR has forbidden "
                    f"CUDA PE imports ({', '.join(leaked)})"
                )
    return dir_path


# ── Marker / pair management ─────────────────────────────────────────────────


def _pair_dir(m: dict[str, Any]) -> str:
    return f"{m['version']}-{m['set']}-cu{m['cuda']}"


def _read_marker(path: str) -> dict[str, Any] | None:
    name = _ENGINE_MARKER_RE.fullmatch(os.path.basename(path))
    plat = _engine_platform()
    if plat is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        files = [m["host"], m["kernel"]] if plat["kernel"] else [m["host"]]
        ok = (
            name is not None
            and m["set"] == name.group(1)
            and type(m["cuda"]) is int
            and m["cuda"] == int(name.group(2))
            and _ENGINE_VERSION_RE.fullmatch(m["version"]) is not None
            and m["host"] == plat["hosts"].get(m["cuda"])
            and (
                _kernel_regex().fullmatch(m["kernel"]) is not None
                and m["kernel"].lower() == _kernel_filename(m["cuda"]).lower()
                if plat["kernel"]
                else m["kernel"] is None
            )
            and all(type(s) is int and s in _CONSUMER_SMS for s in m["sms"])
            and sorted(m["sha256"]) == sorted(files)
            and all(_ENGINE_SHA256_RE.fullmatch(h) for h in m["sha256"].values())
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return m if ok else None


def _markers(engine_dir: str | os.PathLike[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    d = _engine_bin_dir(engine_dir)
    try:
        names = sorted(n for n in os.listdir(d) if _ENGINE_MARKER_RE.fullmatch(n))
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(d, n)
        m = _read_marker(p)
        if m:
            out.append((p, m))
    return out


def _installed_pair(
    device_id: int | None = None,
    engine_dir: str | os.PathLike[str] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    dev = _ENGINE_DEVICE if device_id is None else int(device_id)
    major, sm = _torch_cuda_major(), _gpu_sm(dev)
    mine = [(p, m) for p, m in _markers(engine_dir) if m["cuda"] == major and sm in m["sms"]]
    return max(mine, key=lambda pm: _version_key(pm[1]["version"])) if mine else (None, None)


def _marker_of(
    so_path: str,
    engine_dir: str | os.PathLike[str] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    here = os.path.realpath(os.path.dirname(so_path))
    bdir = _engine_bin_dir(engine_dir)
    return next(
        ((p, m) for p, m in _markers(engine_dir) if os.path.realpath(os.path.join(bdir, _pair_dir(m))) == here),
        (None, None),
    )


def _pair_intact(m: dict[str, Any], engine_dir: str | os.PathLike[str] | None = None) -> bool:
    pair = os.path.join(_engine_bin_dir(engine_dir), _pair_dir(m))
    try:
        return all(_sha256_of(os.path.join(pair, n)) == h for n, h in m["sha256"].items())
    except OSError:
        return False


def _claim(
    bin_dir: str,
    marker: str,
    sms: list[int],
    chosen: dict[str, Any] | None = None,
    engine_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Assign ownership of ``sms`` for one CUDA major to ``marker``, stripping or removing superseded pairs."""
    m_re = _ENGINE_MARKER_RE.fullmatch(os.path.basename(marker))
    assert m_re is not None
    major = int(m_re.group(2))
    want = sorted(set(sms))
    for p, m in _markers(engine_dir):
        if m["cuda"] != major or p == marker:
            continue
        left = [s for s in m["sms"] if s not in sms]
        if left == m["sms"]:
            continue
        if left:
            _engine_write_file(p, json.dumps(dict(m, sms=left)).encode("utf-8"))
        else:
            os.remove(p)
            shutil.rmtree(os.path.join(bin_dir, _pair_dir(m)), ignore_errors=True)
    if chosen is None:
        chosen = _read_marker(marker)
        if chosen is None or sorted(chosen["sms"]) == want:
            return
        chosen = dict(chosen, sms=want)
    _engine_write_file(marker, json.dumps(chosen).encode("utf-8"))


def _manifest_key(gpu_set: str, name: str) -> str:
    plat = _engine_platform()
    if plat and plat["kernel"] and name in plat["hosts"].values():
        return name
    return f"{gpu_set}/{name}"


def _resolve_kernel_for_set(
    version: str,
    gpu_set: str,
    sm: int,
    major: int,
    hashes: dict[str, str],
) -> str:
    cand = _kernel_filename(major)
    if _BIN_SUBDIR == "windows":
        prefix = f"{gpu_set}/"
        found = [
            k[len(prefix) :]
            for k in hashes
            if k.startswith(prefix) and k[len(prefix) :].lower() == cand.lower()
        ]
        if len(found) == 1:
            return found[0]
    elif _manifest_key(gpu_set, cand) in hashes:
        return cand
    raise RuntimeError(
        f"Release {version} manifest has no per-SM CUDA kernel plugin for {gpu_set} (SM {sm}, CUDA {major})"
    )


def _engine_pick_version(
    versions: dict[str, Any] | None,
    compat_version: str,
    major: int,
) -> str | None:
    """Select the newest compatible engine version, preferring ``sdk`` key with ``comfy`` fallback."""
    suffix = "-12" if major == 12 else ""
    best = best_lib = None
    for key, info_entry in (versions or {}).items():
        if not (
            isinstance(key, str)
            and _ENGINE_VERSION_RE.fullmatch(key)
            and isinstance(info_entry, dict)
            and info_entry.get("kernel_so")
        ):
            continue
        req = info_entry.get(
            "sdk" + suffix,
            info_entry.get("sdk", info_entry.get("comfy" + suffix, info_entry.get("comfy"))),
        )
        lib = info_entry.get("lib" + suffix, info_entry.get("lib", key))
        if not all(isinstance(x, str) and _ENGINE_VERSION_RE.fullmatch(x) for x in (req, lib)):
            continue
        if _version_key(req) <= _version_key(compat_version) and (
            best is None or _version_key(lib) > _version_key(best_lib)
        ):
            best, best_lib = key, lib
    return best


def _engine_sets(version: str, hashes: dict[str, str], base_url: str) -> dict[str, list[int]]:
    if "sets.json" not in hashes:
        raise RuntimeError(f"Release {version} publishes no sets.json; no engine was installed")
    raw = _engine_http_get(f"{base_url}/{version}/{_BIN_SUBDIR}/sets.json")
    if hashlib.sha256(raw).hexdigest() != hashes["sets.json"]:
        raise RuntimeError(f"Release {version} sets.json does not match its published SHA-256")
    doc = json.loads(raw)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if not (type(schema) is int and schema == _ENGINE_SETS_SCHEMA):
        raise RuntimeError(
            f"Release {version} sets.json has schema {schema!r}; expected schema {_ENGINE_SETS_SCHEMA}"
        )
    sets = doc.get("sets")
    if not (
        isinstance(sets, dict)
        and sets
        and all(
            isinstance(k, str)
            and _ENGINE_SET_RE.fullmatch(k)
            and isinstance(v, list)
            and v
            and all(type(s) is int for s in v)
            for k, v in sets.items()
        )
    ):
        raise RuntimeError(f"Release {version} sets.json has an invalid GPU-class map")
    consumer = {k: [s for s in v if s in _CONSUMER_SMS] for k, v in sets.items()}
    consumer = {k: v for k, v in consumer.items() if v}
    if not consumer:
        raise RuntimeError(f"Release {version} sets.json publishes no supported consumer GPU architectures")
    return consumer


def _install_addons(
    pair: str,
    version: str,
    info_entry: dict[str, Any],
    gpu_set: str,
    major: int,
    hashes: dict[str, str],
    base_url: str,
) -> None:
    """Install optional add-on shared libraries declared in version.json without overwriting host/kernel."""
    names = info_entry.get("addons-12" if major == 12 else "addons")
    plat = _engine_platform()
    reserved = set(plat["hosts"].values()) if plat else set()
    kpat = _kernel_regex()
    for name in names if isinstance(names, list) else ():
        if not (isinstance(name, str) and _ENGINE_ADDON_RE.fullmatch(name)) or name in reserved or kpat.fullmatch(name):
            continue
        key = _manifest_key(gpu_set, name)
        want = hashes.get(key)
        if want is None:
            continue
        dest = os.path.join(pair, name)
        part = _make_part_path(pair, name)
        try:
            if not os.path.islink(dest) and os.path.isfile(dest) and _sha256_of(dest) == want:
                _harden_posix_mode(dest, 0o755)
                continue
            if _engine_fetch_to(f"{base_url}/{version}/{_BIN_SUBDIR}/{key}", part, f"engine {version}: {name}") != want:
                raise RuntimeError(f"{key} does not match its published SHA-256")
            os.replace(part, dest)
        except Exception:  # noqa: BLE001
            for leftover in (part, dest):
                try:
                    os.remove(leftover)
                except OSError:
                    pass


@contextlib.contextmanager
def _install_lock(path: str):
    with open(path, "a+b") as fh:
        if os.name == "nt" or _BIN_SUBDIR == "windows":
            import errno
            import msvcrt

            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno != errno.EACCES:
                        raise
                    time.sleep(0.2)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
            yield


def _resolve_base_url(base_url: str | None = None) -> str:
    raw = (base_url if base_url is not None else os.environ.get(_ENV_ENGINE_BASE_URL, "")).strip()
    url = raw or _DEFAULT_ENGINE_BASE_URL
    return validate_https_url(url).rstrip("/")


def check_engine_update(
    device_idx: int = 0,
    engine_dir: str | os.PathLike[str] | None = None,
    base_url: str | None = None,
) -> UpdateInfo:
    """Read-only check for newer or drifted engine releases without modifying disk state."""
    why = _engine_local_choice(engine_dir)
    if why:
        return UpdateInfo(
            installed_version=None,
            latest_version=None,
            update_available=False,
            detail=why,
        )
    resolved_base = _resolve_base_url(base_url)
    plat = _engine_platform()
    if plat is None:
        return UpdateInfo(
            installed_version=None,
            latest_version=None,
            update_available=False,
            detail=f"Unsupported platform {platform.system()}",
        )
    major, sm = engine_choice(device_idx)
    _, installed = _installed_pair(device_id=device_idx, engine_dir=engine_dir)
    installed_ver = installed["version"] if installed else None

    compat_version = _engine_compat_version()
    versions = json.loads(_engine_http_get(f"{resolved_base}/version.json")).get(plat["key"])
    latest = _engine_pick_version(versions, compat_version, major)
    if latest is None:
        return UpdateInfo(
            installed_version=installed_ver,
            latest_version=None,
            update_available=False,
            cuda_major=major,
            detail=f"No published release compatible with {compat_version} (CUDA {major})",
        )
    manifest = json.loads(_engine_http_get(f"{resolved_base}/{latest}/verify.json"))
    if not (
        isinstance(manifest, dict)
        and isinstance(manifest.get("schema"), int)
        and 1 <= manifest["schema"] <= _ENGINE_VERIFY_SCHEMA_MAX
        and isinstance(manifest.get(plat["key"]), dict)
    ):
        raise RuntimeError(f"Release {latest} verify.json is not a recognized manifest")
    hashes = manifest[plat["key"]]
    sets = _engine_sets(latest, hashes, resolved_base)
    gpu_set = next((k for k, sms in sets.items() if sm in sms), None)
    if gpu_set is None:
        return UpdateInfo(
            installed_version=installed_ver,
            latest_version=latest,
            update_available=False,
            cuda_major=major,
            detail=f"No kernel plugin published for SM {sm // 10}.{sm % 10}",
        )

    if installed is None:
        return UpdateInfo(
            installed_version=None,
            latest_version=latest,
            update_available=True,
            gpu_set=gpu_set,
            cuda_major=major,
            detail=f"Engine {latest} available for installation ({gpu_set}, CUDA {major})",
        )

    if installed_ver != latest:
        return UpdateInfo(
            installed_version=installed_ver,
            latest_version=latest,
            update_available=True,
            gpu_set=gpu_set,
            cuda_major=major,
            detail=f"Update available: {installed_ver} -> {latest} ({gpu_set}, CUDA {major})",
        )

    hashes_match = all(hashes.get(_manifest_key(gpu_set, n)) == h for n, h in installed["sha256"].items())
    intact = _pair_intact(installed, engine_dir=engine_dir)
    if not hashes_match or not intact:
        return UpdateInfo(
            installed_version=installed_ver,
            latest_version=latest,
            update_available=True,
            gpu_set=gpu_set,
            cuda_major=major,
            detail=f"SHA-256 refresh required for engine {latest} ({gpu_set}, CUDA {major})",
        )

    return UpdateInfo(
        installed_version=installed_ver,
        latest_version=latest,
        update_available=False,
        gpu_set=gpu_set,
        cuda_major=major,
        detail=f"Engine {installed_ver} is up to date ({gpu_set}, CUDA {major})",
    )


def install_engine(
    device_idx: int | None = None,
    force: bool = False,
    engine_dir: str | os.PathLike[str] | None = None,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Download, verify (HTTPS + SHA-256 + ELF/PE + pair-id), and atomically install the engine pair."""
    why = _engine_local_choice(engine_dir)
    if why:
        _engine_status("local", why)
        return None
    resolved_base = _resolve_base_url(base_url)
    plat = _engine_platform()
    if plat is None:
        raise EngineNotInstallable(
            f"QuantFunc engines are published for Linux and Windows only (current platform: {platform.system()})"
        )
    if platform.machine() not in plat["arches"]:
        raise EngineNotInstallable(
            f"QuantFunc engines are published for {'/'.join(plat['arches'])} only, "
            f"and this machine is {platform.machine() or 'unknown'}"
        )
    root_dir = default_engine_dir(engine_dir)
    _refuse_symlink_dir(root_dir, "engine cache root")
    bin_dir = _engine_bin_dir(engine_dir)
    _refuse_symlink_dir(bin_dir, "engine bin")
    os.makedirs(bin_dir, exist_ok=True)
    _harden_posix_mode(bin_dir, 0o755)
    with _install_lock(os.path.join(bin_dir, _ENGINE_INSTALL_LOCKFILE)):
        return _install_pair(
            bin_dir,
            _ENGINE_DEVICE if device_idx is None else int(device_idx),
            force=bool(force),
            engine_dir=engine_dir,
            base_url=resolved_base,
        )


def _install_pair(
    bin_dir: str,
    device_idx: int,
    *,
    force: bool,
    engine_dir: str | os.PathLike[str] | None,
    base_url: str,
) -> dict[str, Any]:
    _engine_status("checking")
    major, sm = engine_choice(device_idx)
    compat_version = _engine_compat_version()
    plat = _engine_platform()
    assert plat is not None
    versions = json.loads(_engine_http_get(f"{base_url}/version.json")).get(plat["key"])
    version = _engine_pick_version(versions, compat_version, major)
    if version is None:
        raise EngineNotInstallable(
            f"No published QuantFunc engine with per-SM layout is compatible with {compat_version} (CUDA {major})"
        )
    manifest = json.loads(_engine_http_get(f"{base_url}/{version}/verify.json"))
    if not (
        isinstance(manifest, dict)
        and isinstance(manifest.get("schema"), int)
        and 1 <= manifest["schema"] <= _ENGINE_VERIFY_SCHEMA_MAX
        and isinstance(manifest.get(plat["key"]), dict)
    ):
        raise RuntimeError(f"Release {version} verify.json is not a recognized manifest")
    hashes = manifest[plat["key"]]
    sets = _engine_sets(version, hashes, base_url)
    gpu_set = next((k for k, sms in sets.items() if sm in sms), None)
    if gpu_set is None:
        published = ", ".join(f"{s // 10}.{s % 10}" for s in sorted({s for v in sets.values() for s in v}))
        raise EngineNotInstallable(
            f"No QuantFunc engine is published for SM {sm // 10}.{sm % 10}; "
            f"release {version} publishes SM {published}"
        )
    host = plat["hosts"][major]
    marker = os.path.join(bin_dir, f".engine-{gpu_set}-cu{major}.json")
    have = _read_marker(marker)
    known_bad = None
    if have and not force and have["version"] == version:
        if all(hashes.get(_manifest_key(gpu_set, n)) == h for n, h in have["sha256"].items()):
            pair_have = os.path.join(bin_dir, _pair_dir(have))
            _refuse_symlink_dir(pair_have, "engine pair")
            if _pair_intact(have, engine_dir=engine_dir):
                _harden_posix_mode(pair_have, 0o755)
                for n in have["sha256"]:
                    _harden_posix_mode(os.path.join(pair_have, n), 0o755)
                _harden_posix_mode(marker, 0o644)
                _claim(bin_dir, marker, sets[gpu_set], engine_dir=engine_dir)
                _install_addons(pair_have, version, versions[version], gpu_set, major, hashes, base_url)
                _cleanup_stale(pair_have)
                _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
                return have
        known_bad, have = version, None

    kernel = _resolve_kernel_for_set(version, gpu_set, sm, major, hashes) if plat["kernel"] else None
    pair = os.path.join(bin_dir, f"{version}-{gpu_set}-cu{major}")
    _refuse_symlink_dir(pair, "engine pair")
    os.makedirs(pair, exist_ok=True)
    _harden_posix_mode(pair, 0o755)
    _cleanup_stale(pair)
    order = ([kernel] if kernel else []) + [host]
    got: dict[str, str] = {}
    part_by_name: dict[str, str] = {}
    backups: dict[str, str] = {}
    stales: dict[str, str] = {}
    replaced: list[str] = []
    m = {
        "version": version,
        "set": gpu_set,
        "cuda": major,
        "sms": sets[gpu_set],
        "host": host,
        "kernel": kernel,
        "sha256": got,
    }
    rollback_failed = False
    try:
        for name in ((host, kernel) if kernel else (host,)):
            key = _manifest_key(gpu_set, name)
            if not _ENGINE_SHA256_RE.fullmatch(str(hashes.get(key, ""))):
                raise RuntimeError(f"Release {version} manifest has no SHA-256 for {key}")
            part_path = _make_part_path(pair, name)
            part_by_name[name] = part_path
            got[name] = _engine_fetch_to(
                f"{base_url}/{version}/{_BIN_SUBDIR}/{key}",
                part_path,
                f"engine {version}: {name}",
            )
            if got[name] != hashes[key]:
                raise RuntimeError(f"{key} does not match its published SHA-256 (download corrupt or tampered)")
        if kernel:
            host_part = part_by_name[host]
            kernel_part = part_by_name[kernel]
            if _BIN_SUBDIR == "windows":
                host_cuda = _host_forbidden_cuda_pe(host_part)
                if host_cuda:
                    raise RuntimeError(
                        f"Release {version} host library unexpectedly imports CUDA ({', '.join(host_cuda)}); "
                        "PC Engine host must have zero CUDA PE imports"
                    )
            else:
                host_cuda = _host_forbidden_cuda(host_part)
                if host_cuda:
                    raise RuntimeError(
                        f"Release {version} host library unexpectedly links CUDA ({', '.join(host_cuda)}); "
                        "PC Engine host must have zero CUDA DT_NEEDED imports"
                    )
                ids = [elf_pair_id(host_part), elf_pair_id(kernel_part)]
                if ids[0] is None or ids[0] != ids[1]:
                    raise RuntimeError(
                        f"Release {version} host and kernel libraries are not from one build "
                        f"(pair ids {ids[0]} / {ids[1]}); nothing was installed"
                    )
        for n in order:
            dst = os.path.join(pair, n)
            if os.path.exists(dst):
                bak = os.path.join(pair, f".{n}.bak-{os.getpid()}-{secrets.token_hex(6)}")
                shutil.copy2(dst, bak)
                _harden_posix_mode(bak, 0o755)
                backups[n] = bak
        for n in order:
            part_path = part_by_name[n]
            dst = os.path.join(pair, n)
            try:
                os.replace(part_path, dst)
                replaced.append(n)
            except PermissionError:
                if _BIN_SUBDIR != "windows" or not os.path.exists(dst):
                    raise
                stale = os.path.join(pair, f".{n}.stale-{os.getpid()}-{secrets.token_hex(4)}")
                os.rename(dst, stale)
                stales[n] = stale
                replaced.append(n)
                os.replace(part_path, dst)
        _install_addons(pair, version, versions[version], gpu_set, major, hashes, base_url)
        _claim(bin_dir, marker, sets[gpu_set], chosen=m, engine_dir=engine_dir)
    except Exception as exc:
        for n in reversed(replaced):
            dst = os.path.join(pair, n)
            bak = backups.get(n)
            stale = stales.get(n)
            try:
                if stale is not None and os.path.exists(stale):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(stale, dst)
                    stales.pop(n, None)
                elif bak is not None and os.path.exists(bak):
                    os.replace(bak, dst)
                    _harden_posix_mode(dst, 0o755)
                elif os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            try:
                os.remove(marker)
            except OSError:
                pass
            raise RuntimeError(
                console_safe(
                    f"Engine {version} update failed ({type(exc).__name__}: {exc}) and rollback failed; "
                    "stale marker removed"
                )
            ) from exc
        if known_bad:
            raise RuntimeError(
                console_safe(
                    f"Installed engine {known_bad} no longer matches SHA-256 and re-download failed "
                    f"({type(exc).__name__}: {exc}); existing installation preserved"
                )
            ) from exc
        raise
    finally:
        for p in list(part_by_name.values()) + list(backups.values()) + list(stales.values()):
            try:
                os.remove(p)
            except OSError:
                pass

    keep = {_pair_dir(m), have and _pair_dir(have)}
    for d in os.listdir(bin_dir):
        p_match = _ENGINE_PAIR_RE.fullmatch(d)
        if p_match and p_match.group(2) == gpu_set and int(p_match.group(3)) == major and d not in keep:
            shutil.rmtree(os.path.join(bin_dir, d), ignore_errors=True)
    _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
    return m


def _cleanup_stale(pair_dir: str) -> None:
    for stale_fn in os.listdir(pair_dir):
        if _ENGINE_STALE_RE.fullmatch(stale_fn):
            stale_p = os.path.join(pair_dir, stale_fn)
            if os.path.islink(stale_p) or not os.path.isfile(stale_p):
                continue
            try:
                os.remove(stale_p)
            except OSError:
                pass


# ── Self-healing on load failure & background daemon auto-updater ────────────


def start_engine_install(
    device_idx: int | None = None,
    engine_dir: str | os.PathLike[str] | None = None,
    base_url: str | None = None,
) -> threading.Thread:
    """Run ``install_engine`` in a background daemon thread with offline-tolerant status updates."""
    global _ENGINE_DEVICE
    if device_idx is not None:
        _ENGINE_DEVICE = int(device_idx)
    eff_dev = _ENGINE_DEVICE if device_idx is None else int(device_idx)

    def _run() -> None:
        try:
            install_engine(device_idx=eff_dev, engine_dir=engine_dir, base_url=base_url)
        except EngineNotInstallable as exc:
            _engine_status("unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001
            kept = _installed_pair(device_id=eff_dev, engine_dir=engine_dir)[1]
            _engine_status("offline" if kept else "failed", f"{type(exc).__name__}: {exc}")

    t = threading.Thread(target=_run, name="qf-engine-install", daemon=True)
    t.start()
    return t


def _engine_load_failed(
    so_path: str,
    err: BaseException,
    *,
    engine_dir: str | os.PathLike[str] | None = None,
    trigger_reinstall: bool = True,
) -> str:
    """Record one-shot ``.reinstalled`` sentinel on CDLL failure and trigger a single background reinstall."""
    detail = str(err)[:400]
    mpath, m = _marker_of(so_path, engine_dir=engine_dir)
    if not m or not mpath:
        return f"the engine library failed to load: {detail}"
    flag = os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED)
    if os.path.exists(flag):
        return (
            f"the engine library {m['version']} failed to load again after one re-download: {detail}. "
            "It will not be re-downloaded again for this release: check the NVIDIA driver and GPU."
        )
    _engine_write_file(flag, m["version"].encode("utf-8"))
    try:
        os.remove(mpath)
    except OSError:
        pass
    if trigger_reinstall:
        start_engine_install(engine_dir=engine_dir)
    return (
        f"the engine library {m['version']} failed to load: {detail}. "
        "It is being re-downloaded once; retry after installation finishes."
    )


def _engine_load_ok(
    so_path: str,
    *,
    engine_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Clear the one-shot ``.reinstalled`` sentinel once the engine loads cleanly."""
    mpath, m = _marker_of(so_path, engine_dir=engine_dir)
    if m and mpath:
        try:
            os.remove(os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED))
        except OSError:
            pass


# ── Resolution & toolchain compatibility check ───────────────────────────────


def _local_host_names(plat: dict[str, Any] | None) -> list[str]:
    names: list[str] = []
    if plat is not None:
        preferred = plat["hosts"].get(_torch_cuda_major())
        if preferred:
            names.append(preferred)
    if _LIB_BASENAME not in names:
        names.append(_LIB_BASENAME)
    return names


def has_ready_so(
    device_id: int | None = None,
    engine_dir: str | os.PathLike[str] | None = None,
) -> bool:
    """Read-only probe mirroring ``resolve_so_path`` priority without downloading or deleting markers."""
    try:
        override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
        if override:
            return os.path.isfile(override)
        root_dir = default_engine_dir(engine_dir)
        if os.path.islink(root_dir):
            return False
        bin_dir = _engine_bin_dir(engine_dir)
        if os.path.islink(bin_dir):
            return False
        plat = _engine_platform()
        local = _has_local_build_lock(engine_dir)
        if local or plat is None:
            names = _local_host_names(plat)
            return any(
                os.path.isfile(os.path.join(d, name))
                for name in names
                for d in (bin_dir, root_dir)
            )
        _, m = _installed_pair(device_id=device_id, engine_dir=engine_dir)
        if not m:
            return False
        pair = os.path.join(bin_dir, _pair_dir(m))
        if os.path.islink(pair):
            return False
        return _pair_intact(m, engine_dir=engine_dir)
    except OSError:
        return False


def resolve_so_path(
    device_id: int | None = None,
    engine_dir: str | os.PathLike[str] | None = None,
    *,
    trigger_self_heal: bool = True,
) -> str:
    """Resolve the verified engine host library path; self-heals tampered cache markers and fails closed."""
    override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"quantfunc: {_ENV_SO_OVERRIDE} names {override!r}, which is not an existing file")
        return os.path.realpath(override)
    root_dir = default_engine_dir(engine_dir)
    _refuse_symlink_dir(root_dir, "engine cache root")
    bin_dir = _engine_bin_dir(engine_dir)
    _refuse_symlink_dir(bin_dir, "engine bin")
    plat = _engine_platform()
    local = _has_local_build_lock(engine_dir)
    if local or plat is None:
        names = _local_host_names(plat)
        for name in names:
            for d in (bin_dir, root_dir):
                cand = os.path.join(d, name)
                if os.path.isfile(cand):
                    return os.path.realpath(cand)
        raise RuntimeError(
            f"quantfunc: no engine library found in {bin_dir}; run `quantfunc install-engine` "
            f"or set {_ENV_SO_OVERRIDE}=<abs path>"
        )
    mpath, m = _installed_pair(device_id=device_id, engine_dir=engine_dir)
    if m and mpath:
        pair = os.path.join(bin_dir, _pair_dir(m))
        _refuse_symlink_dir(pair, "engine pair")
        if _pair_intact(m, engine_dir=engine_dir):
            _harden_posix_mode(bin_dir, 0o755)
            _harden_posix_mode(pair, 0o755)
            for n in m["sha256"]:
                _harden_posix_mode(os.path.join(pair, n), 0o755)
            return os.path.realpath(os.path.join(pair, m["host"]))
        try:
            os.remove(mpath)
        except OSError:
            pass
        if trigger_self_heal:
            start_engine_install(device_idx=device_id, engine_dir=engine_dir)
        raise RuntimeError(
            f"quantfunc: installed engine {m['version']} failed SHA-256 verification on disk "
            "(marker removed; self-healing re-download triggered)"
        )
    state, detail = engine_install_status()
    if state in ("checking", "downloading"):
        raise RuntimeError(f"quantfunc: engine library is still downloading ({detail}); retry once complete")
    raise RuntimeError(
        f"quantfunc: no QuantFunc engine is installed in {bin_dir} "
        f"({state}: {detail or 'run `quantfunc install-engine`'})."
    )


def _find_kernel_in_dir(d: str, want: str, pat, is_pe: bool) -> str | None:
    try:
        entries = sorted(n for n in os.listdir(d) if pat.fullmatch(n) and os.path.isfile(os.path.join(d, n)))
    except OSError:
        return None
    for n in entries:
        if (n.lower() == want.lower()) if is_pe else (n == want):
            return os.path.join(d, n)
    return None


def _select_kernel_plugin(
    so_path: str,
    *,
    engine_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    is_pe = _is_pe(so_path)
    want = _kernel_filename(_host_cuda_major(so_path, is_pe), is_pe)
    pat = _kernel_regex(is_pe)
    so_dir = os.path.dirname(os.path.realpath(so_path))
    env_dir = os.environ.get(_ENV_KERNEL_PLUGIN_DIR, "").strip()
    if env_dir:
        safe_dir = validate_plugin_dir(env_dir)
        found = _find_kernel_in_dir(safe_dir, want, pat, is_pe)
        if found is not None:
            return found
    _, marker = _marker_of(so_path, engine_dir=engine_dir)
    if marker and marker.get("kernel") and (
        marker["kernel"].lower() == want.lower() if is_pe else marker["kernel"] == want
    ):
        p = os.path.join(so_dir, marker["kernel"])
        if os.path.isfile(p):
            return p
    return _find_kernel_in_dir(so_dir, want, pat, is_pe)


def assert_toolchain_compatible(so_path: str | os.PathLike[str]) -> None:
    """Fail-closed CUDA toolchain guard (zero-CUDA host + per-SM kernel plugin match)."""
    env_plugin_dir = os.environ.get(_ENV_KERNEL_PLUGIN_DIR, "").strip()
    if env_plugin_dir:
        validate_plugin_dir(env_plugin_dir)
    if _allow_unverified_toolchain():
        return
    path_str = os.fspath(so_path)
    torch_major = _torch_cuda_major()
    is_elf, is_pe = _is_elf(path_str), _is_pe(path_str)
    if not is_elf and not is_pe:
        raise RuntimeError(
            f"quantfunc: refusing to load {path_str} — expected Linux ELF or Windows PE engine binary "
            f"(torch CUDA={torch_major or 'none'}; set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 only if verified)"
        )
    leaked = _host_forbidden_cuda(path_str) if is_elf else _host_forbidden_cuda_pe(path_str)
    if leaked:
        kind = "DT_NEEDED" if is_elf else "PE"
        raise RuntimeError(
            f"quantfunc: refusing to load host library ({path_str}) — forbidden CUDA {kind} imports "
            f"({', '.join(leaked)}); CUDA must live exclusively in the per-SM kernel plugin"
        )
    target_bin = _select_kernel_plugin(path_str)
    if target_bin is None:
        want_name = _kernel_filename(_host_cuda_major(path_str, is_pe), is_pe)
        raise RuntimeError(
            f"quantfunc: refusing to load — missing per-SM CUDA kernel plugin ({want_name}) beside {path_str}"
        )
    so_major = _so_cuda_major(target_bin)
    if torch_major is None or so_major is None:
        raise RuntimeError(
            f"quantfunc: refusing to load — cannot verify CUDA toolchain match "
            f"(torch CUDA={torch_major or 'none'}, kernel plugin CUDA={so_major or 'unknown'} from {target_bin})"
        )
    if torch_major != so_major:
        raise RuntimeError(
            f"quantfunc: refusing to load — CUDA toolchain mismatch "
            f"(torch CUDA {torch_major} vs kernel plugin CUDA {so_major} in {target_bin})"
        )


__all__ = [
    "ENGINE_COMPAT_VERSION",
    "UpdateInfo",
    "default_engine_dir",
    "validate_https_url",
    "validate_plugin_dir",
    "engine_choice",
    "check_engine_update",
    "install_engine",
    "start_engine_install",
    "engine_install_status",
    "has_ready_so",
    "resolve_so_path",
    "elf_needed",
    "pe_imports",
    "elf_pair_id",
    "assert_toolchain_compatible",
]
