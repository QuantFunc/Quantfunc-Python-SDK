"""L0 unit, security, engine-update parity, and M1 mutation tests (pure CPU)."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import stat
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeQFLib

from quantfunc.core.config import WIRE_QUALITY_RATIO_KEY
from quantfunc.native import capi, errors, handle, installer, loader

# ── Pure-stdlib synthetic ELF64 & PE32+ builders for binary-guard testing ────


def _build_synthetic_elf(needed: list[str], pair_id: str | None = None) -> bytes:
    """Construct a minimal valid little-endian ELF64 image with DT_NEEDED and optional .qf_pair_id."""
    shstrtab = bytearray(b"\x00")
    dynstr = bytearray(b"\x00")

    def add_str(buf: bytearray, s: str) -> int:
        off = len(buf)
        buf += s.encode("ascii") + b"\x00"
        return off

    name_dyn = add_str(shstrtab, ".dynamic")
    name_dynstr = add_str(shstrtab, ".dynstr")
    name_shstrtab = add_str(shstrtab, ".shstrtab")
    name_pair = add_str(shstrtab, ".qf_pair_id") if pair_id is not None else None

    dyn_entries = bytearray()
    for lib in needed:
        str_off = add_str(dynstr, lib)
        dyn_entries += struct.pack("<qQ", 1, str_off)  # DT_NEEDED
    dyn_entries += struct.pack("<qQ", 0, 0)  # DT_NULL

    pair_bytes = (pair_id.encode("ascii") + b"\x00") if pair_id is not None else b""

    ehdr_size = 64
    shdr_size = 64
    cur = ehdr_size

    dyn_off = cur
    cur += len(dyn_entries)
    dynstr_off = cur
    cur += len(dynstr)
    shstr_off = cur
    cur += len(shstrtab)
    pair_off = cur
    cur += len(pair_bytes)

    shoff = cur
    sections = [
        # (name_idx, sh_type, sh_offset, sh_size, sh_link)
        (0, 0, 0, 0, 0),
        (name_dyn, 6, dyn_off, len(dyn_entries), 2),  # SHT_DYNAMIC -> links to section 2 (.dynstr)
        (name_dynstr, 3, dynstr_off, len(dynstr), 0),  # SHT_STRTAB
        (name_shstrtab, 3, shstr_off, len(shstrtab), 0),
    ]
    if pair_id is not None and name_pair is not None:
        sections.append((name_pair, 1, pair_off, len(pair_bytes), 0))  # SHT_PROGBITS

    shstrndx = 3
    ehdr = bytearray(ehdr_size)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # ELFCLASS64
    ehdr[5] = 1  # ELFDATA2LSB
    ehdr[6] = 1
    struct.pack_into("<Q", ehdr, 0x28, shoff)
    struct.pack_into("<HHH", ehdr, 0x3A, shdr_size, len(sections), shstrndx)

    shdrs = bytearray()
    for n_idx, s_type, s_off, s_sz, s_link in sections:
        ent = bytearray(shdr_size)
        struct.pack_into("<II", ent, 0, n_idx, s_type)
        struct.pack_into("<QQ", ent, 0x18, s_off, s_sz)
        struct.pack_into("<I", ent, 0x28, s_link)
        shdrs += ent

    return bytes(ehdr + dyn_entries + dynstr + shstrtab + pair_bytes + shdrs)


def _build_synthetic_pe(dll_names: list[str]) -> bytes:
    """Construct a minimal valid PE32+ image with an import directory listing ``dll_names``."""
    pe_off = 0x80
    opt_size = 240  # PE32+ optional header (112 bytes + 16 data directories * 8)
    sec_va = 0x1000
    sec_raw_ptr = 0x200

    # Build .idata section content: (len(dll_names) + 1) descriptors * 20 bytes + NUL-terminated names
    desc_table_size = (len(dll_names) + 1) * 20
    names_buf = bytearray()
    descs = bytearray()
    for name in dll_names:
        name_rva = sec_va + desc_table_size + len(names_buf)
        names_buf += name.encode("ascii") + b"\x00"
        desc = bytearray(20)
        struct.pack_into("<I", desc, 12, name_rva)
        descs += desc
    descs += bytes(20)  # null terminator descriptor
    idata = bytes(descs + names_buf)
    raw_size = max(0x200, ((len(idata) + 0x1FF) // 0x200) * 0x200)

    buf = bytearray(sec_raw_ptr + raw_size)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, pe_off)
    buf[pe_off : pe_off + 4] = b"PE\x00\x00"
    struct.pack_into("<H", buf, pe_off + 6, 1)  # NumberOfSections
    struct.pack_into("<H", buf, pe_off + 20, opt_size)  # SizeOfOptionalHeader

    opt = pe_off + 24
    struct.pack_into("<H", buf, opt, 0x20B)  # PE32+ magic
    dirs = opt + 112
    # DataDirectory[1] = Import Table (RVA, Size)
    struct.pack_into("<II", buf, dirs + 8 * 1, sec_va, len(idata))

    sec_hdr = opt + opt_size
    buf[sec_hdr : sec_hdr + 6] = b".idata"
    # VirtualSize, VirtualAddress, SizeOfRawData, PointerToRawData at +8
    struct.pack_into("<IIII", buf, sec_hdr + 8, len(idata), sec_va, raw_size, sec_raw_ptr)
    buf[sec_raw_ptr : sec_raw_ptr + len(idata)] = idata
    return bytes(buf)


class _FakeHttpResponse:
    def __init__(self, url: str, data: bytes) -> None:
        self._url = url
        self._stream = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def geturl(self) -> str:
        return self._url

    def read(self, n: int = -1) -> bytes:
        return self._stream.read(n)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _make_mock_release(
    *,
    version: str = "0.0.18",
    cuda: int = 13,
    gpu_set: str = "sm89",
    sms: list[int] | None = None,
    pair_id: str = "0123456789abcdef0123456789abcdef",
    host_needed: list[str] | None = None,
    kernel_needed: list[str] | None = None,
    kernel_pair_id: str | None = None,
    version_entry_extra: dict[str, Any] | None = None,
    addons_bytes: dict[str, bytes] | None = None,
    base_url: str = "https://releases.quantfunc.example/engine",
) -> tuple[dict[str, bytes], bytes, bytes]:
    sms = sms or [89]
    host_name = "libquantfunc-12.so" if cuda == 12 else "libquantfunc.so"
    kernel_name = "libquantfunc_kernels-12.so" if cuda == 12 else "libquantfunc_kernels.so"

    host_bytes = _build_synthetic_elf(host_needed if host_needed is not None else ["libc.so.6"], pair_id=pair_id)
    kernel_bytes = _build_synthetic_elf(
        kernel_needed if kernel_needed is not None else [f"libcudart.so.{cuda}"],
        pair_id=pair_id if kernel_pair_id is None else kernel_pair_id,
    )
    sets_bytes = json.dumps({"schema": 2, "sets": {gpu_set: sms}}).encode("utf-8")

    hashes = {
        "sets.json": hashlib.sha256(sets_bytes).hexdigest(),
        host_name: hashlib.sha256(host_bytes).hexdigest(),
        f"{gpu_set}/{kernel_name}": hashlib.sha256(kernel_bytes).hexdigest(),
    }
    routes: dict[str, bytes] = {}
    ver_entry: dict[str, Any] = {"kernel_so": True, "comfy": "0.0.08", "lib": version}
    if version_entry_extra:
        ver_entry.update(version_entry_extra)
    if addons_bytes:
        addon_key = "addons-12" if cuda == 12 else "addons"
        ver_entry[addon_key] = list(addons_bytes.keys())
        for aname, abytes in addons_bytes.items():
            hashes[f"{gpu_set}/{aname}"] = hashlib.sha256(abytes).hexdigest()
            routes[f"{base_url}/{version}/linux/{gpu_set}/{aname}"] = abytes

    version_json = json.dumps({"linux": {version: ver_entry}}).encode("utf-8")
    verify_json = json.dumps({"schema": 1, "linux": hashes}).encode("utf-8")

    routes.update(
        {
            f"{base_url}/version.json": version_json,
            f"{base_url}/{version}/verify.json": verify_json,
            f"{base_url}/{version}/linux/sets.json": sets_bytes,
            f"{base_url}/{version}/linux/{host_name}": host_bytes,
            f"{base_url}/{version}/linux/{gpu_set}/{kernel_name}": kernel_bytes,
        }
    )
    return routes, host_bytes, kernel_bytes


# ── Test 1: 25 symbols + 8 structs + constants ───────────────────────────────


def test_abi_25_symbols_and_structs(fake_qf_lib: FakeQFLib) -> None:
    assert len(capi._CAPI_SIGNATURES) == 25
    names = [n for n, _, _ in capi._CAPI_SIGNATURES]
    assert len(set(names)) == 25

    bound = loader._bind(fake_qf_lib)
    assert bound is fake_qf_lib
    for name, restype, argtypes in capi._CAPI_SIGNATURES:
        fn = getattr(bound, name)
        assert fn.restype == restype
        assert fn.argtypes == argtypes

    # Constants parity
    assert capi.QF_OK == 0
    assert capi.QF_ERR_UNSUPPORTED == 8
    assert capi.QF_DTYPE_FP32 == 0
    assert capi.QF_DTYPE_E2M1_PACKED == 7
    assert capi.QF_ATTN_CAP_FP16 == 1
    assert capi.QF_ATTN_CAP_INT8 == 2
    assert capi.QF_ATTN_CAP_INT4 == 4
    assert capi.QF_MAX_STEP_REFS == 16
    assert capi.QF_MAX_STEP_REF_AREA == 1 << 24
    assert capi.QF_UINT64_MAX == (1 << 64) - 1

    # Verify exact field ordering of all 8 structs
    assert [f[0] for f in capi.QFAuthConfig._fields_] == [
        "struct_size",
        "server_url",
        "api_key",
        "cache_dir",
        "device_id",
    ]
    assert [f[0] for f in capi.QFDeviceCaps._fields_] == [
        "struct_size",
        "backend_type",
        "device_id",
        "arch_code",
        "compute_unit_count",
        "total_vram_bytes",
        "free_vram_bytes",
        "attn_caps_mask",
        "reserved",
        "backend_feature_flags",
        "device_name",
        "uuid_hex",
        "plugin_path",
    ]
    assert [f[0] for f in capi.QFXfmCreateParams._fields_] == [
        "struct_size",
        "weights_path",
        "device_id",
        "config_json",
    ]
    assert [f[0] for f in capi.QFVramEstimate._fields_] == [
        "struct_size",
        "resident_weight_bytes",
        "min_offload_bytes",
        "working_set_floor_bytes",
        "recommended_offload_mode",
        "reserved",
    ]
    assert [f[0] for f in capi.QFSessionBeginParams._fields_] == [
        "struct_size",
        "width",
        "height",
        "num_steps",
        "max_context_dims",
        "cond_dtype",
        "producer_stream",
        "options_json",
    ]
    assert [f[0] for f in capi.QFStepParams._fields_] == [
        "struct_size",
        "latent_in",
        "velocity_out",
        "velocity_out_capacity",
        "dims",
        "dtype",
        "sigma",
        "step_index",
        "total_steps",
        "context",
        "context_dims",
        "context_dtype",
        "cfg_context_key",
        "producer_stream",
    ]
    assert [f[0] for f in capi.QFStepAvParams._fields_] == [
        "struct_size",
        "base",
        "audio_latent_in",
        "audio_velocity_out",
        "audio_velocity_out_capacity",
        "audio_dims",
        "audio_dtype",
        "audio_scale",
        "audio_reference_active",
    ]
    assert [f[0] for f in capi.QFStepRefsParams._fields_] == [
        "struct_size",
        "base",
        "num_refs",
        "ref_latents",
        "ref_dims",
        "image_slots",
    ]
    assert ctypes.sizeof(capi.QFStepAvParams) > ctypes.sizeof(capi.QFStepParams)
    assert ctypes.sizeof(capi.QFStepRefsParams) > ctypes.sizeof(capi.QFStepParams)


# ── Test 2: Missing symbol fail-closed ───────────────────────────────────────


def test_bind_missing_symbol_raises(fake_qf_lib: FakeQFLib) -> None:
    delattr(fake_qf_lib, "qf_xfm_step_refs")
    with pytest.raises(RuntimeError, match="qf_xfm_step_refs"):
        loader._bind(fake_qf_lib)


# ── Test 3: Status code -> 8 exception subclasses + console_safe ─────────────


def test_status_exception_mapping_and_console_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    assert errors.status_to_exception(capi.QF_OK, "ctx", "ok") is None

    expected = {
        capi.QF_ERR_INVALID_ARG: errors.InvalidArgError,
        capi.QF_ERR_BACKEND: errors.BackendError,
        capi.QF_ERR_OOM: errors.OutOfMemoryError,
        capi.QF_ERR_IO: errors.EngineIOError,
        capi.QF_ERR_INTERNAL: errors.InternalEngineError,
        capi.QF_ERR_BUSY: errors.EngineBusyError,
        capi.QF_ERR_AUTH: errors.AuthError,
        capi.QF_ERR_UNSUPPORTED: errors.UnsupportedError,
    }
    for code, cls in expected.items():
        exc = errors.status_to_exception(code, context="op", detail="boom")
        assert isinstance(exc, cls)
        assert isinstance(exc, errors.QuantFuncError)
        assert isinstance(exc, RuntimeError)
        assert exc.status_code == code
        assert "boom" in exc.detail

    # Unknown non-zero status falls back to base QuantFuncError
    unk = errors.status_to_exception(99, context="op", detail="weird")
    assert type(unk) is errors.QuantFuncError
    assert unk.status_code == 99

    # Simulate GBK Windows stdout codepage where emoji/box characters fail encoding
    fake_stream = types.SimpleNamespace(encoding="gbk")
    monkeypatch.setattr(sys, "stdout", fake_stream)
    monkeypatch.setattr(sys, "stderr", fake_stream)
    safe = errors.console_safe("engine error \U0001f680 \u2026 done")
    safe.encode("gbk")  # must not raise UnicodeEncodeError
    assert "\\U0001f680" in safe


# ── Test 4: default_engine_dir never points inside site-packages / quantfunc ─


def test_default_engine_dir_not_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QF_ENGINE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    default_dir = Path(installer.default_engine_dir()).resolve()
    pkg_dir = Path(installer.__file__).resolve().parent.parent
    assert not default_dir.is_relative_to(pkg_dir)
    if os.name == "posix":
        assert default_dir == (Path.home() / ".cache" / "quantfunc" / "engine").resolve()

    xdg_dir = tmp_path / "xdg_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_dir))
    if os.name == "posix":
        assert Path(installer.default_engine_dir()) == (xdg_dir / "quantfunc" / "engine").resolve()

    custom = tmp_path / "custom_engine"
    monkeypatch.setenv("QF_ENGINE_DIR", str(custom))
    assert Path(installer.default_engine_dir()) == custom.resolve()


# ── Test 5: Happy path installer POSIX + Windows mock branch ─────────────────


def test_installer_atomic_happy_path_posix_and_win(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    routes, _, _ = _make_mock_release(
        version="0.0.18",
        cuda=13,
        gpu_set="sm89",
        sms=[89],
        version_entry_extra={"sdk": "0.0.08"},
        base_url=base_url,
    )
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes[url]))

    # Track os.open calls to verify O_CREAT|O_EXCL, mode 0o600, and random part suffix
    open_calls: list[tuple[str, int, int]] = []
    real_open = os.open

    def spy_open(path: Any, flags: int, mode: int = 0o777, *args: Any, **kwargs: Any) -> int:
        open_calls.append((str(path), flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)

    eng_dir = tmp_path / "engine_cache"
    res = installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    assert res is not None
    assert res["version"] == "0.0.18"
    assert installer.has_ready_so(engine_dir=eng_dir) is True

    part_opens = [(p, f, m) for p, f, m in open_calls if ".part-" in os.path.basename(p)]
    assert len(part_opens) >= 2
    for p, f, m in part_opens:
        assert (f & os.O_CREAT) and (f & os.O_EXCL) and (f & os.O_WRONLY)
        assert m == 0o600
        # Random hex suffix after pid
        assert len(os.path.basename(p).rsplit("-", 1)[-1]) == 16

    resolved = installer.resolve_so_path(engine_dir=eng_dir)
    assert os.path.isfile(resolved)
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(resolved).st_mode) == 0o755

    # Windows mock branch: os.name == "nt", no O_NOFOLLOW, no fchmod, O_BINARY present
    monkeypatch.setattr(os, "open", real_open)
    monkeypatch.setattr(installer.os, "name", "nt")
    monkeypatch.delattr(installer.os, "O_NOFOLLOW", raising=False)
    monkeypatch.delattr(installer.os, "fchmod", raising=False)
    monkeypatch.setattr(installer.os, "O_BINARY", 0, raising=False)
    win_target = tmp_path / "win_atomic.json"
    installer._engine_write_file(str(win_target), b'{"ok":true}')
    assert win_target.read_bytes() == b'{"ok":true}'


# ── Test 6: Installer security refusals & rollback ───────────────────────────


def test_installer_security_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strict URL validation refusals
    for bad_url in (
        "http://service.quantfunc.com",
        "https://user:pass@service.quantfunc.com",
        "https://user@service.quantfunc.com",
        "https://service.quantfunc.com?foo=bar",
        "https://service.quantfunc.com#frag",
        "https:///no-host",
        "ftp://service.quantfunc.com",
        "https://service.quantfunc.com\nX-Injected: 1",
    ):
        with pytest.raises(ValueError):
            installer.validate_https_url(bad_url)

    assert installer.validate_https_url("https://service.quantfunc.com") == "https://service.quantfunc.com"

    # HTTP redirect downgrade refused before plaintext request
    handler = installer._HttpsOnlyRedirectHandler()
    req = installer.urllib.request.Request("https://releases.quantfunc.example/a")
    with pytest.raises(RuntimeError, match="non-HTTPS"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example/a")
    with pytest.raises(RuntimeError, match="userinfo"):
        handler.redirect_request(req, None, 302, "Found", {}, "https://u:p@evil.example/a")

    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    # Symlinked engine directory refused
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    sym_dir = tmp_path / "sym_engine"
    sym_dir.symlink_to(real_target)
    with pytest.raises(RuntimeError, match="symlink"):
        installer.install_engine(engine_dir=sym_dir, base_url=base_url)

    # SHA-256 tamper refused with zero leftovers
    routes, _, _ = _make_mock_release(base_url=base_url)
    tampered = dict(routes)
    tampered[f"{base_url}/0.0.18/linux/libquantfunc.so"] = b"corrupted"
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, tampered[url]))
    d_tamper = tmp_path / "tamper"
    with pytest.raises(RuntimeError, match="SHA-256"):
        installer.install_engine(engine_dir=d_tamper, base_url=base_url)
    assert not installer.has_ready_so(engine_dir=d_tamper)

    # Host ELF importing libcudart.so.12 refused
    routes_cuda_host, _, _ = _make_mock_release(host_needed=["libc.so.6", "libcudart.so.12"], base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes_cuda_host[url]))
    with pytest.raises(RuntimeError, match="zero CUDA"):
        installer.install_engine(engine_dir=tmp_path / "cuda_host", base_url=base_url)

    # Mismatched .qf_pair_id refused
    routes_pair_bad, _, _ = _make_mock_release(
        pair_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        kernel_pair_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        base_url=base_url,
    )
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes_pair_bad[url]))
    with pytest.raises(RuntimeError, match="not from one build"):
        installer.install_engine(engine_dir=tmp_path / "pair_bad", base_url=base_url)

    # PyTorch CUDA mismatch refused in assert_toolchain_compatible
    ok_dir = tmp_path / "tc_mismatch"
    ok_dir.mkdir()
    host_p = ok_dir / "libquantfunc.so"
    kern_p = ok_dir / "libquantfunc_kernels.so"
    host_p.write_bytes(_build_synthetic_elf(["libc.so.6"]))
    kern_p.write_bytes(_build_synthetic_elf(["libcudart.so.12"]))
    monkeypatch.delenv("QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN", raising=False)
    monkeypatch.delenv("QF_KERNEL_PLUGIN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="CUDA toolchain mismatch"):
        installer.assert_toolchain_compatible(host_p)


# ── Test 7: QF_KERNEL_PLUGIN_DIR cross-platform + env overrides ──────────────


def test_plugin_dir_posix_win_and_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # QF_NATIVE_SO_PATH missing file fails closed
    monkeypatch.setenv("QF_NATIVE_SO_PATH", str(tmp_path / "nonexistent.so"))
    assert installer.has_ready_so() is False
    with pytest.raises(RuntimeError, match="not an existing file"):
        installer.resolve_so_path()
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    # POSIX plugin dir validation
    pdir = tmp_path / "plugins_posix"
    pdir.mkdir(mode=0o755)
    os.chmod(pdir, 0o755)
    kern_so = pdir / "libquantfunc_kernels.so"
    kern_so.write_bytes(_build_synthetic_elf(["libcudart.so.13"]))
    os.chmod(kern_so, 0o755)
    assert installer.validate_plugin_dir(pdir) == str(pdir.resolve())

    # Reject world-writable dir (0o777) and group-writable file (0o775)
    os.chmod(pdir, 0o777)
    with pytest.raises(RuntimeError, match="writable"):
        installer.validate_plugin_dir(pdir)
    os.chmod(pdir, 0o755)

    os.chmod(kern_so, 0o775)
    with pytest.raises(RuntimeError, match="writable"):
        installer.validate_plugin_dir(pdir)
    os.chmod(kern_so, 0o755)

    # Reject symlinked .so
    sym_so = pdir / "libquantfunc_kernels-12.so"
    sym_so.symlink_to(kern_so)
    with pytest.raises(RuntimeError, match="symlink"):
        installer.validate_plugin_dir(pdir)
    sym_so.unlink()

    # Reject untrusted owner uid
    real_lstat = os.lstat

    def fake_untrusted_lstat(p: Any) -> Any:
        st = real_lstat(p)
        if os.path.basename(str(p)) == "libquantfunc_kernels.so":
            return os.stat_result(
                (st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 65534, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime)
            )
        return st

    monkeypatch.setattr(os, "lstat", fake_untrusted_lstat)
    with pytest.raises(RuntimeError, match="untrusted uid"):
        installer.validate_plugin_dir(pdir)
    monkeypatch.setattr(os, "lstat", real_lstat)

    # Windows mock branch: reject symlinked .dll and host DLL importing CUDA
    wdir = tmp_path / "plugins_win"
    wdir.mkdir()
    good_kdll = wdir / "libquantfunc_kernels.dll"
    good_kdll.write_bytes(_build_synthetic_pe(["kernel32.dll", "cudart64_13.dll"]))
    monkeypatch.setattr(installer.os, "name", "nt")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    assert installer.validate_plugin_dir(wdir) == str(wdir.resolve())

    # Symlink .dll rejected on Windows branch
    sym_dll = wdir / "extra.dll"
    sym_dll.symlink_to(good_kdll)
    with pytest.raises(RuntimeError, match="symlink"):
        installer.validate_plugin_dir(wdir)
    sym_dll.unlink()

    # Non-kernel DLL importing CUDA rejected on Windows branch
    bad_host_dll = wdir / "quantfunc.dll"
    bad_host_dll.write_bytes(_build_synthetic_pe(["kernel32.dll", "cudart64_12.dll"]))
    with pytest.raises(RuntimeError, match="forbidden CUDA PE imports"):
        installer.validate_plugin_dir(wdir)
    bad_host_dll.unlink()

    # QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 bypasses version mismatch check
    monkeypatch.setattr(installer.os, "name", "posix")
    host_so = tmp_path / "libquantfunc.so"
    host_so.write_bytes(_build_synthetic_elf(["libc.so.6"]))
    (tmp_path / "libquantfunc_kernels.so").write_bytes(_build_synthetic_elf(["libcudart.so.12"]))
    monkeypatch.delenv("QF_KERNEL_PLUGIN_DIR", raising=False)
    monkeypatch.setenv("QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN", "1")
    installer.assert_toolchain_compatible(host_so)  # does not raise


# ── Test 8: PE guard, Windows add_dll_directory, Linux dlinfo, QF_LOG_LEVEL ──


def test_pe_guard_windows_dll_dir_dlinfo_and_log_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Synthetic PE with forbidden cudart64_12.dll rejected
    bad_pe = tmp_path / "quantfunc.dll"
    bad_pe.write_bytes(_build_synthetic_pe(["kernel32.dll", "cudart64_12.dll"]))
    assert "cudart64_12.dll" in installer.pe_imports(bad_pe)
    monkeypatch.delenv("QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN", raising=False)
    monkeypatch.delenv("QF_KERNEL_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 12)
    with pytest.raises(RuntimeError, match="forbidden CUDA PE imports"):
        installer.assert_toolchain_compatible(bad_pe)

    # Windows os.add_dll_directory called and persisted in _DLL_DIR_HANDLES
    added_dirs: list[str] = []

    class _DummyHandle:
        pass

    monkeypatch.setattr(loader, "_IS_WINDOWS", True)
    monkeypatch.setattr(os, "add_dll_directory", lambda d: (added_dirs.append(d), _DummyHandle())[1], raising=False)
    before_count = len(loader._DLL_DIR_HANDLES)
    loader._register_windows_dll_dirs(str(bad_pe))
    assert str(tmp_path.resolve()) in added_dirs
    assert len(loader._DLL_DIR_HANDLES) == before_count + 1

    # Linux dlinfo guard refuses external non-torch CUDA soname
    with pytest.raises(RuntimeError, match="outside PyTorch's CUDA library set"):
        loader._assert_torch_cuda_family(
            provided={"libcublas.so.13"},
            dirs={"/opt/torch/lib"},
            linker=lambda name: "/usr/local/cuda-12/lib64/libcublas.so.13",
        )
    # Matching torch directory passes cleanly
    loader._assert_torch_cuda_family(
        provided={"libcublas.so.13"},
        dirs={"/opt/torch/lib"},
        linker=lambda name: "/opt/torch/lib/libcublas.so.13",
    )

    # set_log_level / get_log_level / set_log_stderr never trigger library loading
    monkeypatch.setattr(loader, "_LIB", None)
    monkeypatch.setattr(loader, "_LOG_LEVEL", None)
    loader.set_log_level(2)
    loader.set_log_stderr(1)
    assert loader._LIB is None
    assert loader.get_log_level() == 2


# ── Test 9: 12/19 allowlists, 25/11 rejections, 256KiB / 1MiB / depth 16 ─────


def test_allowlists_legacy_rejection_and_json_bounds(fake_qf_lib: FakeQFLib) -> None:
    assert len(handle.CREATE_CONFIG_ALLOWLIST) == 12
    assert len(handle.FORBIDDEN_CREATE_KEYS) == 25
    assert len(handle.SESSION_OPTIONS_ALLOWLIST) == 19
    assert len(handle.REJECTED_LEGACY_OPTION_KEYS) == 11
    assert WIRE_QUALITY_RATIO_KEY in handle.FORBIDDEN_CREATE_KEYS
    assert WIRE_QUALITY_RATIO_KEY in handle.SESSION_OPTIONS_ALLOWLIST

    # All 12 valid create keys pass
    valid_create = {
        "model_arch": "krea2",
        "config_path": "/tmp/config.json",
        "connectors_path": "",
        "attention_backend": "auto",
        "offload_mode": "auto",
        "pinned_memory": False,
        "vram_budget_mb": 0,
        "act_scale_g32": False,
        "glu_fuse": True,
        "server_url": "https://service.quantfunc.com",
        "api_key": "",
        "cache_dir": "/tmp/cache",
    }
    encoded = handle.validate_create_config(valid_create)
    assert encoded is not None

    # Every one of the 25 FORBIDDEN_CREATE_KEYS is rejected even when nested inside dict/list
    for bad_key in handle.FORBIDDEN_CREATE_KEYS:
        with pytest.raises(errors.InvalidArgError):
            handle.validate_create_config({bad_key: 1})
        with pytest.raises(errors.InvalidArgError):
            handle.validate_create_config({"model_arch": [{"nested": {bad_key: 1}}]})

    # Unknown create key rejected
    with pytest.raises(errors.InvalidArgError):
        handle.validate_create_config({"unknown_key": True})

    # All 19 valid session keys pass
    valid_session = {
        "attention_backend": "int8",
        "step_cache_thresh": 0.1,
        "block_cache_thresh": 0.05,
        WIRE_QUALITY_RATIO_KEY: 0.8,
        "cat_external_signal": False,
        "sparse": {"enabled": True, "selector": "svg2", "tau": 1.3, "min_tokens": 4096},
        "num_frames": 17,
        "fps": 24.0,
        "audio_dims": [1, 32, 2, 10],
        "av_sigma_shift_video": 12.0,
        "av_sigma_shift_audio": 3.0,
        "av_unprocessed_ctx": False,
        "av_conds": [],
        "text_vision_row_ranges": [[0, 4]],
        "frame_t_scales": [1.0],
        "audio_frame_t_scales": [1.0],
        "stg_blocks": [0, 1],
        "disable_av_cross_attn": False,
        "idle_timeout_ms": 60000,
    }
    assert handle.validate_session_options(valid_session) is not None

    # Every one of the 11 REJECTED_LEGACY_OPTION_KEYS is rejected
    for leg_key in handle.REJECTED_LEGACY_OPTION_KEYS:
        with pytest.raises(errors.InvalidArgError):
            handle.validate_session_options({leg_key: True})

    # Size bounds: create > 256 KiB, session > 1 MiB
    oversize_create = {"config_path": "a" * (handle.MAX_CREATE_JSON_BYTES + 16)}
    with pytest.raises(errors.InvalidArgError, match="byte size"):
        handle.validate_create_config(oversize_create)

    oversize_session = {"av_conds": ["x" * (handle.MAX_SESSION_JSON_BYTES + 16)]}
    with pytest.raises(errors.InvalidArgError, match="byte size"):
        handle.validate_session_options(oversize_session)

    # Depth bound: depth 17 rejected for both create and session
    deep: Any = "leaf"
    for _ in range(16):
        deep = [deep]
    with pytest.raises(errors.InvalidArgError, match="depth"):
        handle.validate_create_config({"model_arch": deep})
    with pytest.raises(errors.InvalidArgError, match="depth"):
        handle.validate_session_options({"av_conds": deep})

    # Verify QFEngineHandle.session_begin enforces validation before calling C-API
    eng = handle.QFEngineHandle.create(fake_qf_lib, "/fake/model.safetensors", {"model_arch": "ltx25"})
    with pytest.raises(errors.InvalidArgError):
        eng.session_begin(512, 512, 4, (1, 16, 64), capi.QF_DTYPE_BF16, {"audio_enhance": True})
    assert len(fake_qf_lib.sessions) == 0
    eng.destroy()


# ── Test 10: QFEngineHandle VRAM helpers & ref-audio lifecycle ───────────────


def test_handle_vram_helpers_and_ref_audio_lifecycle(fake_qf_lib: FakeQFLib) -> None:
    params = handle.make_create_params(weights_path="/fake/m.safetensors", device_id=0, config_json={"model_arch": "ltx25"})
    cold = handle.estimate_vram(fake_qf_lib, params)
    assert cold.resident_weight_bytes == fake_qf_lib.estimate_resident_bytes
    assert cold.effective_model_bytes(total_vram_bytes=24 << 30) == fake_qf_lib.estimate_resident_bytes
    # Tight VRAM forces offload estimate below full resident bytes
    tight = cold.effective_model_bytes(total_vram_bytes=6 << 30)
    assert cold.min_offload_bytes is not None
    assert cold.min_offload_bytes <= tight < fake_qf_lib.estimate_resident_bytes

    eng = handle.QFEngineHandle.create(fake_qf_lib, "/fake/m.safetensors", {"model_arch": "ltx25"})
    assert eng.resident_vram_bytes() == fake_qf_lib.initial_resident_bytes
    assert eng.vram_need_bytes([1, 16, 4, 8, 8]) > 0

    with pytest.raises(ValueError):
        eng.unload_vram(-1)
    with pytest.raises(ValueError):
        eng.unload_vram(capi.QF_UINT64_MAX + 1)

    freed = eng.unload_vram(1 << 30)
    assert freed == (1 << 30)
    assert eng.resident_vram_bytes() == fake_qf_lib.initial_resident_bytes - (1 << 30)

    # Reference-audio borrow prevents destroy until unload_sync() completes
    owner = object()
    eng.retain_reference_audio(owner)
    assert eng.reference_audio_pending() is True
    with pytest.raises(RuntimeError, match="pending"):
        eng.destroy()
    assert eng.xfm is not None  # handle still alive

    assert eng.unload_sync() == capi.QF_OK
    assert eng.reference_audio_pending() is False
    eng.destroy()
    assert eng.xfm is None
    assert len(fake_qf_lib.destroyed) == 1


# ── Test 11 (Mutation M1): Breaking L0 security guards must fail assertions ──


def test_mutation_l0_security_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # M1a: mutating host CUDA guard to ignore libcudart must fail the security check
    host_bad = tmp_path / "libquantfunc.so"
    kern_ok = tmp_path / "libquantfunc_kernels.so"
    host_bad.write_bytes(_build_synthetic_elf(["libc.so.6", "libcudart.so.13"]))
    kern_ok.write_bytes(_build_synthetic_elf(["libcudart.so.13"]))
    monkeypatch.delenv("QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN", raising=False)
    monkeypatch.delenv("QF_KERNEL_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)

    # Baseline passes (security guard rejects bad host)
    with pytest.raises(RuntimeError, match="forbidden CUDA"):
        installer.assert_toolchain_compatible(host_bad)

    # Mutate: disable _host_forbidden_cuda -> guard stops rejecting!
    monkeypatch.setattr(installer, "_host_forbidden_cuda", lambda p: [])
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(RuntimeError, match="forbidden CUDA"):
            installer.assert_toolchain_compatible(host_bad)

    # M1b: mutating validate_https_url to accept http:// breaks security assertion
    with pytest.raises(ValueError):
        installer.validate_https_url("http://insecure.example")
    monkeypatch.setattr(installer, "validate_https_url", lambda u: u)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(ValueError):
            installer.validate_https_url("http://insecure.example")

    # M1c: mutating validate_plugin_dir to skip symlink check breaks symlink assertion
    real_d = tmp_path / "real_p"
    real_d.mkdir(mode=0o755)
    sym_d = tmp_path / "sym_p"
    sym_d.symlink_to(real_d)
    with pytest.raises(RuntimeError, match="symlink"):
        installer.validate_plugin_dir(sym_d)
    monkeypatch.setattr(installer, "validate_plugin_dir", lambda p: str(p))
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        with pytest.raises(RuntimeError, match="symlink"):
            installer.validate_plugin_dir(sym_d)


# ── Engine Update Parity Tests (ComfyUI-QuantFunc qf_engine.py alignment) ────


def test_engine_pick_version_sdk_comfy_and_cu12(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "0.0.15": {"kernel_so": True, "comfy": "0.0.07", "lib": "0.0.15"},
        "0.0.18": {"kernel_so": True, "comfy": "0.0.08", "comfy-12": "0.0.08", "lib": "0.0.18", "lib-12": "0.0.18"},
        "0.0.20": {"kernel_so": True, "sdk": "0.0.09", "comfy": "0.0.08", "lib": "0.0.20"},
        "0.0.21": {"kernel_so": False, "sdk": "0.0.08", "lib": "0.0.21"},
    }
    # At compat 0.0.08, 0.0.20 requires sdk 0.0.09 so 0.0.18 wins; 0.0.21 lacks kernel_so
    assert installer._engine_pick_version(versions, "0.0.08", 13) == "0.0.18"
    assert installer._engine_pick_version(versions, "0.0.08", 12) == "0.0.18"
    # At compat 0.0.09 via QF_ENGINE_COMPAT_VERSION, 0.0.20 wins
    monkeypatch.setenv("QF_ENGINE_COMPAT_VERSION", "0.0.09")
    assert installer._engine_compat_version() == "0.0.09"
    assert installer._engine_pick_version(versions, installer._engine_compat_version(), 13) == "0.0.20"


def test_claim_strips_superseded_sms_and_removes_old_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    eng_dir = tmp_path / "claim_test"
    # Install v0.0.17 claiming [86, 89] under gpu_set "legacy"
    r17, _, _ = _make_mock_release(version="0.0.17", gpu_set="legacy", sms=[86, 89], base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r17[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    bin_dir = eng_dir / "linux"
    assert (bin_dir / ".engine-legacy-cu13.json").is_file()

    # Now install v0.0.18 claiming [89] under gpu_set "ada" -> legacy marker should be stripped to [86]
    r18, _, _ = _make_mock_release(version="0.0.18", gpu_set="ada", sms=[89], base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r18[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    legacy_m = json.loads((bin_dir / ".engine-legacy-cu13.json").read_text(encoding="utf-8"))
    assert legacy_m["sms"] == [86]

    # Now install v0.0.19 claiming [86, 89] under "ada" -> legacy marker + dir completely removed, old 0.0.18 kept as previous until next upgrade
    r19, _, _ = _make_mock_release(version="0.0.19", gpu_set="ada", sms=[86, 89], base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r19[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    assert not (bin_dir / ".engine-legacy-cu13.json").exists()
    assert not (bin_dir / "0.0.17-legacy-cu13").exists()

    # Install v0.0.20 under "ada" -> 0.0.18-ada-cu13 must now be removed (keep only 0.0.20 and previous 0.0.19)
    r20, _, _ = _make_mock_release(version="0.0.20", gpu_set="ada", sms=[86, 89], base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r20[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    assert (bin_dir / "0.0.20-ada-cu13").is_dir()
    assert (bin_dir / "0.0.19-ada-cu13").is_dir()
    assert not (bin_dir / "0.0.18-ada-cu13").exists()


def test_same_version_sha256_drift_redownload_and_atomic_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    eng_dir = tmp_path / "drift"
    r1, host1, kern1 = _make_mock_release(
        version="0.0.18",
        pair_id="11111111111111111111111111111111",
        base_url=base_url,
    )
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r1[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    host_path = Path(installer.resolve_so_path(engine_dir=eng_dir))
    assert host_path.read_bytes() == host1

    # Same version 0.0.18 republishes with new SHA-256, but replace fails mid-way -> rollback preserves host1!
    r2, host2, kern2 = _make_mock_release(
        version="0.0.18",
        pair_id="22222222222222222222222222222222",
        base_url=base_url,
    )
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r2[url]))
    real_replace = os.replace
    replace_calls = {"n": 0}

    def fail_on_host_replace(src: Any, dst: Any) -> None:
        replace_calls["n"] += 1
        if str(dst).endswith("libquantfunc.so"):
            raise OSError("simulated disk full during host replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_host_replace)
    with pytest.raises(RuntimeError, match="existing installation preserved"):
        installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    monkeypatch.setattr(os, "replace", real_replace)

    # Original pair is completely intact after rollback!
    assert host_path.read_bytes() == host1
    assert (host_path.parent / "libquantfunc_kernels.so").read_bytes() == kern1

    # Now allow replace to succeed -> same-version SHA-256 drift upgrades cleanly to host2/kern2
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    assert host_path.read_bytes() == host2
    assert (host_path.parent / "libquantfunc_kernels.so").read_bytes() == kern2


def test_install_addons_and_reserved_name_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    addon_data = b"ELF-addon-payload"
    routes, host_bytes, _ = _make_mock_release(
        addons_bytes={"libextra_helper.so": addon_data, "libquantfunc.so": b"evil-overwrite"},
        base_url=base_url,
    )
    # Restore genuine host bytes in routes so host passes ELF check
    routes[f"{base_url}/0.0.18/linux/libquantfunc.so"] = host_bytes
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes[url]))

    eng_dir = tmp_path / "addons_dir"
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    pair_dir = Path(installer.resolve_so_path(engine_dir=eng_dir)).parent
    assert (pair_dir / "libextra_helper.so").read_bytes() == addon_data
    # Reserved host name was never overwritten by addon installer
    assert (pair_dir / "libquantfunc.so").read_bytes() == host_bytes


def test_reinstalled_oneshot_self_heal_and_disk_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    routes, _, _ = _make_mock_release(base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes[url]))

    eng_dir = tmp_path / "self_heal"
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    so_path = installer.resolve_so_path(engine_dir=eng_dir)

    # First CDLL load failure writes .reinstalled, removes marker, triggers one-shot reinstall
    reinstall_calls = {"n": 0}
    monkeypatch.setattr(installer, "start_engine_install", lambda **kw: reinstall_calls.__setitem__("n", reinstall_calls["n"] + 1))

    msg1 = installer._engine_load_failed(so_path, OSError("missing libcudart"), engine_dir=eng_dir)
    assert "re-downloaded once" in msg1
    assert reinstall_calls["n"] == 1
    flag = Path(so_path).parent / installer._ENGINE_REINSTALLED
    assert flag.is_file()

    # Re-create marker to simulate post-reinstall second failure -> .reinstalled stops infinite retry!
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    assert flag.is_file()
    msg2 = installer._engine_load_failed(so_path, OSError("still broken"), engine_dir=eng_dir)
    assert "not be re-downloaded again" in msg2
    assert reinstall_calls["n"] == 1  # NO second reinstall!

    # Successful load clears .reinstalled sentinel
    installer._engine_load_ok(so_path, engine_dir=eng_dir)
    assert not flag.exists()

    # Disk corruption triggers marker removal & self-heal refusal
    Path(so_path).write_bytes(b"tampered-on-disk")
    with pytest.raises(RuntimeError, match="SHA-256"):
        installer.resolve_so_path(engine_dir=eng_dir, trigger_self_heal=True)
    assert reinstall_calls["n"] == 2
    assert not installer.has_ready_so(engine_dir=eng_dir)


def test_check_engine_update_readonly_and_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    eng_dir = tmp_path / "update_check"
    r18, _, _ = _make_mock_release(version="0.0.18", base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r18[url]))

    # Before install: read-only check reports update available and creates no files
    info0 = installer.check_engine_update(engine_dir=eng_dir, base_url=base_url)
    assert info0.installed_version is None
    assert info0.latest_version == "0.0.18"
    assert info0.update_available is True
    assert not eng_dir.exists()

    # After install: up to date
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)
    info1 = installer.check_engine_update(engine_dir=eng_dir, base_url=base_url)
    assert info1.installed_version == "0.0.18"
    assert info1.latest_version == "0.0.18"
    assert info1.update_available is False

    # Newer remote release -> update_available True
    r19, _, _ = _make_mock_release(version="0.0.19", base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r19[url]))
    info2 = installer.check_engine_update(engine_dir=eng_dir, base_url=base_url)
    assert info2.installed_version == "0.0.18"
    assert info2.latest_version == "0.0.19"
    assert info2.update_available is True


def test_start_engine_install_offline_keeps_existing_and_dev_lib_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    eng_dir = tmp_path / "offline_test"
    r18, _, _ = _make_mock_release(version="0.0.18", base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, r18[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)

    # Simulate network offline -> background start_engine_install sets status "offline" and keeps installed pair
    def _fail_net(url: str) -> Any:
        raise OSError("network unreachable")

    monkeypatch.setattr(installer, "_engine_http_open", _fail_net)
    t = installer.start_engine_install(device_idx=0, engine_dir=eng_dir, base_url=base_url)
    t.join(timeout=5.0)
    state, _detail = installer.engine_install_status()
    assert state == "offline"
    assert installer.has_ready_so(engine_dir=eng_dir) is True

    # .dev_lib_lock skips download with status "local"
    dev_dir = tmp_path / "dev_lock_dir"
    (dev_dir / "linux").mkdir(parents=True)
    (dev_dir / "linux" / ".dev_lib_lock").write_text("local", encoding="utf-8")
    assert installer.install_engine(engine_dir=dev_dir, base_url=base_url) is None
    assert installer.engine_install_status()[0] == "local"


def test_load_lib_auto_update_background_and_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_qf_lib: FakeQFLib,
) -> None:
    base_url = "https://releases.quantfunc.example/engine"
    monkeypatch.setattr(installer, "_BIN_SUBDIR", "linux")
    monkeypatch.setattr(installer, "_torch_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_driver_cuda_major", lambda: 13)
    monkeypatch.setattr(installer, "_gpu_sm", lambda idx=0: 89)
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("QF_NATIVE_SO_PATH", raising=False)

    eng_dir = tmp_path / "load_auto_upd"
    monkeypatch.setenv("QF_ENGINE_DIR", str(eng_dir))
    routes, _, _ = _make_mock_release(base_url=base_url)
    monkeypatch.setattr(installer, "_engine_http_open", lambda url: _FakeHttpResponse(url, routes[url]))
    installer.install_engine(engine_dir=eng_dir, base_url=base_url)

    bg_calls: list[Any] = []
    monkeypatch.setattr(loader, "_LIB", None)
    monkeypatch.setattr(loader, "_LIB_PATH", None)
    monkeypatch.setattr(loader, "_IS_WINDOWS", False)
    monkeypatch.setattr(loader, "assert_toolchain_compatible", lambda p: None)
    monkeypatch.setattr(loader, "_torch_cuda_plan", lambda *a: (set(), set(), []))
    monkeypatch.setattr(loader, "_sidecar_preloads", lambda p: [])
    monkeypatch.setattr(loader, "_assert_torch_cuda_family", lambda *a: None)
    monkeypatch.setattr(loader, "start_engine_install", lambda **kw: bg_calls.append(kw))
    monkeypatch.setattr(ctypes, "CDLL", lambda path, **kw: fake_qf_lib)

    # QF_AUTO_UPDATE=0 disables background update check
    monkeypatch.setenv("QF_AUTO_UPDATE", "0")
    loaded = loader.load_lib(device_id=0)
    assert loaded is fake_qf_lib
    assert bg_calls == []

    # QF_AUTO_UPDATE=1 spawns non-blocking background update check after loading existing intact pair
    monkeypatch.setattr(loader, "_LIB", None)
    monkeypatch.setattr(loader, "_LIB_PATH", None)
    monkeypatch.setenv("QF_AUTO_UPDATE", "1")
    loaded2 = loader.load_lib(device_id=0)
    assert loaded2 is fake_qf_lib
    assert len(bg_calls) == 1
