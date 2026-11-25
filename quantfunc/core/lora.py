"""Pure-stdlib LoRA format detection, canonical safetensors conversion, and cached LoRAManager.

100% model-agnostic; stdlib + L0/L1 config only (zero torch, zero numpy, zero model names).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .config import default_cache_dir

_MAX_HEADER_BYTES = 512 * 1024 * 1024

# Multi-token module names whose internal underscores must survive Kohya dot->underscore flattening.
_PROTECTED_TOKENS = sorted(
    {
        "single_transformer_blocks",
        "transformer_blocks",
        "layerwise_blocks",
        "refiner_blocks",
        "token_refiner",
        "text_fusion",
        "self_attn",
        "cross_attn",
        "audio_attn1",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
        "audio_ff",
        "to_gate_logits",
        "to_gate",
        "to_out",
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "add_q",
        "add_k",
        "add_v",
        "feed_forward",
        "time_mod_proj",
        "time_embed",
        "final_layer",
        "audio_patch_proj",
        "audio_proj_in",
        "audio_time_embed",
        "norm_out",
        "proj_out",
        "proj_in",
        "img_mlp",
        "txt_mlp",
        "img_mod",
        "txt_mod",
        "img_in",
        "txt_in",
        "linear_1",
        "linear_2",
        "norm_q",
        "norm_k",
        "norm1",
        "norm2",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "audio_prompt_adaln_single",
        "audio_adaln_single",
        "prompt_adaln_single",
        "adaln_single",
        "timestep_embedder",
        "audio_patchify_proj",
        "patchify_proj",
        "audio_proj_out",
        "caption_projection",
    },
    key=len,
    reverse=True,
)

_ROOT_PREFIXES = (
    "diffusion_model.",
    "transformer.",
    "lora_unet.",
    "unet.",
    "base_model.model.",
    "model.",
)

# BFL / ai-toolkit community LoRA naming -> canonical diffusers module names.
_BFL_ROOT_MAP = {
    "first": "img_in",
    "last.linear": "final_layer.linear",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
}
_BFL_TAIL_MAP = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0",
    "attn.gate": "attn.to_gate",
    "mlp.gate": "ff.gate",
    "mlp.up": "ff.up",
    "mlp.down": "ff.down",
}
_BFL_MARKERS = ("txtfusion.", "tmlp.", "txtmlp.", "tproj.")
_BFL_BLOCK_RE = re.compile(r"(blocks|txtfusion\.(?:layerwise|refiner)_blocks)\.(\d+)\.(.+)$")

_TE_PREFIXES = ("lora_te_", "lora_te1_", "lora_te2_")
_DIFFUSERS_TE_RE = re.compile(r"text_encoder(_\d+)?\.")
_PEFT_ADAPTER_RE = re.compile(r"(.+)\.(lora_A|lora_B|lora_down|lora_up)\.(?!weight$)[^.]+\.weight$")


@dataclass(frozen=True)
class LoRAEntry:
    """Single LoRA adapter specification."""

    path: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError(f"LoRAEntry.path must be a non-empty string, got {self.path!r}")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int | float):
            raise ValueError(f"LoRAEntry.scale must be a finite number, got {self.scale!r}")
        sc = float(self.scale)
        if not math.isfinite(sc):
            raise ValueError(f"LoRAEntry.scale must be finite, got {sc!r}")
        object.__setattr__(self, "scale", sc)


def _read_safetensors_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"Truncated safetensors header in {path!r}")
        n = struct.unpack("<Q", raw_len)[0]
        if n == 0 or n > _MAX_HEADER_BYTES:
            raise ValueError(f"Implausible safetensors header size {n} in {path!r}")
        raw_hdr = fh.read(n)
        if len(raw_hdr) < n:
            raise ValueError(f"Truncated safetensors header JSON in {path!r}")
    hdr = json.loads(raw_hdr.decode("utf-8"))
    if not isinstance(hdr, dict):
        raise ValueError(f"Safetensors header in {path!r} must be a JSON object")
    return hdr


def _read_safetensors(path: str) -> tuple[dict[str, Any], bytes]:
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"Truncated safetensors header in {path!r}")
        n = struct.unpack("<Q", raw_len)[0]
        if n == 0 or n > _MAX_HEADER_BYTES:
            raise ValueError(f"Implausible safetensors header size {n} in {path!r}")
        raw_hdr = fh.read(n)
        if len(raw_hdr) < n:
            raise ValueError(f"Truncated safetensors header JSON in {path!r}")
        hdr = json.loads(raw_hdr.decode("utf-8"))
        if not isinstance(hdr, dict):
            raise ValueError(f"Safetensors header in {path!r} must be a JSON object")
        blob = fh.read()
    return hdr, blob


def _write_safetensors_atomic(
    path: str,
    ordered_tensors: list[tuple[str, dict[str, Any]]],
    blob_src: bytes,
    metadata: dict[str, str],
) -> None:
    new_hdr: dict[str, Any] = {}
    if metadata:
        new_hdr["__metadata__"] = metadata
    out = bytearray()
    off = 0
    for name, info in ordered_tensors:
        b0, b1 = info["data_offsets"]
        data = blob_src[b0:b1]
        new_hdr[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [off, off + len(data)],
        }
        out += data
        off += len(data)
    hj = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
    hj += b" " * ((8 - (len(hj) % 8)) % 8)

    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(parent, f".{os.path.basename(path)}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(tmp_path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fd = -1
            fh.write(struct.pack("<Q", len(hj)))
            fh.write(hj)
            fh.write(bytes(out))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _kohya_join(underscored: str) -> str:
    s = underscored
    holds: dict[str, str] = {}
    for i, w in enumerate(_PROTECTED_TOKENS):
        tok = f"\x00{i}\x00"
        j = 0
        while True:
            k = s.find(w, j)
            if k < 0:
                break
            before_ok = (k == 0) or (s[k - 1] in "_.\x00")
            e = k + len(w)
            after_ok = (e == len(s)) or (s[e] in "_.\x00")
            if before_ok and after_ok:
                s = s[:k] + tok + s[e:]
                j = k + len(tok)
            else:
                j = k + 1
        holds[tok] = w
    s = s.replace("_", ".")
    for tok, w in holds.items():
        s = s.replace(tok, w)
    return s


def _build_model_inverse(model_path: str) -> dict[str, str]:
    hdr = _read_safetensors_header(model_path)
    mods: set[str] = set()
    for k in hdr:
        if k == "__metadata__" or "." not in k:
            continue
        mods.add(k.rsplit(".", 1)[0])
    inv: dict[str, str] = {}
    clash: set[str] = set()
    for m in mods:
        flat = m.replace(".", "_")
        if flat in inv and inv[flat] != m:
            clash.add(flat)
        inv[flat] = m
    for c in clash:
        del inv[c]
    return inv


def _strip_root(path: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _ROOT_PREFIXES:
            if path.startswith(p):
                path = path[len(p) :]
                changed = True
    return path


def _bfl_to_engine_body(body: str) -> str:
    rest = _strip_root(body)
    root = body[: len(body) - len(rest)]
    if rest in _BFL_ROOT_MAP:
        return root + _BFL_ROOT_MAP[rest]
    m = _BFL_BLOCK_RE.match(rest)
    if not m:
        return body
    grp = "transformer_blocks" if m.group(1) == "blocks" else "text_fusion." + m.group(1)[len("txtfusion.") :]
    tail = _BFL_TAIL_MAP.get(m.group(3), m.group(3))
    return f"{root}{grp}.{m.group(2)}.{tail}"


def _is_text_encoder_key(key: str) -> bool:
    return key.startswith(_TE_PREFIXES) or bool(_DIFFUSERS_TE_RE.match(key))


def unsupported_lora_kind(keys: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(kind, reason)`` if ``keys`` represent an unsupported adapter family (LyCORIS/DoRA/OFT), else ``None``."""
    for k in keys:
        if ".hada_" in k or ".lokr_" in k:
            return ("LyCORIS (LoHa/LoKr)", "factored decomposition cannot be converted to additive (A, B) LoRA")
        if ".dora_scale" in k or "lora_magnitude_vector" in k:
            return ("DoRA", "magnitude vectors rescale merged weights and cannot be dropped safely")
        if ".oft_" in k or ".boft_" in k:
            return ("OFT/BOFT", "multiplicative rotation is not an additive (A, B) LoRA delta")
    return None


def detect_lora_format(keys: Iterable[str]) -> str:
    """Detect LoRA checkpoint format: ``'lycoris'`` | ``'kohya'`` | ``'diffusers'`` | ``'unknown'``."""
    ky = list(keys)
    if any(".hada_" in k or ".lokr_" in k for k in ky):
        return "lycoris"
    if any(k.startswith(("lora_unet_", "lora_transformer_", "lora_te_", "lora_te1_", "lora_te2_")) for k in ky):
        return "kohya"
    if any(
        ".lora_A." in k
        or ".lora_B." in k
        or ".lora_down." in k
        or ".lora_up." in k
        or ".lora_linear_layer." in k
        for k in ky
    ):
        return "diffusers"
    return "unknown"


def _canon_leaf(role: str) -> str | None:
    if role in ("lora_down", "lora_A", "lora_linear_layer.down"):
        return "lora_A"
    if role in ("lora_up", "lora_B", "lora_linear_layer.up"):
        return "lora_B"
    if role == "alpha":
        return "alpha"
    return None


def _split_role(key: str) -> tuple[str | None, str | None]:
    for suf, role in (
        (".lora_down.weight", "lora_down"),
        (".lora_up.weight", "lora_up"),
        (".lora_A.weight", "lora_A"),
        (".lora_B.weight", "lora_B"),
        (".lora_linear_layer.down.weight", "lora_linear_layer.down"),
        (".lora_linear_layer.up.weight", "lora_linear_layer.up"),
        (".alpha", "alpha"),
    ):
        if key.endswith(suf):
            return key[: -len(suf)], role
    m = _PEFT_ADAPTER_RE.match(key)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _convert_key(orig: str, src_fmt: str, model_inv: dict[str, str] | None = None) -> str | None:
    if orig == "__metadata__" or _is_text_encoder_key(orig):
        return None
    body, role = _split_role(orig)
    if body is None or role is None:
        return None
    if src_fmt == "kohya":
        for pre in ("lora_unet_", "lora_transformer_"):
            if body.startswith(pre):
                body = body[len(pre) :]
                break
        dotted = (model_inv or {}).get(body) or _kohya_join(body)
    else:
        dotted = body
    leaf = _canon_leaf(role)
    if leaf is None:
        return None
    if leaf == "alpha":
        return f"{dotted}.alpha"
    return f"{dotted}.{leaf}.weight"


def _convert_keys(
    keys: Sequence[str],
    fmt: str,
    model_inv: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str | None]], bool]:
    conv = [(k, _convert_key(k, fmt, model_inv)) for k in keys]
    has_bfl = any(
        nk is not None and _strip_root(_split_role(nk)[0] or "").startswith(_BFL_MARKERS)
        for _, nk in conv
    )
    if not has_bfl:
        return conv, False
    out: list[tuple[str, str | None]] = []
    for k, nk in conv:
        if nk is not None:
            body = _split_role(nk)[0] or ""
            nk = _bfl_to_engine_body(body) + nk[len(body) :]
        out.append((k, nk))
    return out, True


def _check_refusal(
    conv: Sequence[tuple[str, str | None]],
    meta: dict[str, Any],
    bfl_renamed: bool,
) -> str | None:
    lost = [k for k, nk in conv if nk is None and not _is_text_encoder_key(k)]
    if lost:
        return (
            f"{len(lost)} key(s) cannot be mapped to canonical LoRA form (first: {lost[0]}); "
            "refusing rather than silently dropping weights"
        )
    lam = meta.get("lora_adapter_metadata")
    if bfl_renamed and lam:
        try:
            cfg = json.loads(lam) if isinstance(lam, str) else lam
        except (ValueError, TypeError):
            cfg = None
        if not isinstance(cfg, dict) or any(
            v for field, v in cfg.items() if field.endswith(("rank_pattern", "alpha_pattern"))
        ):
            return (
                "lora_adapter_metadata contains per-module rank_pattern or alpha_pattern that would be "
                "orphaned by module-name conversion; refusing rather than silently altering LoRA strength"
            )
    return None


def convert_lora_file(
    in_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    """Convert a Kohya/ai-toolkit/PEFT/BFL LoRA safetensors file to canonical PEFT form (pure stdlib).

    Raises ``ValueError`` (fail-closed) on unsupported adapter kinds (LyCORIS/DoRA/OFT),
    unmappable keys, colliding target names, or orphaned ``alpha_pattern``/``rank_pattern``.
    Returns the resolved output path.
    """
    src = os.fspath(in_path)
    dst = os.fspath(out_path)
    model_inv = _build_model_inverse(os.fspath(model_path)) if model_path else None

    hdr, blob = _read_safetensors(src)
    meta = hdr.get("__metadata__", {})
    if not isinstance(meta, dict):
        meta = {}
    keys = [k for k in hdr if k != "__metadata__"]

    kind = unsupported_lora_kind(keys)
    if kind is not None:
        name, why = kind
        raise ValueError(
            f"Unsupported LoRA kind {name}: {why}. Re-export or merge to a plain (A, B) LoRA first."
        )

    fmt = detect_lora_format(keys)
    if fmt == "unknown":
        raise ValueError(
            f"Could not detect LoRA format in {src!r} (no Kohya lora_unet_* or diffusers .lora_A/.lora_down keys)"
        )

    conv, bfl_renamed = _convert_keys(keys, fmt, model_inv)
    why = _check_refusal(conv, meta, bfl_renamed)
    if why is not None:
        raise ValueError(f"Refusing LoRA conversion for {src!r}: {why}")

    out_tensors: list[tuple[str, dict[str, Any]]] = []
    seen: dict[str, str] = {}
    for orig_k, new_k in conv:
        if new_k is None:
            continue
        if new_k in seen:
            raise ValueError(
                f"Ambiguous LoRA conversion: {seen[new_k]!r} and {orig_k!r} both map to {new_k!r}"
            )
        seen[new_k] = orig_k
        out_tensors.append((new_k, hdr[orig_k]))

    if not out_tensors:
        raise ValueError(f"LoRA conversion produced zero transformer tensors for {src!r}")

    new_meta = {str(k): str(v) for k, v in meta.items()}
    suffix = ", BFL names normalized" if bfl_renamed else ""
    new_meta["quantfunc_lora_converted"] = f"from {os.path.basename(src)} ({fmt}{suffix})"
    _write_safetensors_atomic(dst, out_tensors, blob, new_meta)
    return os.path.abspath(dst)


def _needs_conversion(keys: Sequence[str], fmt: str) -> bool:
    if fmt == "kohya":
        return True
    for k in keys:
        if _is_text_encoder_key(k):
            return True
        if ".lora_down." in k or ".lora_up." in k or ".lora_linear_layer." in k:
            return True
        if _PEFT_ADAPTER_RE.match(k):
            return True
        body, _ = _split_role(k)
        if body is not None and _strip_root(body).startswith(_BFL_MARKERS):
            return True
    return False


class LoRAManager:
    """Manages LoRA validation, cached auto-conversion, session teardown, and live hot-swap."""

    def __init__(
        self,
        handle: Any,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._handle = handle
        self._cache_dir = (
            os.path.abspath(os.path.expanduser(os.fspath(cache_dir)))
            if cache_dir is not None
            else os.path.join(default_cache_dir(), "loras")
        )
        self._default_model_path = os.fspath(model_path) if model_path is not None else None
        self._active_loras: list[LoRAEntry] = []

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def active_loras(self) -> tuple[LoRAEntry, ...]:
        return tuple(self._active_loras)

    def _normalize_entries(
        self,
        entries: Sequence[LoRAEntry | tuple[str | os.PathLike[str], float] | str | os.PathLike[str]],
    ) -> list[LoRAEntry]:
        out: list[LoRAEntry] = []
        for item in entries:
            if isinstance(item, LoRAEntry):
                out.append(item)
            elif isinstance(item, str | os.PathLike):
                out.append(LoRAEntry(path=os.fspath(item), scale=1.0))
            elif isinstance(item, tuple) and len(item) == 2:
                out.append(LoRAEntry(path=os.fspath(item[0]), scale=float(item[1])))
            else:
                raise TypeError(f"Unsupported LoRA specification: {item!r}")
        return out

    def _prepare_path(self, raw_path: str, *, model_path: str | None = None) -> str:
        abs_path = os.path.realpath(os.path.expanduser(raw_path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"LoRA file not found: {raw_path!r}")
        hdr = _read_safetensors_header(abs_path)
        meta = hdr.get("__metadata__", {})
        keys = [k for k in hdr if k != "__metadata__"]

        kind = unsupported_lora_kind(keys)
        if kind is not None:
            name, why = kind
            raise ValueError(f"Unsupported LoRA kind {name}: {why}")

        fmt = detect_lora_format(keys)
        if fmt == "unknown":
            raise ValueError(f"Could not detect LoRA format in {raw_path!r}")

        if not _needs_conversion(keys, fmt):
            conv, bfl = _convert_keys(keys, fmt)
            why = _check_refusal(conv, meta if isinstance(meta, dict) else {}, bfl)
            if why is not None:
                raise ValueError(f"Refusing LoRA {raw_path!r}: {why}")
            return abs_path

        st = os.stat(abs_path)
        eff_model = os.path.realpath(model_path) if model_path else ""
        cache_key = hashlib.sha256(
            f"{abs_path}|{st.st_size}|{st.st_mtime_ns}|{eff_model}".encode()
        ).hexdigest()[:24]
        os.makedirs(self._cache_dir, exist_ok=True)
        cached_path = os.path.join(self._cache_dir, f"{cache_key}.safetensors")
        if os.path.isfile(cached_path):
            return cached_path
        return convert_lora_file(abs_path, cached_path, model_path=eff_model or None)

    def set_loras(
        self,
        entries: Sequence[LoRAEntry | tuple[str | os.PathLike[str], float] | str | os.PathLike[str]],
        *,
        model_path: str | os.PathLike[str] | None = None,
    ) -> list[LoRAEntry]:
        """Validate/convert entries, close any active session, and apply via ``qf_xfm_set_loras``."""
        normalized = self._normalize_entries(entries)
        eff_model = os.fspath(model_path) if model_path is not None else self._default_model_path

        prepared: list[dict[str, Any]] = []
        for entry in normalized:
            ready_path = self._prepare_path(entry.path, model_path=eff_model)
            prepared.append({"path": ready_path, "scale": float(entry.scale)})

        self._handle.end_session_if_open()
        self._handle.set_loras(prepared)
        self._active_loras = list(normalized)
        return list(self._active_loras)

    def load_lora(
        self,
        path: str | os.PathLike[str],
        scale: float = 1.0,
        *,
        model_path: str | os.PathLike[str] | None = None,
    ) -> list[LoRAEntry]:
        """Hot-load a single LoRA adapter."""
        return self.set_loras([LoRAEntry(path=os.fspath(path), scale=float(scale))], model_path=model_path)

    def unload_loras(self) -> None:
        """Close any active session and clear active LoRAs by sending ``[]`` to ``qf_xfm_set_loras``."""
        self._handle.end_session_if_open()
        self._handle.set_loras([])
        self._active_loras = []


__all__ = [
    "LoRAEntry",
    "detect_lora_format",
    "unsupported_lora_kind",
    "convert_lora_file",
    "LoRAManager",
]
