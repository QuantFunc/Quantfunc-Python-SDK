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
import stat
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
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


_LORA_ITEM_ALLOWED_KEYS: frozenset[str] = frozenset({"path", "name", "scale"})


def _is_policy_unset(value: Any) -> bool:
    return value is ... or type(value).__name__ == "_UnsetType"


def _require_nonempty_path_str(value: Any, label: str) -> str:
    if isinstance(value, os.PathLike):
        raw = os.fspath(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise ValueError(f"{label} must be a non-empty string or PathLike, got {type(value).__name__}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    if "\x00" in raw:
        raise ValueError(f"{label} must not contain NUL bytes")
    return raw


def _copy_lora_registry(
    lora_registry: Mapping[str, str | os.PathLike[str]] | None,
) -> dict[str, str] | None:
    if lora_registry is None:
        return None
    if not isinstance(lora_registry, Mapping):
        raise ValueError(f"lora_registry must be a mapping or None, got {type(lora_registry).__name__}")
    out: dict[str, str] = {}
    for k, v in lora_registry.items():
        if not isinstance(k, str) or not k.strip() or "\x00" in k:
            raise ValueError(f"lora_registry key must be a non-empty string without NUL bytes, got {k!r}")
        out[k] = _require_nonempty_path_str(v, f"lora_registry[{k!r}]")
    return out


def _copy_allowed_roots(
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None,
) -> tuple[str, ...] | None:
    if allowed_lora_roots is None:
        return None
    if isinstance(allowed_lora_roots, str | bytes | bytearray | os.PathLike) or not isinstance(
        allowed_lora_roots, Sequence
    ):
        raise ValueError(
            "allowed_lora_roots must be a sequence of directory paths (not a bare string or PathLike)"
        )
    return tuple(_require_nonempty_path_str(r, "allowed_lora_roots entry") for r in allowed_lora_roots)


@dataclass(frozen=True)
class LoRAEntry:
    """Single LoRA adapter specification."""

    path: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError(f"LoRAEntry.path must be a non-empty string, got {self.path!r}")
        if "\x00" in self.path:
            raise ValueError("LoRAEntry.path must not contain NUL bytes")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int | float):
            raise ValueError(f"LoRAEntry.scale must be a finite number, got {self.scale!r}")
        sc = float(self.scale)
        if not math.isfinite(sc):
            raise ValueError(f"LoRAEntry.scale must be finite, got {sc!r}")
        if sc == 0.0:
            sc = 0.0
        object.__setattr__(self, "scale", sc)


def normalize_lora_entries(
    entries: Sequence[
        LoRAEntry
        | tuple[str | os.PathLike[str], float]
        | list[Any]
        | Mapping[str, Any]
        | str
        | os.PathLike[str]
    ]
    | None,
) -> list[LoRAEntry]:
    """Normalize user/JSON LoRA specifications into a validated list of ``LoRAEntry`` items."""
    if entries is None:
        return []
    if isinstance(entries, str | bytes | bytearray | Mapping) or not isinstance(entries, Sequence):
        raise ValueError(
            f"loras must be a sequence of LoRA specifications (not {type(entries).__name__}); pass [] to clear"
        )
    out: list[LoRAEntry] = []
    for item in entries:
        if isinstance(item, LoRAEntry):
            out.append(item)
        elif isinstance(item, str | os.PathLike):
            path_str = _require_nonempty_path_str(item, "LoRA path")
            out.append(LoRAEntry(path=path_str, scale=1.0))
        elif isinstance(item, Mapping):
            unknown: list[str] = []
            for k in item:
                if not isinstance(k, str):
                    raise ValueError(f"LoRA entry mapping keys must be strings, got {type(k).__name__}")
                if k not in _LORA_ITEM_ALLOWED_KEYS:
                    unknown.append(k)
            if unknown:
                raise ValueError(f"Unknown LoRA entry key(s): {', '.join(sorted(unknown))}")
            has_path = "path" in item
            has_name = "name" in item
            if not has_path and not has_name:
                raise ValueError("LoRA entry mapping must specify 'path' or 'name'")
            p_str = _require_nonempty_path_str(item["path"], "LoRA entry 'path'") if has_path else None
            n_str = _require_nonempty_path_str(item["name"], "LoRA entry 'name'") if has_name else None
            if p_str is not None and n_str is not None and p_str != n_str:
                raise ValueError(
                    f"Conflicting LoRA entry keys 'path' ({p_str!r}) and 'name' ({n_str!r})"
                )
            chosen = p_str if p_str is not None else n_str
            assert chosen is not None
            raw_scale = item["scale"] if "scale" in item else 1.0
            out.append(LoRAEntry(path=chosen, scale=raw_scale))
        elif isinstance(item, tuple | list):
            if len(item) != 2:
                raise ValueError(
                    f"LoRA tuple/list entry must have length 2 (path, scale), got length {len(item)}"
                )
            path_str = _require_nonempty_path_str(item[0], "LoRA path")
            out.append(LoRAEntry(path=path_str, scale=item[1]))
        else:
            raise ValueError(
                f"Unsupported LoRA specification {item!r}; expected LoRAEntry, path, (path, scale), or mapping"
            )
    return out


def resolve_lora_path(
    raw: str | os.PathLike[str],
    *,
    lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
    allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
    strict_registry: bool = False,
) -> str:
    """Resolve a LoRA alias or path and enforce ``strict_registry`` / ``allowed_lora_roots`` BEFORE stat/open.

    Executes containment via ``os.path.realpath`` + ``os.path.commonpath`` prior to any filesystem
    existence checks, closing path-traversal, sibling-prefix, symlink-escape, and existence-oracle vectors.
    """
    raw_str = _require_nonempty_path_str(raw, "LoRA path")
    if not isinstance(strict_registry, bool):
        raise ValueError(f"strict_registry must be bool, got {strict_registry!r}")
    reg = _copy_lora_registry(lora_registry)
    roots = _copy_allowed_roots(allowed_lora_roots)

    if reg is not None and raw_str in reg:
        candidate = reg[raw_str]
    else:
        if strict_registry:
            raise PermissionError(f"LoRA {raw_str!r} is not registered in lora_registry")
        candidate = raw_str

    real_path = os.path.realpath(os.path.expanduser(candidate))

    if roots is not None:
        matched = False
        for r in roots:
            root_real = os.path.realpath(os.path.expanduser(r))
            try:
                common = os.path.commonpath([root_real, real_path])
            except ValueError:
                continue
            if real_path != root_real and common == root_real:
                matched = True
                break
        if not matched:
            raise PermissionError(f"LoRA path {raw_str!r} is outside allowed_lora_roots")

    return real_path


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
    """Manages LoRA policy enforcement, stat-cached O(1) no-op hot-swap, and auto-conversion."""

    def __init__(
        self,
        handle: Any,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool = False,
        on_before_rebind: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(strict_registry, bool):
            raise ValueError(f"strict_registry must be bool, got {strict_registry!r}")
        self._handle = handle
        self._cache_dir = (
            os.path.abspath(os.path.expanduser(os.fspath(cache_dir)))
            if cache_dir is not None
            else os.path.join(default_cache_dir(), "loras")
        )
        self._default_model_path = os.fspath(model_path) if model_path is not None else None
        self._lora_registry: dict[str, str] | None = _copy_lora_registry(lora_registry)
        self._allowed_lora_roots: tuple[str, ...] | None = _copy_allowed_roots(allowed_lora_roots)
        self._strict_registry: bool = strict_registry
        self._on_before_rebind: Callable[[], Any] | None = on_before_rebind

        self._active_loras: list[LoRAEntry] = []
        self._bound_stack_sig: tuple[tuple[str, int, int, int, float], ...] | None = ()
        self._bound_model_sig: str = ""
        self._bound_prepared_sig: str = "[]"
        self._prepared_cache: dict[tuple[str, int, int, int, str], str] = {}

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def active_loras(self) -> tuple[LoRAEntry, ...]:
        return tuple(self._active_loras)

    @property
    def lora_registry(self) -> dict[str, str] | None:
        return dict(self._lora_registry) if self._lora_registry is not None else None

    @property
    def allowed_lora_roots(self) -> tuple[str, ...] | None:
        return self._allowed_lora_roots

    @property
    def strict_registry(self) -> bool:
        return self._strict_registry

    def configure_policy(
        self,
        *,
        lora_registry: Any = ...,
        allowed_lora_roots: Any = ...,
        strict_registry: Any = ...,
    ) -> None:
        """Atomically update default alias registry, allowed roots, and strict registry mode."""
        new_reg = self._lora_registry if _is_policy_unset(lora_registry) else _copy_lora_registry(lora_registry)
        new_roots = (
            self._allowed_lora_roots
            if _is_policy_unset(allowed_lora_roots)
            else _copy_allowed_roots(allowed_lora_roots)
        )
        if _is_policy_unset(strict_registry):
            new_strict = self._strict_registry
        else:
            if not isinstance(strict_registry, bool):
                raise ValueError(f"strict_registry must be bool, got {strict_registry!r}")
            new_strict = strict_registry

        self._lora_registry = new_reg
        self._allowed_lora_roots = new_roots
        self._strict_registry = new_strict

    def _invoke_before_rebind(self) -> None:
        if self._on_before_rebind is not None:
            self._on_before_rebind()
        else:
            self._handle.end_session_if_open()

    def _normalize_entries(
        self,
        entries: Sequence[
            LoRAEntry
            | tuple[str | os.PathLike[str], float]
            | list[Any]
            | Mapping[str, Any]
            | str
            | os.PathLike[str]
        ],
    ) -> list[LoRAEntry]:
        return normalize_lora_entries(entries)

    def _prepare_resolved_path(
        self,
        abs_path: str,
        *,
        raw_label: str,
        st: os.stat_result,
        model_path: str | None = None,
        force: bool = False,
    ) -> str:
        hdr = _read_safetensors_header(abs_path)
        meta = hdr.get("__metadata__", {})
        keys = [k for k in hdr if k != "__metadata__"]

        kind = unsupported_lora_kind(keys)
        if kind is not None:
            name, why = kind
            raise ValueError(f"Unsupported LoRA kind {name}: {why}")

        fmt = detect_lora_format(keys)
        if fmt == "unknown":
            raise ValueError(f"Could not detect LoRA format in {raw_label!r}")

        if not _needs_conversion(keys, fmt):
            conv, bfl = _convert_keys(keys, fmt)
            why = _check_refusal(conv, meta if isinstance(meta, dict) else {}, bfl)
            if why is not None:
                raise ValueError(f"Refusing LoRA {raw_label!r}: {why}")
            return abs_path

        eff_model = os.path.realpath(model_path) if model_path else ""
        cache_key = hashlib.sha256(
            f"{abs_path}|{st.st_size}|{st.st_mtime_ns}|{st.st_ino}|{eff_model}".encode()
        ).hexdigest()[:24]
        os.makedirs(self._cache_dir, exist_ok=True)
        cached_path = os.path.join(self._cache_dir, f"{cache_key}.safetensors")
        if not force and os.path.isfile(cached_path):
            return cached_path
        return convert_lora_file(abs_path, cached_path, model_path=eff_model or None)

    def _prepare_path(self, raw_path: str, *, model_path: str | None = None) -> str:
        real_path = resolve_lora_path(
            raw_path,
            lora_registry=self._lora_registry,
            allowed_lora_roots=self._allowed_lora_roots,
            strict_registry=self._strict_registry,
        )
        try:
            st = os.stat(real_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"LoRA file not found: {raw_path!r}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise FileNotFoundError(f"LoRA file not found: {raw_path!r}")
        return self._prepare_resolved_path(real_path, raw_label=raw_path, st=st, model_path=model_path)

    def set_loras(
        self,
        entries: Sequence[
            LoRAEntry
            | tuple[str | os.PathLike[str], float]
            | list[Any]
            | Mapping[str, Any]
            | str
            | os.PathLike[str]
        ],
        *,
        model_path: str | os.PathLike[str] | None = None,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool | None = None,
        force: bool = False,
    ) -> list[LoRAEntry]:
        """Validate/convert entries with O(1) stat-cached no-op and apply via ``qf_xfm_set_loras``."""
        normalized = normalize_lora_entries(entries)
        if not normalized:
            self.unload_loras(force=force)
            return []

        eff_registry = self._lora_registry if lora_registry is None else _copy_lora_registry(lora_registry)
        eff_roots = (
            self._allowed_lora_roots if allowed_lora_roots is None else _copy_allowed_roots(allowed_lora_roots)
        )
        if strict_registry is None:
            eff_strict = self._strict_registry
        else:
            if not isinstance(strict_registry, bool):
                raise ValueError(f"strict_registry must be bool, got {strict_registry!r}")
            eff_strict = strict_registry

        eff_model_raw = os.fspath(model_path) if model_path is not None else self._default_model_path
        eff_model_real = os.path.realpath(os.path.expanduser(eff_model_raw)) if eff_model_raw else ""

        resolved_items: list[tuple[LoRAEntry, str, os.stat_result]] = []
        sig_parts: list[tuple[str, int, int, int, float]] = []
        for entry in normalized:
            real_path = resolve_lora_path(
                entry.path,
                lora_registry=eff_registry,
                allowed_lora_roots=eff_roots,
                strict_registry=eff_strict,
            )
            try:
                st = os.stat(real_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"LoRA file not found: {entry.path!r}") from exc
            if not stat.S_ISREG(st.st_mode):
                raise FileNotFoundError(f"LoRA path is not a regular file: {entry.path!r}")
            resolved_items.append((entry, real_path, st))
            sig_parts.append((real_path, st.st_mtime_ns, st.st_size, st.st_ino, float(entry.scale)))

        candidate_sig = tuple(sig_parts)
        handle_sig = getattr(self._handle, "applied_lora_sig", self._bound_prepared_sig)
        if (
            not force
            and self._bound_stack_sig is not None
            and candidate_sig == self._bound_stack_sig
            and eff_model_real == self._bound_model_sig
            and handle_sig == self._bound_prepared_sig
        ):
            return list(self._active_loras)

        prepared: list[dict[str, Any]] = []
        for entry, real_path, st in resolved_items:
            file_key = (real_path, st.st_mtime_ns, st.st_size, st.st_ino, eff_model_real)
            cached_ready = None if force else self._prepared_cache.get(file_key)
            if cached_ready is not None and os.path.isfile(cached_ready):
                ready_path = cached_ready
            else:
                ready_path = self._prepare_resolved_path(
                    real_path,
                    raw_label=entry.path,
                    st=st,
                    model_path=eff_model_real or None,
                    force=force,
                )
                # Evict stale stat snapshots for the same (real_path, model_sig)
                stale_keys = [
                    k for k in self._prepared_cache if k[0] == real_path and k[4] == eff_model_real and k != file_key
                ]
                for sk in stale_keys:
                    self._prepared_cache.pop(sk, None)
                self._prepared_cache[file_key] = ready_path
            prepared.append({"path": ready_path, "scale": float(entry.scale)})

        self._invoke_before_rebind()
        prepared_sig = json.dumps(prepared, allow_nan=False, separators=(",", ":"))
        try:
            self._handle.set_loras(prepared)
        except BaseException:
            self._bound_stack_sig = None
            raise

        self._active_loras = list(normalized)
        self._bound_stack_sig = candidate_sig
        self._bound_model_sig = eff_model_real
        self._bound_prepared_sig = getattr(self._handle, "applied_lora_sig", prepared_sig)
        return list(self._active_loras)

    def load_lora(
        self,
        path: str | os.PathLike[str],
        scale: float = 1.0,
        *,
        model_path: str | os.PathLike[str] | None = None,
        lora_registry: Mapping[str, str | os.PathLike[str]] | None = None,
        allowed_lora_roots: Sequence[str | os.PathLike[str]] | None = None,
        strict_registry: bool | None = None,
        force: bool = False,
    ) -> list[LoRAEntry]:
        """Hot-load a single LoRA adapter."""
        path_str = _require_nonempty_path_str(path, "LoRA path")
        return self.set_loras(
            [LoRAEntry(path=path_str, scale=scale)],
            model_path=model_path,
            lora_registry=lora_registry,
            allowed_lora_roots=allowed_lora_roots,
            strict_registry=strict_registry,
            force=force,
        )

    def unload_loras(self, *, force: bool = False) -> None:
        """Close any active session and clear active LoRAs by sending ``[]`` (O(1) no-op when already empty)."""
        handle_sig = getattr(self._handle, "applied_lora_sig", "[]")
        if not force and self._bound_stack_sig == () and handle_sig == "[]":
            return
        self._invoke_before_rebind()
        try:
            self._handle.set_loras([])
        except BaseException:
            self._bound_stack_sig = None
            raise
        self._active_loras = []
        self._bound_stack_sig = ()
        self._bound_model_sig = ""
        self._bound_prepared_sig = getattr(self._handle, "applied_lora_sig", "[]")


__all__ = [
    "LoRAEntry",
    "normalize_lora_entries",
    "resolve_lora_path",
    "detect_lora_format",
    "unsupported_lora_kind",
    "convert_lora_file",
    "LoRAManager",
]
