"""Model-agnostic preset metadata loader, safetensors header sniffer, and checkpoint resolver.

100% model-agnostic: discovers bundled presets dynamically from ``quantfunc/presets/*/preset.json``
without hardcoding any architecture names.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_MAX_ST_HEADER_BYTES = 512 * 1024 * 1024
_ST_HEADER_LEN_BYTES = struct.calcsize("<Q")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_JSON_WS_BYTES = frozenset((0x20, 0x09, 0x0A, 0x0D))
_JSON_PRIM_STOP_BYTES = frozenset((0x2C, 0x7D, 0x5D, 0x20, 0x09, 0x0A, 0x0D))
_JSON_STRUCT_TOKEN_RE = re.compile(rb'["{}\[\]]')
_JSON_SIMPLE_ESCAPES: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


@dataclass(frozen=True)
class PresetMetadata:
    """Data-driven model preset metadata loaded from ``presets/<arch>/preset.json``."""

    arch: str
    display_name: str = ""
    modality: str = "image"
    vae_scale_factors: tuple[int, ...] = (1, 8, 8)
    spatial_compression: int = 8
    temporal_compression: int = 1
    default_quality_ratio: float = 0.80
    audio_latent_channels: int = 0
    audio_mel_bins: int = 0
    audio_stereo_channels: int = 0
    default_video_sigma_shift: float | None = None
    default_audio_sigma_shift: float | None = None
    frame_grid: dict[str, int] | None = None
    default_fps: float | None = None
    pipeline_classes: tuple[str, ...] = ()
    transformer_classes: tuple[str, ...] = ()
    file_hints: tuple[str, ...] = ()
    tensor_hints: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> PresetMetadata:
        with open(os.fspath(path), encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict) or not raw.get("arch"):
            raise ValueError(f"Invalid preset.json at {path!r}: missing 'arch'")
        fg = raw.get("frame_grid")
        frame_grid = {str(k): int(v) for k, v in fg.items()} if isinstance(fg, dict) else None
        fps_raw = raw.get("default_fps")
        vshift_raw = raw.get("default_video_sigma_shift")
        ashift_raw = raw.get("default_audio_sigma_shift")
        return cls(
            arch=str(raw["arch"]),
            display_name=str(raw.get("display_name", raw["arch"])),
            modality=str(raw.get("modality", "image")),
            vae_scale_factors=tuple(int(x) for x in raw.get("vae_scale_factors", (1, 8, 8))),
            spatial_compression=int(raw.get("spatial_compression", 8)),
            temporal_compression=int(raw.get("temporal_compression", 1)),
            default_quality_ratio=float(raw.get("default_quality_ratio", 0.80)),
            audio_latent_channels=int(raw.get("audio_latent_channels", 0)),
            audio_mel_bins=int(raw.get("audio_mel_bins", 0)),
            audio_stereo_channels=int(raw.get("audio_stereo_channels", 0)),
            default_video_sigma_shift=float(vshift_raw) if vshift_raw is not None else None,
            default_audio_sigma_shift=float(ashift_raw) if ashift_raw is not None else None,
            frame_grid=frame_grid,
            default_fps=float(fps_raw) if fps_raw is not None else None,
            pipeline_classes=tuple(str(x) for x in raw.get("pipeline_classes", ())),
            transformer_classes=tuple(str(x) for x in raw.get("transformer_classes", ())),
            file_hints=tuple(str(x) for x in raw.get("file_hints", ())),
            tensor_hints=tuple(str(x) for x in raw.get("tensor_hints", ())),
        )

    @classmethod
    def from_arch(cls, arch: str) -> PresetMetadata:
        return cls.from_file(bundled_preset_dir(arch) / "preset.json")

    def frames_from_latent_t(self, latent_t: int) -> int:
        """Compute pixel frame count from latent temporal length using ``frame_grid`` or ``temporal_compression``."""
        lt = int(latent_t)
        if lt <= 0:
            raise ValueError(f"latent_t must be positive, got {latent_t!r}")
        if self.frame_grid:
            base_f = int(self.frame_grid["base_frames"])
            stride_f = int(self.frame_grid["frame_stride"])
            base_t = int(self.frame_grid["base_latent_t"])
            stride_t = int(self.frame_grid["latent_t_stride"])
            if lt < base_t or (lt - base_t) % stride_t != 0:
                raise ValueError(
                    f"latent_t={lt} does not lie on {self.arch} frame_grid "
                    f"(base_latent_t={base_t}, latent_t_stride={stride_t})"
                )
            return base_f + ((lt - base_t) // stride_t) * stride_f
        return (lt - 1) * int(self.temporal_compression) + 1

    def latent_t_from_frames(self, num_frames: int) -> int:
        """Compute latent temporal length from pixel frame count."""
        nf = int(num_frames)
        if nf <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames!r}")
        if self.frame_grid:
            base_f = int(self.frame_grid["base_frames"])
            stride_f = int(self.frame_grid["frame_stride"])
            base_t = int(self.frame_grid["base_latent_t"])
            stride_t = int(self.frame_grid["latent_t_stride"])
            if nf <= base_f:
                return base_t
            return base_t + ((nf - base_f) // stride_f) * stride_t
        tc = max(1, int(self.temporal_compression))
        return (nf - 1) // tc + 1

    def align_frame_count(self, num_frames: int) -> int:
        """Snap ``num_frames`` upward to the nearest valid frame-grid multiple."""
        nf = max(1, int(num_frames))
        if self.frame_grid:
            base_f = int(self.frame_grid["base_frames"])
            stride_f = int(self.frame_grid["frame_stride"])
            if nf <= base_f:
                return base_f
            rem = (nf - base_f) % stride_f
            return nf if rem == 0 else nf + (stride_f - rem)
        tc = max(1, int(self.temporal_compression))
        rem = (nf - 1) % tc
        return nf if rem == 0 else nf + (tc - rem)


def name_matches_hints(path: str | os.PathLike[str], patterns: Sequence[str]) -> bool:
    """Case-insensitive fnmatch of ``basename(path)`` against ``patterns``."""
    base = os.path.basename(os.fspath(path)).lower()
    return any(fnmatch.fnmatch(base, str(p).lower()) for p in patterns)


def _read_raw_safetensors_header(
    path: str | os.PathLike[str],
    *,
    max_header_bytes: int = _MAX_ST_HEADER_BYTES,
) -> bytes:
    """Read and length-validate the raw JSON header bytes of a ``.safetensors`` file."""
    p = os.fspath(path)
    with open(p, "rb") as fh:
        raw_len = fh.read(_ST_HEADER_LEN_BYTES)
        if len(raw_len) < _ST_HEADER_LEN_BYTES:
            raise ValueError(f"Truncated safetensors file (missing header length): {p!r}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len == 0 or header_len > max_header_bytes:
            raise ValueError(f"Implausible safetensors header length {header_len} in {p!r}")
        raw_hdr = fh.read(header_len)
        if len(raw_hdr) < header_len:
            raise ValueError(f"Truncated safetensors header payload in {p!r}")
    return raw_hdr


def _skip_json_ws(buf: bytes, i: int, n: int) -> int:
    while i < n and buf[i] in _JSON_WS_BYTES:
        i += 1
    return i


def _scan_json_string_end(buf: bytes, start_quote: int, n: int) -> int:
    """Locate the closing unescaped quote index for a JSON string starting at ``start_quote``."""
    pos = start_quote + 1
    while pos < n:
        q = buf.find(b'"', pos)
        if q == -1:
            break
        bs = q - 1
        while bs > start_quote and buf[bs] == 0x5C:  # '\\'
            bs -= 1
        if ((q - 1 - bs) & 1) == 0:
            return q
        pos = q + 1
    raise ValueError("Unterminated JSON string in safetensors header")


def _parse_hex4(h: str, err_msg: str) -> int:
    if len(h) != 4 or not all(ch in _HEX_DIGITS for ch in h):
        raise ValueError(err_msg)
    return int(h, 16)


def _unescape_json_string(raw: bytes) -> str:
    """Decode RFC-8259 JSON string escapes without invoking ``json.loads``."""
    text = raw.decode("utf-8")
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            raise ValueError("Unterminated escape sequence in safetensors header key")
        esc = text[i + 1]
        simple = _JSON_SIMPLE_ESCAPES.get(esc)
        if simple is not None:
            out.append(simple)
            i += 2
            continue
        if esc == "u":
            if i + 6 > n:
                raise ValueError("Truncated \\u escape in safetensors header key")
            code = _parse_hex4(
                text[i + 2 : i + 6],
                "Invalid \\u escape in safetensors header key",
            )
            i += 6
            if 0xD800 <= code <= 0xDBFF and i + 6 <= n and text[i : i + 2] == "\\u":
                low = _parse_hex4(
                    text[i + 2 : i + 6],
                    "Invalid surrogate \\u escape in safetensors header key",
                )
                if 0xDC00 <= low <= 0xDFFF:
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    i += 6
            out.append(chr(code))
            continue
        raise ValueError(f"Invalid escape \\{esc} in safetensors header key")
    return "".join(out)


def _decode_json_key(raw: bytes) -> str:
    if b"\\" not in raw:
        return raw.decode("utf-8")
    return _unescape_json_string(raw)


def _skip_json_value(buf: bytes, i: int, n: int) -> int:
    """Skip one complete JSON value starting at ``i`` and return the next byte index."""
    if i >= n:
        raise ValueError("Unexpected end of safetensors header JSON value")
    b = buf[i]
    if b == 0x22:  # '"'
        return _scan_json_string_end(buf, i, n) + 1
    if b in (0x7B, 0x5B):  # '{' or '['
        stack = [0x7D if b == 0x7B else 0x5D]
        pos = i + 1
        while stack:
            m = _JSON_STRUCT_TOKEN_RE.search(buf, pos)
            if m is None:
                raise ValueError("Unterminated JSON container in safetensors header")
            tok = buf[m.start()]
            if tok == 0x22:  # '"'
                pos = _scan_json_string_end(buf, m.start(), n) + 1
            elif tok == 0x7B:
                stack.append(0x7D)
                pos = m.end()
            elif tok == 0x5B:
                stack.append(0x5D)
                pos = m.end()
            else:  # '}' or ']'
                if tok != stack.pop():
                    raise ValueError("Mismatched JSON container delimiter in safetensors header")
                pos = m.end()
        return pos
    start = i
    while i < n and buf[i] not in _JSON_PRIM_STOP_BYTES:
        i += 1
    if i == start:
        raise ValueError("Malformed JSON primitive in safetensors header")
    return i


def safetensors_header_has_key_markers(
    path: str | os.PathLike[str],
    markers: Sequence[str],
    *,
    match: Literal["prefix", "substring"] = "substring",
    max_header_bytes: int = _MAX_ST_HEADER_BYTES,
) -> dict[str, bool]:
    """Scan depth-1 JSON tensor keys in a ``.safetensors`` header without ``json.loads``.

    Skips ``__metadata__`` and all nested value bodies (including escaped strings),
    matching only top-level tensor key names by ``prefix`` or ``substring``.
    """
    if match not in ("prefix", "substring"):
        raise ValueError(f"Unsupported match mode {match!r}; expected 'prefix' or 'substring'")
    raw_hdr = _read_raw_safetensors_header(path, max_header_bytes=max_header_bytes)
    results: dict[str, bool] = {str(m): False for m in markers}
    n = len(raw_hdr)
    p = os.fspath(path)

    i = _skip_json_ws(raw_hdr, 0, n)
    if i >= n or raw_hdr[i] != 0x7B:  # '{'
        raise ValueError(f"Safetensors header in {p!r} is not a JSON object")
    if not results:
        return results

    i = _skip_json_ws(raw_hdr, i + 1, n)
    if i < n and raw_hdr[i] == 0x7D:  # '{}'
        return results

    pending = set(results.keys())
    while i < n:
        if raw_hdr[i] != 0x22:  # '"'
            raise ValueError(f"Malformed safetensors header JSON key in {p!r}")
        end_q = _scan_json_string_end(raw_hdr, i, n)
        try:
            key = _decode_json_key(raw_hdr[i + 1 : end_q])
        except UnicodeDecodeError as exc:
            raise ValueError(f"Malformed UTF-8 in safetensors header key in {p!r}: {exc}") from exc

        i = _skip_json_ws(raw_hdr, end_q + 1, n)
        if i >= n or raw_hdr[i] != 0x3A:  # ':'
            raise ValueError(f"Malformed safetensors header JSON (missing ':') in {p!r}")
        i = _skip_json_ws(raw_hdr, i + 1, n)
        i = _skip_json_value(raw_hdr, i, n)

        if key != "__metadata__":
            matched_now = [
                m
                for m in pending
                if (key.startswith(m) if match == "prefix" else m in key)
            ]
            if matched_now:
                for m in matched_now:
                    results[m] = True
                    pending.discard(m)
                if not pending:
                    return results

        i = _skip_json_ws(raw_hdr, i, n)
        if i >= n:
            raise ValueError(f"Truncated safetensors header JSON in {p!r}")
        ch = raw_hdr[i]
        if ch == 0x2C:  # ','
            i = _skip_json_ws(raw_hdr, i + 1, n)
            continue
        if ch == 0x7D:  # '}'
            return results
        raise ValueError(f"Malformed safetensors header JSON (expected ',' or '}}') in {p!r}")
    raise ValueError(f"Truncated safetensors header JSON in {p!r}")


def sniff_safetensors_header(
    path: str | os.PathLike[str],
    *,
    max_header_bytes: int = _MAX_ST_HEADER_BYTES,
) -> dict[str, Any]:
    """Read only the JSON header of a ``.safetensors`` file without touching tensor bytes."""
    p = os.fspath(path)
    raw_hdr = _read_raw_safetensors_header(p, max_header_bytes=max_header_bytes)
    try:
        parsed = json.loads(raw_hdr.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed safetensors header JSON in {p!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Safetensors header in {p!r} is not a JSON object")
    return parsed


def bundled_presets_root() -> Path:
    """Return the package directory containing bundled model presets."""
    return Path(__file__).resolve().parent.parent / "presets"


def list_bundled_presets() -> tuple[str, ...]:
    """Discover all bundled preset directory names dynamically."""
    root = bundled_presets_root()
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and (entry / "preset.json").is_file()
        )
    )


def bundled_preset_dir(arch: str) -> Path:
    """Return the bundled preset directory for ``arch`` (rejects path traversal)."""
    if not isinstance(arch, str) or not arch.strip() or "/" in arch or "\\" in arch or ".." in arch:
        raise ValueError(f"Invalid preset architecture name: {arch!r}")
    candidate = bundled_presets_root() / arch.strip()
    if not (candidate / "preset.json").is_file():
        raise FileNotFoundError(
            f"No bundled preset found for architecture {arch!r}; available: {list_bundled_presets()}"
        )
    return candidate


def _find_safetensors_in_dir(
    directory: Path,
    *,
    filename: str | None = None,
    subfolder: str | None = None,
    file_hints: Sequence[str] = (),
) -> Path:
    search_root = directory / subfolder if subfolder else directory
    if filename:
        target = search_root / filename
        if target.is_file():
            return target.resolve()
        raise FileNotFoundError(f"Checkpoint file {filename!r} not found in {search_root}")

    candidates: list[Path] = []
    for sub in (search_root, search_root / "transformer"):
        if sub.is_dir():
            candidates.extend(sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors files found under {search_root}")
    if file_hints:
        matched = [p for p in candidates if name_matches_hints(p, file_hints)]
        if len(matched) == 1:
            return matched[0].resolve()
        if matched:
            return matched[0].resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise ValueError(
        f"Multiple .safetensors files found in {search_root} ({[p.name for p in candidates]}); "
        "specify filename= explicitly"
    )


def resolve_checkpoint(
    pretrained_model_name_or_path: str | os.PathLike[str],
    *,
    filename: str | None = None,
    source: str = "auto",
    revision: str | None = None,
    subfolder: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    token: str | None = None,
    file_hints: Sequence[str] = (),
) -> Path:
    """Resolve a checkpoint from a local file/directory, HuggingFace Hub, or ModelScope."""
    raw = os.fspath(pretrained_model_name_or_path)
    expanded = Path(os.path.expanduser(raw))
    if expanded.is_file():
        return expanded.resolve()
    if expanded.is_dir():
        return _find_safetensors_in_dir(
            expanded,
            filename=filename,
            subfolder=subfolder,
            file_hints=file_hints,
        )

    src = source.strip().lower()
    if src in ("auto", "hf", "huggingface"):
        try:
            import huggingface_hub
        except ImportError as exc:
            raise ImportError(
                f"Checkpoint {raw!r} does not exist locally and huggingface_hub is not installed. "
                "Install `quantfunc[hf]` or pass a local .safetensors file path."
            ) from exc
        if filename:
            local = huggingface_hub.hf_hub_download(
                repo_id=raw,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                cache_dir=os.fspath(cache_dir) if cache_dir else None,
                token=token,
            )
            return Path(local).resolve()
        snap = Path(
            huggingface_hub.snapshot_download(
                repo_id=raw,
                revision=revision,
                cache_dir=os.fspath(cache_dir) if cache_dir else None,
                token=token,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
        return _find_safetensors_in_dir(snap, subfolder=subfolder, file_hints=file_hints)

    if src == "modelscope":
        try:
            from modelscope.hub.file_download import model_file_download
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise ImportError(
                f"Checkpoint {raw!r} requested source='modelscope' but modelscope is not installed. "
                "Install `quantfunc[modelscope]` or pass a local .safetensors file path."
            ) from exc
        if filename:
            rel = f"{subfolder}/{filename}" if subfolder else filename
            local = model_file_download(
                model_id=raw,
                file_path=rel,
                revision=revision,
                cache_dir=os.fspath(cache_dir) if cache_dir else None,
            )
            return Path(local).resolve()
        snap = Path(
            snapshot_download(
                model_id=raw,
                revision=revision,
                cache_dir=os.fspath(cache_dir) if cache_dir else None,
            )
        )
        return _find_safetensors_in_dir(snap, subfolder=subfolder, file_hints=file_hints)

    raise ValueError(f"Unsupported checkpoint source {source!r}; expected 'auto', 'hf', or 'modelscope'")


_TRANSFORMER_CONFIG_KEYS = frozenset(
    {
        "in_channels",
        "latents_dim",
        "num_layers",
        "hidden_size",
        "attention_head_dim",
        "num_attention_heads",
        "cross_attention_dim",
    }
)


def _looks_like_transformer_config(data: Mapping[str, Any]) -> bool:
    """Return True when ``data`` carries recognizable transformer geometry keys."""
    return any(k in data for k in _TRANSFORMER_CONFIG_KEYS)


def resolve_transformer_config(
    arch: str | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
    default_config_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve and load transformer ``config.json``, returning ``(config_dict, config_file_path)``."""
    bundled_cfg: dict[str, Any] | None = None
    bundled_path: Path | None = None
    if arch:
        try:
            cand_bundled = bundled_preset_dir(arch) / "transformer" / "config.json"
            if cand_bundled.is_file():
                with open(cand_bundled, encoding="utf-8") as fh:
                    raw_b = json.load(fh)
                if isinstance(raw_b, dict):
                    bundled_cfg = raw_b
                    bundled_path = cand_bundled
        except (FileNotFoundError, ValueError):
            pass

    if config_path is not None:
        cp = Path(os.path.expanduser(os.fspath(config_path)))
        explicit_cands = [cp / "transformer" / "config.json", cp / "config.json"] if cp.is_dir() else [cp]
        for cand in explicit_cands:
            if cand.is_file():
                with open(cand, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    raise ValueError(f"Transformer config at {cand} must be a JSON object")
                return data, str(cand.resolve())

    if weights_path is not None:
        wp = Path(os.path.expanduser(os.fspath(weights_path)))
        base_dir = wp if wp.is_dir() else wp.parent
        for cand in (base_dir / "transformer" / "config.json", base_dir / "config.json"):
            if not cand.is_file():
                continue
            with open(cand, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"Transformer config at {cand} must be a JSON object")
            if not _looks_like_transformer_config(data):
                continue
            if bundled_cfg is not None:
                merged = dict(bundled_cfg)
                merged.update(data)
                # Prefer bundled file path when sibling is missing core keys the native engine expects.
                required_subset = set(bundled_cfg) & _TRANSFORMER_CONFIG_KEYS
                chosen_path = cand if required_subset.issubset(data) else (bundled_path or cand)
                return merged, str(chosen_path.resolve())
            return data, str(cand.resolve())

    if default_config_path is not None:
        cand = Path(os.path.expanduser(os.fspath(default_config_path)))
        if cand.is_file():
            with open(cand, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"Transformer config at {cand} must be a JSON object")
            return data, str(cand.resolve())

    if bundled_cfg is not None and bundled_path is not None:
        return dict(bundled_cfg), str(bundled_path.resolve())

    raise FileNotFoundError(
        f"Could not resolve transformer config.json (arch={arch!r}, config_path={config_path!r}, "
        f"weights_path={weights_path!r})"
    )


__all__ = [
    "PresetMetadata",
    "name_matches_hints",
    "safetensors_header_has_key_markers",
    "sniff_safetensors_header",
    "bundled_presets_root",
    "list_bundled_presets",
    "bundled_preset_dir",
    "resolve_checkpoint",
    "resolve_transformer_config",
]
