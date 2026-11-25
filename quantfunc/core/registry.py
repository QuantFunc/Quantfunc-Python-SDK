"""Extensible model architecture registry and data-driven architecture auto-detection.

100% model-agnostic: zero hardcoded model names; all detection is driven by registered
``ModelSpec`` entries and bundled ``presets/*/preset.json`` + ``transformer/config.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import (
    PresetMetadata,
    bundled_preset_dir,
    list_bundled_presets,
    name_matches_hints,
    sniff_safetensors_header,
)


@dataclass(frozen=True)
class ModelSpec:
    """Registration descriptor for a QuantFunc transformer architecture."""

    arch: str
    cls: type
    pipeline_classes: tuple[str, ...] = ()
    transformer_classes: tuple[str, ...] = ()
    file_hints: tuple[str, ...] = ()
    default_config_path: Path | None = None
    engine_arch: str = "auto"


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    arch: str,
    *,
    cls: type | None = None,
    pipeline_classes: Sequence[str] = (),
    transformer_classes: Sequence[str] = (),
    file_hints: Sequence[str] = (),
    default_config_path: str | os.PathLike[str] | None = None,
    engine_arch: str = "auto",
) -> Any:
    """Register a transformer model class (usable both as ``@register_model(...)`` and direct call)."""
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError(f"arch must be a non-empty string, got {arch!r}")
    norm_arch = arch.strip().lower()
    if not isinstance(engine_arch, str) or not engine_arch.strip():
        raise ValueError(f"engine_arch must be a non-empty string, got {engine_arch!r}")
    cfg_path = Path(os.path.expanduser(os.fspath(default_config_path))) if default_config_path is not None else None

    def _register(target_cls: type) -> type:
        spec = ModelSpec(
            arch=norm_arch,
            cls=target_cls,
            pipeline_classes=tuple(str(x) for x in pipeline_classes),
            transformer_classes=tuple(str(x) for x in transformer_classes),
            file_hints=tuple(str(x) for x in file_hints),
            default_config_path=cfg_path,
            engine_arch=engine_arch.strip(),
        )
        _REGISTRY[norm_arch] = spec
        return target_cls

    if cls is not None:
        return _register(cls)
    return _register


def unregister_model(arch: str) -> None:
    """Remove a registered architecture (primarily for test isolation)."""
    _REGISTRY.pop(arch.strip().lower(), None)


def get_model_spec(arch: str) -> ModelSpec:
    """Return the registered ``ModelSpec`` for ``arch``."""
    key = arch.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model architecture {arch!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def get_model_class(arch: str) -> type:
    """Return the registered transformer class for ``arch``."""
    return get_model_spec(arch).cls


def list_registered_models() -> dict[str, ModelSpec]:
    """Return a shallow copy of registered ``arch -> ModelSpec`` entries."""
    return dict(_REGISTRY)


def _iter_bundled_presets() -> list[tuple[PresetMetadata, dict[str, Any]]]:
    out: list[tuple[PresetMetadata, dict[str, Any]]] = []
    for arch in list_bundled_presets():
        try:
            pdir = bundled_preset_dir(arch)
            meta = PresetMetadata.from_file(pdir / "preset.json")
            cfg_file = pdir / "transformer" / "config.json"
            cfg: dict[str, Any] = {}
            if cfg_file.is_file():
                with open(cfg_file, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        cfg = loaded
            out.append((meta, cfg))
        except Exception:  # noqa: BLE001
            continue
    return out


def _detect_from_mapping(cfg: Mapping[str, Any]) -> str | None:
    bundled = _iter_bundled_presets()
    known_arches = set(_REGISTRY.keys()) | {m.arch.lower() for m, _ in bundled}

    for key in ("model_arch", "arch"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip().lower() in known_arches:
            return val.strip().lower()

    image_model = cfg.get("image_model")
    if isinstance(image_model, str) and image_model.strip():
        im_norm = image_model.strip().lower()
        if im_norm in known_arches:
            return im_norm
        for meta, pcfg in bundled:
            if str(pcfg.get("image_model", "")).strip().lower() == im_norm:
                return meta.arch.lower()

    class_name = cfg.get("_class_name") or cfg.get("transformer_class")
    if isinstance(class_name, str) and class_name.strip():
        cname = class_name.strip()
        for arch, spec in _REGISTRY.items():
            if cname in spec.transformer_classes or cname in spec.pipeline_classes:
                return arch
        for meta, pcfg in bundled:
            if (
                cname in meta.transformer_classes
                or cname in meta.pipeline_classes
                or str(pcfg.get("_class_name", "")) == cname
            ):
                return meta.arch.lower()

    pipe_name = cfg.get("_pipeline_class") or cfg.get("pipeline_class")
    if isinstance(pipe_name, str) and pipe_name.strip():
        pname = pipe_name.strip()
        for arch, spec in _REGISTRY.items():
            if pname in spec.pipeline_classes:
                return arch
        for meta, _ in bundled:
            if pname in meta.pipeline_classes:
                return meta.arch.lower()

    return None


def _detect_from_filename(path_str: str) -> str | None:
    for arch, spec in _REGISTRY.items():
        if spec.file_hints and name_matches_hints(path_str, spec.file_hints):
            return arch
    for meta, _ in _iter_bundled_presets():
        if meta.file_hints and name_matches_hints(path_str, meta.file_hints):
            return meta.arch.lower()
    return None


def _detect_from_safetensors_header(path: Path) -> str | None:
    try:
        hdr = sniff_safetensors_header(path)
    except (OSError, ValueError):
        return None
    meta_dict = hdr.get("__metadata__")
    if isinstance(meta_dict, dict):
        hit = _detect_from_mapping(meta_dict)
        if hit is not None:
            return hit
    keys = [k for k in hdr if k != "__metadata__"]
    for preset, _ in _iter_bundled_presets():
        if preset.tensor_hints and any(any(hint in k for hint in preset.tensor_hints) for k in keys):
            return preset.arch.lower()
    return None


def detect_model_arch(path_or_config: str | os.PathLike[str] | Mapping[str, Any]) -> str:
    """Detect the QuantFunc model architecture from a config dict, directory, JSON file, or checkpoint path.

    Raises ``ValueError`` if no registered architecture or bundled preset matches.
    """
    if isinstance(path_or_config, Mapping):
        hit = _detect_from_mapping(path_or_config)
        if hit is not None:
            return hit
        raise ValueError("Unable to detect QuantFunc model architecture from config mapping")

    raw = os.fspath(path_or_config)
    p = Path(os.path.expanduser(raw))

    if p.is_dir():
        for rel in ("transformer/config.json", "config.json", "model_index.json"):
            cand = p / rel
            if cand.is_file():
                try:
                    with open(cand, encoding="utf-8") as fh:
                        doc = json.load(fh)
                    if isinstance(doc, dict):
                        hit = _detect_from_mapping(doc)
                        if hit is not None:
                            return hit
                except (OSError, ValueError):
                    pass
        for st_file in sorted(p.glob("*.safetensors")):
            hit = _detect_from_filename(str(st_file)) or _detect_from_safetensors_header(st_file)
            if hit is not None:
                return hit
        raise ValueError(f"Unable to detect QuantFunc model architecture from directory {raw!r}")

    if p.is_file() and p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict):
            hit = _detect_from_mapping(doc)
            if hit is not None:
                return hit
        raise ValueError(f"Unable to detect QuantFunc model architecture from config file {raw!r}")

    hit = _detect_from_filename(raw)
    if hit is not None:
        return hit

    if p.is_file() and p.suffix.lower() == ".safetensors":
        hit = _detect_from_safetensors_header(p)
        if hit is not None:
            return hit

    raise ValueError(f"Unable to detect QuantFunc model architecture from {raw!r}")


__all__ = [
    "ModelSpec",
    "register_model",
    "unregister_model",
    "get_model_spec",
    "get_model_class",
    "list_registered_models",
    "detect_model_arch",
]
