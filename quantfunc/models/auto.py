"""Architecture-detecting factory for registered QuantFunc transformers."""

from __future__ import annotations

import os
from typing import Any

from quantfunc.core.registry import detect_model_arch, get_model_class
from quantfunc.models.base import QuantFuncBaseTransformer


def _ensure_builtin_models_registered() -> None:
    """Ensure the four built-in model adapters have registered themselves."""
    import quantfunc.models.krea2  # noqa: F401
    import quantfunc.models.ltx25  # noqa: F401
    import quantfunc.models.minimax_h3  # noqa: F401
    import quantfunc.models.qwenimage21  # noqa: F401


class AutoQuantFuncTransformer:
    """Auto-detecting transformer factory backed by ``quantfunc.core.registry``."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        arch: str | None = None,
        **kwargs: Any,
    ) -> QuantFuncBaseTransformer:
        """Detect architecture (or honor explicit ``arch``) and instantiate the registered model class."""
        _ensure_builtin_models_registered()
        resolved_arch = arch
        if not resolved_arch:
            cfg = kwargs.get("config")
            if cfg is not None:
                try:
                    resolved_arch = detect_model_arch(cfg)
                except ValueError:
                    resolved_arch = detect_model_arch(pretrained_model_name_or_path)
            else:
                resolved_arch = detect_model_arch(pretrained_model_name_or_path)
        model_cls = get_model_class(resolved_arch)
        return model_cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str | os.PathLike[str],
        *,
        arch: str | None = None,
        **kwargs: Any,
    ) -> QuantFuncBaseTransformer:
        """Detect architecture from a single safetensors file and instantiate the registered model."""
        return cls.from_pretrained(pretrained_model_link_or_path, arch=arch, **kwargs)
