"""Qwen-Image-2.1 pipeline loader and pre-norm text-encoder extraction helper."""

from __future__ import annotations

from typing import Any

import torch

from quantfunc.models.qwenimage21 import QuantFuncQwenImage21Transformer2DModel
from quantfunc.pipelines._common import finalize_pipeline_load, import_diffusers_class

_PRE_NORM_PROBE_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "language_model", "norm"),
    ("language_model", "norm"),
    ("model", "norm"),
)


def _resolve_norm_module(text_encoder: Any) -> Any:
    for path in _PRE_NORM_PROBE_PATHS:
        cur: Any = text_encoder
        found = True
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                found = False
                break
        if found and callable(getattr(cur, "register_forward_pre_hook", None)):
            return cur

    probed = ", ".join(".".join(p) for p in _PRE_NORM_PROBE_PATHS)
    raise AttributeError(
        f"Could not locate the final normalization layer on {type(text_encoder).__name__} "
        f"(probed: {probed})."
    )


def encode_qwen3_vl_pre_norm(
    text_encoder: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``text_encoder`` and return the pre-norm hidden states entering its final norm layer.

    Probes ``model.language_model.norm`` -> ``language_model.norm`` -> ``model.norm``
    in order, registers a temporary forward pre-hook, and always removes the hook in
    a ``finally`` block. Fails closed if no final norm module is found or if the
    hook is never triggered.
    """
    norm_module = _resolve_norm_module(text_encoder)
    captured: list[torch.Tensor] = []

    def _capture_pre_norm(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured.append(inputs[0])

    handle = norm_module.register_forward_pre_hook(_capture_pre_norm)
    try:
        text_encoder(*args, **kwargs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            "Pre-norm forward hook was not triggered during text_encoder forward pass."
        )
    return captured[-1]


def load_qwenimage21_pipeline(
    model_id_or_path: str,
    *,
    transformer_path: str | None = None,
    torch_dtype: torch.dtype | None = None,
    attention_backend: str = "auto",
    offload_mode: str | None = None,
    attach_hooks: bool = True,
    variant: str = "t2i",
    pipeline_cls: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Load a Qwen-Image-2.1 T2I or Edit pipeline with a QuantFunc transformer."""
    if variant not in ("t2i", "edit"):
        raise ValueError(f"Unsupported Qwen-Image-2.1 variant {variant!r}; expected 't2i' or 'edit'.")

    if pipeline_cls is not None:
        cls = pipeline_cls
    elif variant == "edit":
        cls = import_diffusers_class(
            "diffusers",
            "QwenImage21EditPipeline",
            fallback_names=("QwenImageEditPipeline",),
            repo_hint="https://github.com/QwenLM/Qwen-Image",
        )
    else:
        cls = import_diffusers_class(
            "diffusers",
            "QwenImage21Pipeline",
            fallback_names=("QwenImagePipeline",),
            repo_hint="https://github.com/QwenLM/Qwen-Image",
        )

    transformer = kwargs.pop("transformer", None)
    if transformer is None:
        transformer = QuantFuncQwenImage21Transformer2DModel.from_pretrained(
            transformer_path or model_id_or_path,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
        )

    pipe_kwargs: dict[str, Any] = dict(kwargs)
    if torch_dtype is not None:
        pipe_kwargs["torch_dtype"] = torch_dtype
    pipe = cls.from_pretrained(model_id_or_path, transformer=transformer, **pipe_kwargs)

    return finalize_pipeline_load(pipe, attach_hooks=attach_hooks, offload=offload_mode)
