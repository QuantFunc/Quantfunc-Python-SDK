"""Container for single-stream or audio-visual latent tensors passed between stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatentBundle:
    """Intermediate latent payload carried between sampling stages.

    Attributes:
        video_or_image: Primary visual latent tensor (image or video).
        audio: Optional secondary audio latent tensor.
        meta: Optional stage metadata (e.g., spatial compression or prior size).
    """

    video_or_image: Any
    audio: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}
        elif not isinstance(self.meta, dict):
            self.meta = dict(self.meta)
