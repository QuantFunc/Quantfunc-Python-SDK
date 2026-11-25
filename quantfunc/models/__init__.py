"""L2 drop-in QuantFunc transformer adapters and auto-discovery factory."""

from __future__ import annotations

from quantfunc.models.auto import AutoQuantFuncTransformer
from quantfunc.models.base import QuantFuncBaseTransformer, Transformer2DModelOutput
from quantfunc.models.krea2 import QuantFuncKrea2Transformer2DModel
from quantfunc.models.ltx25 import QuantFuncLTX25Transformer3DModel, restore_frozen_audio
from quantfunc.models.minimax_h3 import QuantFuncMiniMaxH3Transformer3DModel
from quantfunc.models.qwenimage21 import QuantFuncQwenImage21Transformer2DModel

__all__ = [
    "AutoQuantFuncTransformer",
    "QuantFuncBaseTransformer",
    "QuantFuncKrea2Transformer2DModel",
    "QuantFuncLTX25Transformer3DModel",
    "QuantFuncMiniMaxH3Transformer3DModel",
    "QuantFuncQwenImage21Transformer2DModel",
    "Transformer2DModelOutput",
    "restore_frozen_audio",
]
