"""
synthesis_output.py
=====================
Contrato de saída de S6 pra A6: o evento `S6 -> A6` (ver arquitetura,
"Comando de síntese enviado por A6; marcadores faciais na saída").
A6 manda o comando (lista de sinais a executar, em ordem) e S6 devolve
a trajetória completa já com as transições emendadas, mais a
decomposição em segmentos (qual trecho de frames é sinal de verdade e
qual é transição sintética) — isso é o que o requisito de arquitetura
chama de "trajetória e transições".

Reaproveita o MESMO contrato de marcadores de I4 (`keypoints` +
`marker_layout`, ver pipeline.schema.i4_output / landmark_extractor):
a saída de S6 é mais longa (vários sinais emendados) mas descreve os
keypoints do jeito que A6 já vai estar acostumado a ler de I4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SynthesisSegment:
    tipo: str  # "sinal" ou "transicao"
    sinal: Optional[str]  # nome do sinal (None pra segmentos de transição)
    start_frame: int
    end_frame: int  # exclusivo, como range()
    inicio_ms: int
    fim_ms: int

    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "sinal": self.sinal,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "inicio_ms": self.inicio_ms,
            "fim_ms": self.fim_ms,
        }


@dataclass(frozen=True)
class SynthesisOutput:
    comando: list[str]  # sinais pedidos por A6, na ordem
    segments: list[SynthesisSegment]  # trajetória (segmentos "sinal") + transições
    keypoints: list[dict[str, list[dict[str, float]]]]  # mesmo formato de I4Output.keypoints
    marker_layout: list[dict[str, Any]]  # mesmo formato de I4Output.marker_layout
    frame_timestamps_ms: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "comando": self.comando,
            "segments": [s.to_dict() for s in self.segments],
            "keypoints": self.keypoints,
            "marker_layout": self.marker_layout,
            "frame_timestamps_ms": self.frame_timestamps_ms,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
