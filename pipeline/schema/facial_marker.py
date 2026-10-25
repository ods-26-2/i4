"""
facial_marker.py
=================
Payload do evento `ods.i4.facial_marker` (arquitetura 6c, EV-SINAL): um
marcador facial gramatical reconhecido, com intervalo e confiança. O
vocabulário de `marker` NÃO é nosso: vem da lista `markers` do artefato
LEXICON de A6 (lexicon-a6-1.3: INTERROGATIVA_QU, INTERROGATIVA_SN, NEGACAO,
AFIRMACAO, NEUTRO). Por isso o conjunto aceito é passado por quem publica
e vem do LEXICON carregado, não fixo no código; `DEFAULT_MARKERS` só existe
pra testes e pra rodar sem o LEXICON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Optional

DEFAULT_MARKERS = ("INTERROGATIVA_QU", "INTERROGATIVA_SN", "NEGACAO", "AFIRMACAO", "NEUTRO")

# Marcador que A6 descreve como "Sem marcador". Ausência de marcador não é
# evento: o publicador não envia NEUTRO (confirmar com A6 se querem receber).
NEUTRAL_MARKER = "NEUTRO"


@dataclass(frozen=True)
class FacialMarker:
    marker: str
    confidence: float
    start_ms: int  # epoch ms ABSOLUTO (mesma base dos timestamps dos keypoints)
    end_ms: int

    def validate(self, allowed: Optional[Collection[str]] = None) -> None:
        allowed = DEFAULT_MARKERS if allowed is None else allowed
        if self.marker not in allowed:
            raise ValueError(f"marcador fora do conjunto fechado de A6: {self.marker!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence fora de [0, 1]: {self.confidence}")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms deve ser maior que start_ms")

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "confidence": round(float(self.confidence), 4),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
        }