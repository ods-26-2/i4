"""
i4_output.py
=============
Contrato de saída de I4 pronto para o barramento: payload do evento
`ods.i4.keypoints · sign · facial_marker` (ver arquitetura, INV-2),
consumido por S6 (trajetória/síntese) e A6 (Libras). Só A6 recebe (I4
atende só A6 nesta versão da arquitetura).

Antes, `LibrasPipeline.run()` descartava a sequência de landmarks
assim que o classificador terminava de usá-la — o pipeline calculava
keypoints internamente mas nunca os expunha. `I4Output` é o que corrige
isso: embrulha a classificação (`SignAnnotation`, já existente) junto
com os keypoints brutos e a definição dos marcadores que os descreve,
prontos para publicar.

Expressão facial (classificação de expressão propriamente dita) fica
de fora por enquanto — não implementada, depende de dataset próprio
ainda não disponível. O que já sai aqui são os marcadores faciais
brutos (parte do bloco "face" de `keypoints`/`marker_layout`), que é
diferente de uma classificação de expressão.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pipeline.schema.sign_annotation import SignAnnotation


@dataclass(frozen=True)
class I4Output:
    sign: SignAnnotation
    # Um dict por frame, nomeado por marcador (ver
    # landmark_extractor.sequence_to_keypoint_frames) — os "keypoints e
    # marcadores brutos, direto" que vão pro barramento.
    keypoints: list[dict[str, list[dict[str, float]]]]
    # Definição estável de offsets/pontos por marcador (ver
    # landmark_extractor.describe_marker_layout) — o contrato que S6/A6
    # usam pra interpretar `keypoints` sem depender de detalhes internos
    # de I4.
    marker_layout: list[dict[str, Any]]
    # Timestamp (ms, relativo ao 1º frame da janela) de cada frame em
    # `keypoints` — mesmo len que `keypoints`; é o que permite a S6
    # reconstruir trajetória/transições a partir da sequência bruta.
    frame_timestamps_ms: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sign": self.sign.to_dict(),
            "keypoints": self.keypoints,
            "marker_layout": self.marker_layout,
            "frame_timestamps_ms": self.frame_timestamps_ms,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
