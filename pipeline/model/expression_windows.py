"""
expression_windows.py
======================
Parte do reconhecimento de marcadores faciais que não depende de modelo
(só numpy): cortar a sequência de features em janelas e transformar a
previsão janela-a-janela em SEGMENTOS (marcador, início, fim, confiança),
que é o que o evento `ods.i4.facial_marker` carrega.

Um marcador dura uma fração de segundo até alguns segundos e aparece junto
de vários sinais, então o modelo olha janelas curtas e esta parte junta
janelas vizinhas iguais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Segment:
    marker: str
    confidence: float
    start_ms: int
    end_ms: int


def window_starts(num_frames: int, win: int, stride: int) -> list[int]:
    """Início (em frames) de cada janela. Clipe menor que `win` não gera janela."""
    if num_frames < win:
        return []
    return list(range(0, num_frames - win + 1, stride))


def sliding_windows(features: np.ndarray, win: int, stride: int) -> tuple[np.ndarray, list[int]]:
    """(T, D) -> ((N, win, D), starts)."""
    starts = window_starts(len(features), win, stride)
    if not starts:
        return np.empty((0, win, features.shape[1]), dtype=features.dtype), []
    return np.stack([features[s : s + win] for s in starts]), starts


def segments_from_probs(
    probs: np.ndarray,
    starts: Sequence[int],
    win: int,
    timestamps_ms: Sequence[int],
    class_names: Sequence[str],
    *,
    neutral: str = "NEUTRO",
    min_confidence: float = 0.5,
    min_duration_ms: int = 300,
    max_gap_windows: int = 1,
) -> list[Segment]:
    """Junta janelas consecutivas com o mesmo marcador.

    probs: (N, C) por janela. A janela i cobre os frames [starts[i], starts[i]+win).
    - janela com confiança < `min_confidence` conta como neutra;
    - NEUTRO nunca vira segmento;
    - até `max_gap_windows` janelas fora do marcador no meio ainda juntam
      (uma janela ruim não pode partir um balanço de cabeça em dois);
    - segmento mais curto que `min_duration_ms` é descartado;
    - confiança do segmento = média da probabilidade do marcador nas janelas dele.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[0] == 0:
        return []
    labels: list[Optional[str]] = []
    for p in probs:
        k = int(np.argmax(p))
        name = class_names[k]
        labels.append(None if (name == neutral or p[k] < min_confidence) else name)

    segments: list[Segment] = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        marker = labels[i]
        j, last, gap = i, i, 0
        while j + 1 < n:
            j += 1
            if labels[j] == marker:
                last, gap = j, 0
            elif gap < max_gap_windows and labels[j] is None:
                gap += 1
            else:
                break
        idx = [k for k in range(i, last + 1) if labels[k] == marker]
        col = list(class_names).index(marker)
        start_ms = int(timestamps_ms[starts[i]])
        end_ms = int(timestamps_ms[min(starts[last] + win, len(timestamps_ms)) - 1])
        if end_ms - start_ms >= min_duration_ms:
            segments.append(Segment(marker, float(np.mean(probs[idx, col])), start_ms, end_ms))
        i = last + 1
    return segments