"""
expression_features.py
=======================
Features para reconhecer os marcadores gramaticais de A6 (INTERROGATIVA_QU,
INTERROGATIVA_SN, NEGACAO, AFIRMACAO, NEUTRO) a partir do que I4 já publica
em `ods.i4.keypoints`: os 33 pontos de `pose` e os 92 de `face`.

Duas fontes, porque os marcadores são de dois tipos:

- Sobrancelha, olho e boca (INTERROGATIVA_QU / _SN): razões geométricas do
  rosto, as mesmas fórmulas de `calculos_ar.py`, reescritas pra funcionar
  só com os 92 pontos do contrato (o `calculos_ar` assume a malha de 478).
- Movimento de cabeça (NEGACAO = balanço, AFIRMACAO = aceno): os 92 pontos
  do rosto NÃO incluem nariz, queixo nem contorno da face, então yaw/pitch
  /roll saem dos pontos de `pose` (nariz, olhos, orelhas), que sempre estão
  no contrato. Movimento só aparece na SEQUÊNCIA: por isso `add_dynamics`
  acrescenta a velocidade de cabeça e sobrancelha.

Convenções (iguais no treino e na inferência, senão o modelo erra):
- Entrada: o vetor (T, feature_dim) já normalizado por centro/largura dos
  ombros, como sai de `extract_landmarks` / `load_landmark_sequence`.
- Distâncias com w = h = 1: as coordenadas já são adimensionais; as razões
  deste módulo só são comparáveis entre si, não com as do `calculos_ar`
  calculadas em pixels (testes comparam os termos que coincidem).
- Frame sem rosto/pose (tudo zero) vira NaN e depois repete o último valor
  válido (ou 0 se o clipe inteiro não tiver).
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    FACE_SEMANTIC_LANDMARK_INDICES,
    HAND_DIM,
    NUM_FACE_LANDMARKS_REDUCED,
    POSE_DIM,
    LandmarkExtractionConfig,
)

# ---- layout do vetor de entrada (ver describe_marker_layout) ---------------
POSE_POINTS = 33
FACE_OFFSET = POSE_DIM + 2 * HAND_DIM  # pose, mão esq. global, mão dir. global, depois face

# Índices do modelo de pose do MediaPipe.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8

# mesh index (0..477) -> posição no array de 92 pontos
_FACE_POS = {mesh_idx: pos for pos, mesh_idx in enumerate(FACE_SEMANTIC_LANDMARK_INDICES)}

# Pontos usados (todos fazem parte dos 92; ver teste).
_EYE_L = (160, 144, 158, 153, 33, 133)
_EYE_R = (385, 380, 387, 373, 362, 263)

BASE_FEATURE_NAMES = (
    "ear_esq",       # abertura do olho esquerdo
    "ear_dir",       # abertura do olho direito
    "mar",           # abertura da boca
    "brow_altura",   # distância sobrancelha-olho / distância interocular (↑ = sobrancelha levantada)
    "brow_juntas",   # distância entre as sobrancelhas / interocular (↓ = franzida)
    "nmar",          # largura da boca / interocular
    "pup",           # bico / projeção dos lábios
    "yaw",           # cabeça virada para um lado (balanço de cabeça = NEGACAO)
    "pitch",         # cabeça para cima/baixo (aceno = AFIRMACAO; QU = cabeça à frente/baixo)
    "roll",          # cabeça inclinada de lado
)
N_BASE = len(BASE_FEATURE_NAMES)

# Colunas do vetor base que ganham velocidade em `add_dynamics`.
DYNAMIC_SOURCES = ("brow_altura", "yaw", "pitch", "roll")
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(f"d_{n}" for n in DYNAMIC_SOURCES)
N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------- geometria

def _d3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _d2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _is_missing(points: np.ndarray) -> bool:
    return points.size == 0 or not np.any(points)


def face_features(face: np.ndarray) -> np.ndarray:
    """(92, 3) -> (7,): ear_esq, ear_dir, mar, brow_altura, brow_juntas, nmar, pup.
    NaN em tudo se o rosto não foi detectado."""
    out = np.full(7, np.nan, dtype=np.float32)
    if face.shape != (NUM_FACE_LANDMARKS_REDUCED, 3) or _is_missing(face):
        return out

    def p(i: int) -> np.ndarray:
        return face[_FACE_POS[i]]

    def ear(ids: Sequence[int]) -> float:
        h = _d2(p(ids[4]), p(ids[5]))
        if h < 1e-6:
            return float("nan")
        return (_d2(p(ids[0]), p(ids[1])) + _d2(p(ids[2]), p(ids[3]))) / (2.0 * h)

    boca_larg_2d = _d2(p(61), p(291))
    mar = float("nan")
    if boca_larg_2d >= 1e-6:
        mar = (_d2(p(13), p(14)) + _d2(p(82), p(87)) + _d2(p(312), p(317))) / (3.0 * boca_larg_2d)
        pup = ((_d2(p(0), p(17)) + _d2(p(13), p(14))) / 2.0) / boca_larg_2d
    else:
        pup = float("nan")

    escala = (_d3(p(33), p(263)) + _d3(p(133), p(362))) / 2.0
    if escala >= 1e-6:
        altura = ((_d3(p(70), p(159)) + _d3(p(300), p(386))) / 2.0) / escala
        # Variante do `calcular_bar`: sem os termos que usam o ponto 9 (não
        # está nos 92). Só pontas (55-285) e arcos (107-336) das sobrancelhas.
        juntas = ((_d3(p(55), p(285)) + _d3(p(107), p(336))) / 2.0) / escala
        nmar = _d3(p(61), p(291)) / escala
    else:
        altura = juntas = nmar = float("nan")

    out[:] = (ear(_EYE_L), ear(_EYE_R), mar, altura, juntas, nmar, pup)
    return out


def head_features(pose: np.ndarray) -> np.ndarray:
    """(33, >=3) -> (3,): yaw, pitch, roll (adimensionais/radianos), a partir
    de nariz, olhos e orelhas do `pose`. NaN se esses pontos faltarem.

    Sinais só precisam ser consistentes: yaw > 0 = nariz mais perto da orelha
    esquerda do modelo; pitch > 0 = nariz abaixo da linha das orelhas
    (cabeça baixa); roll em radianos."""
    out = np.full(3, np.nan, dtype=np.float32)
    if pose.shape[0] < POSE_POINTS:
        return out
    pts = pose[[NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR], :3]
    if not np.all(np.any(pts, axis=1)):
        return out
    nose, eye_l, eye_r, ear_l, ear_r = pts

    d_l = abs(nose[0] - ear_l[0])
    d_r = abs(ear_r[0] - nose[0])
    total = d_l + d_r
    if total < 1e-6:
        return out
    yaw = (d_r - d_l) / total

    ear_dist = _d3(ear_l, ear_r)
    if ear_dist < 1e-6:
        return out
    pitch = (nose[1] - (ear_l[1] + ear_r[1]) / 2.0) / ear_dist
    roll = float(np.arctan2(eye_l[1] - eye_r[1], abs(eye_l[0] - eye_r[0])))
    out[:] = (yaw, pitch, roll)
    return out


def frame_features(pose: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Um frame -> vetor base (N_BASE,), com NaN onde não deu pra medir."""
    return np.concatenate([face_features(face), head_features(pose)])


# --------------------------------------------------------------- sequência

def _ffill(feats: np.ndarray) -> np.ndarray:
    """Repete o último valor válido em cada coluna; o que ficar sem valor vira 0."""
    out = feats.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            col[:] = 0.0
            continue
        idx = np.where(valid, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        first = np.argmax(valid)
        col[:first] = col[first]
    return out


fill_missing = _ffill  # público: usado no treino


def base_features(sequence: np.ndarray, config: Optional[LandmarkExtractionConfig] = None) -> np.ndarray:
    """(T, feature_dim) -> (T, N_BASE) com NaN nos frames sem rosto/pose.
    Exige a face reduzida (92 pontos) e normalizada, como no contrato."""
    config = config or LandmarkExtractionConfig()
    if not config.include_face or config.face_mode != "reduced":
        raise ValueError("expression_features exige face_mode='reduced' (92 pontos)")
    seq = np.asarray(sequence, dtype=np.float32)
    out = np.empty((seq.shape[0], N_BASE), dtype=np.float32)
    face_dim = NUM_FACE_LANDMARKS_REDUCED * 3
    for t in range(seq.shape[0]):
        pose = seq[t, :POSE_DIM].reshape(POSE_POINTS, 4)
        face = seq[t, FACE_OFFSET : FACE_OFFSET + face_dim].reshape(NUM_FACE_LANDMARKS_REDUCED, 3)
        out[t] = frame_features(pose, face)
    return out


def add_dynamics(base: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """(T, N_BASE) sem NaN -> (T, N_FEATURES): acrescenta velocidade (por
    segundo) de sobrancelha, yaw, pitch e roll. É isso que distingue balanço
    e aceno de uma cabeça só virada ou baixa."""
    cols = [BASE_FEATURE_NAMES.index(n) for n in DYNAMIC_SOURCES]
    vel = np.zeros((base.shape[0], len(cols)), dtype=np.float32)
    if base.shape[0] > 1:
        vel[1:] = (base[1:, cols] - base[:-1, cols]) * fps
    return np.concatenate([base, vel], axis=1)


def compute_baseline(base_sequences: Sequence[np.ndarray]) -> np.ndarray:
    """Mediana de cada feature base sobre os frames neutros de UMA pessoa.
    Cada pessoa tem sobrancelha, boca e postura diferentes; sem tirar a
    linha de base, o modelo aprende quem é a pessoa em vez da expressão."""
    stacked = np.concatenate([b for b in base_sequences if len(b)], axis=0)
    base = np.nanmedian(stacked, axis=0)
    return np.nan_to_num(base).astype(np.float32)


def sequence_features(
    sequence: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    fps: float = 25.0,
    config: Optional[LandmarkExtractionConfig] = None,
) -> np.ndarray:
    """Entrada do classificador: (T, feature_dim) -> (T, N_FEATURES).
    Sem `baseline`, usa a mediana do próprio clipe (serve pra teste; no
    treino use o neutro da pessoa, na inferência `RollingBaseline`)."""
    base = base_features(sequence, config)
    if baseline is None:
        baseline = np.nan_to_num(np.nanmedian(base, axis=0)) if len(base) else np.zeros(N_BASE)
    base = _ffill(base)
    base = base - np.asarray(baseline, dtype=np.float32)
    return add_dynamics(base, fps=fps)


class RollingBaseline:
    """Linha de base ao vivo: mediana dos últimos `seconds` de features base
    de quem está na câmera (não sabemos o neutro dele de antemão). Cada
    frame novo entra com `update`; `value` devolve a mediana atual (zeros
    até ter `min_frames`)."""

    def __init__(self, seconds: float = 10.0, fps: float = 25.0, min_frames: int = 25):
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, int(seconds * fps)))
        self._min = min_frames

    def update(self, base_frame: np.ndarray) -> None:
        if not np.any(np.isnan(base_frame)):
            self._buf.append(np.asarray(base_frame, dtype=np.float32))

    @property
    def value(self) -> np.ndarray:
        if len(self._buf) < self._min:
            return np.zeros(N_BASE, dtype=np.float32)
        return np.median(np.stack(self._buf), axis=0).astype(np.float32)