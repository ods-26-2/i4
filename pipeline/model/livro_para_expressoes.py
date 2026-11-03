"""
livro_para_expressoes.py
=========================
Converte os .npz do livro (extraídos no Colab, 553 pontos por quadro) para os
.parquet que `train_expressions` lê. Rótulos pela numeração do vídeo:
    197-200 -> NEGACAO | 201-213 -> INTERROGATIVA_QU | 4-29 e 232-252 -> NEUTRO

Uso:
    python -m pipeline.model.livro_para_expressoes --npz-dir PASTA_DOS_NPZ --out dataset/features_expr
"""
import argparse
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    FACE_SEMANTIC_LANDMARK_INDICES as FACE_IDX,
    LandmarkExtractionConfig,
    append_hand_velocity,
    result_to_feature_vector,
    save_landmark_sequence,
)

NEG = range(197, 201)
QU = range(201, 214)
NEUTRO = list(range(4, 30)) + list(range(232, 253))


def rotulo(num: int):
    if num in NEG:
        return "NEGACAO"
    if num in QU:
        return "INTERROGATIVA_QU"
    if num in NEUTRO:
        return "NEUTRO"
    return None


def _lm(a):
    return [NS(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0) for p in a]


def converter(pts: np.ndarray, cfg: LandmarkExtractionConfig) -> np.ndarray:
    """(T, 553, 3) [pose 0-32, mãos, rosto 75-552] ou (T, 125, 3) [pose + 92 do rosto]."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[1] == 553:
        face = pts[:, [75 + i for i in FACE_IDX]]
    else:
        face = pts[:, 33:]
    pose = pts[:, :33]
    rows = []
    for p, f in zip(pose, face):
        if np.isnan(p).any() or np.isnan(f).any():
            rows.append(np.zeros(cfg.static_feature_dim, np.float32))
            continue
        full = np.zeros((478, 3), np.float32)
        full[list(FACE_IDX)] = f
        r = NS(pose_landmarks=_lm(p), left_hand_landmarks=None, right_hand_landmarks=None,
               face_landmarks=_lm(full))
        rows.append(result_to_feature_vector(r, cfg))
    return append_hand_velocity(np.stack(rows), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", default="dataset/features_expr")
    args = ap.parse_args()
    cfg = LandmarkExtractionConfig()
    n = 0
    for p in sorted(Path(args.npz_dir).glob("*.npz")):
        m = re.match(r"(\d+)_", p.name)
        lab = rotulo(int(m.group(1))) if m else None
        if lab is None:
            continue
        d = np.load(p, allow_pickle=True)
        pts = d["landmarks"] if "landmarks" in d.files else d["pts"]
        seq = converter(pts, cfg)
        # todos são da mesma pessoa; 4 grupos só pra o validador poder separar
        # vídeos (validação otimista: não mede generalização entre pessoas)
        signer = f"livro_{int(m.group(1)) % 4}"
        save_landmark_sequence(p.stem, lab, signer, seq, args.out)
        n += 1
    print(n, "clipes salvos em", args.out)


if __name__ == "__main__":
    main()