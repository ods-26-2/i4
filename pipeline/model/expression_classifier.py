"""
expression_classifier.py
=========================
Classificador de marcadores faciais gramaticais por janela de tempo
(GRU bidirecional pequena sobre as features de `expression_features`) e o
predictor que transforma uma sequência de keypoints em segmentos
(marcador, início, fim, confiança) prontos pro evento `ods.i4.facial_marker`.

Modelo pequeno de propósito: a entrada são 14 números por frame (não os 92
pontos brutos) e o dataset vai ser gravado por nós, poucas pessoas.

Obs.: depende de torch; a lógica sem torch (janelas, segmentos, features)
está em `expression_windows.py` e `expression_features.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from pipeline.feature_extraction.expression_features import FEATURE_NAMES, N_FEATURES, sequence_features
from pipeline.model.expression_windows import Segment, segments_from_probs, sliding_windows


@dataclass
class ExpressionConfig:
    num_classes: int
    input_dim: int = N_FEATURES
    hidden_dim: int = 64
    dropout: float = 0.2
    win: int = 30        # frames por janela (~1,2 s a 25 fps)
    stride: int = 5
    fps: float = 25.0    # fps efetivo das features (25 / frame_stride do dataset)


class ExpressionClassifier(nn.Module):
    def __init__(self, config: ExpressionConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim)
        self.gru = nn.GRU(config.input_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(2 * config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, win, D) -> logits (B, C)
        out, _ = self.gru(self.norm(x))
        return self.head(self.drop(out.mean(dim=1)))


def save_checkpoint(model: ExpressionClassifier, class_names: Sequence[str], path: str | Path, extra: Optional[dict] = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "class_names": list(class_names),
            "feature_names": list(FEATURE_NAMES),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ExpressionClassifier, list[str]]:
    ckpt = torch.load(path, map_location=device)
    model = ExpressionClassifier(ExpressionConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, list(ckpt["class_names"])


class ExpressionPredictor:
    """sequência de keypoints (T, feature_dim) + timestamps -> segmentos."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = device
        self.model, self.class_names = load_checkpoint(checkpoint, device)
        self.config = self.model.config

    @torch.no_grad()
    def window_probs(self, sequence: np.ndarray, baseline: Optional[np.ndarray] = None):
        feats = sequence_features(sequence, baseline=baseline, fps=self.config.fps)
        windows, starts = sliding_windows(feats, self.config.win, self.config.stride)
        if not starts:
            return np.empty((0, len(self.class_names))), []
        logits = self.model(torch.from_numpy(windows).float().to(self.device))
        return torch.softmax(logits, dim=-1).cpu().numpy(), starts

    def predict_segments(
        self,
        sequence: np.ndarray,
        timestamps_ms: Sequence[int],
        baseline: Optional[np.ndarray] = None,
        **segment_kwargs,
    ) -> list[Segment]:
        probs, starts = self.window_probs(sequence, baseline)
        return segments_from_probs(
            probs, starts, self.config.win, timestamps_ms, self.class_names, **segment_kwargs
        )