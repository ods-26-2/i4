"""
classifier.py
==============
Arquitetura do classificador de sinais: uma LSTM (bidirecional por
padrão) sobre a sequência de landmarks, seguida de uma cabeça linear
que classifica a partir do último estado oculto. Inclui também
serialização (save/load) do checkpoint treinado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class ClassifierConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = True
    # Normaliza cada feature de entrada (média/variância aprendidas) antes
    # da LSTM. Sem isso, blocos de escalas bem diferentes (pose tem
    # visibility em [0,1], mãos/face são coordenadas normalizadas pelo
    # corpo, velocidade de mão é a diferença entre essas) competem de forma
    # desigual nos pesos iniciais da LSTM — LayerNorm resolve isso sem
    # exigir normalização manual por bloco.
    input_layer_norm: bool = True
    # Pooling por atenção sobre todos os timesteps, em vez de só o último
    # estado oculto. O último estado comprime a sequência inteira num
    # vetor só; com sinais de duração variável (ver temporal_jitter em
    # augmentation.py) e padding, deixar o modelo aprender QUAIS frames
    # importam mais (tipicamente o pico do movimento) tende a generalizar
    # melhor do que depender só da recorrência pra isso.
    attention_pooling: bool = True


class _AttentionPool(nn.Module):
    """Pooling por atenção aditiva sobre a sequência de saídas da LSTM,
    mascarando os timesteps de padding (ver `lengths` em `forward`)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # outputs: (B, T, dim); mascara posições >= length de cada amostra.
        mask = torch.arange(outputs.size(1), device=outputs.device)[None, :] < lengths[:, None].to(outputs.device)
        scores = self.score(outputs).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (outputs * weights).sum(dim=1)  # (B, dim)


class SignLSTMClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim) if config.input_layer_norm else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.pool = _AttentionPool(out_dim) if config.attention_pooling else None
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, config.num_classes),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        sequences: (B, T, F) já com padding.
        lengths: (B,) comprimento real de cada sequência (antes do padding).
        Retorna logits (B, num_classes).
        """
        sequences = self.input_norm(sequences)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.pool is not None:
            packed_out, _ = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            pooled = self.pool(outputs, lengths)
        else:
            _, (h_n, _) = self.lstm(packed)
            if self.config.bidirectional:
                pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                pooled = h_n[-1]
        return self.classifier(pooled)


def save_checkpoint(
    model: SignLSTMClassifier,
    label_classes: list[str],
    path: str | Path,
    meta: dict | None = None,
) -> None:
    """`meta` guarda como o modelo foi treinado (ex.: frame_stride), pra a
    inferência aplicar o mesmo pré-processamento sem depender de quem chama."""
    torch.save(
        {
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "label_classes": label_classes,
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint_meta(path: str | Path, map_location: str = "cpu") -> dict:
    """Metadados de treino do checkpoint ({} em checkpoints antigos, sem meta)."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint.get("meta", {})


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[SignLSTMClassifier, list[str]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = ClassifierConfig(**checkpoint["config"])
    model = SignLSTMClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["label_classes"]
