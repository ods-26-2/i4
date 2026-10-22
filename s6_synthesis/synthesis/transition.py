"""
transition.py
===============
Gera a transição sintética entre o fim de um sinal e o começo do
próximo — a parte de "transição" do contrato de S6, que não precisa de
dataset próprio: é interpolação no espaço de landmarks (técnica comum
de blending em animação), não um modelo treinado.

Usa "smoothstep" (easing cúbico) em vez de interpolação linear pura:
começa e termina com velocidade ~0 em relação aos frames vizinhos, o
que evita um "solavanco" na junção sinal->transição->sinal (visível
numa interpolação linear, que tem uma quebra de derivada nas pontas).
"""

from __future__ import annotations

import numpy as np


def _smoothstep(n: int) -> np.ndarray:
    """n pesos em (0, 1), estritamente crescentes, com easing cúbico
    (derivada ~0 nas duas pontas)."""
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]  # exclui os extremos (0 e 1 exatos)
    return t * t * (3 - 2 * t)


def generate_transition(end_frame: np.ndarray, start_frame: np.ndarray, num_frames: int) -> np.ndarray:
    """Interpola `num_frames` frames ENTRE `end_frame` (último frame do
    sinal anterior) e `start_frame` (primeiro frame do próximo sinal),
    sem repetir nenhum dos dois — quem concatena já tem os dois.
    `end_frame`/`start_frame`: vetores 1D da MESMA dimensão (a parte
    estática do vetor de features, sem velocidade — ver synthesizer.py,
    que recalcula velocidade depois de montar a sequência inteira)."""
    if num_frames <= 0:
        return np.zeros((0, end_frame.shape[0]), dtype=end_frame.dtype)
    weights = _smoothstep(num_frames)[:, None]
    return ((1.0 - weights) * end_frame[None, :] + weights * start_frame[None, :]).astype(end_frame.dtype)
