"""
synthesizer.py
================
Orquestra a síntese: recebe o comando de A6 (lista de sinais, na ordem
desejada) e o léxico de trajetórias (ver lexicon/build_lexicon.py) e
devolve a trajetória completa — clipe do sinal 1, transição sintética,
clipe do sinal 2, transição, ... — como um único SynthesisOutput.

Detalhe importante: a interpolação de transição (transition.py) só
mexe na parte ESTÁTICA do vetor de features (pose/mãos/rosto —
`config.static_feature_dim` primeiras colunas). A velocidade de mão
(últimas colunas, ver landmark_extractor.append_hand_velocity) NÃO é
interpolada ponto a ponto — ela é RECALCULADA depois de montar a
sequência inteira, com a mesma função que I4 usa. Isso garante que a
velocidade nas bordas de uma transição reflita o movimento de verdade
entre os frames vizinhos (coerente), em vez de uma mistura linear de
duas velocidades que não tem significado físico direto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    append_hand_velocity,
    describe_marker_layout,
    sequence_to_keypoint_frames,
)
from s6_synthesis.schema.synthesis_output import SynthesisOutput, SynthesisSegment
from s6_synthesis.synthesis.transition import generate_transition


@dataclass
class SynthesizerConfig:
    landmarks: LandmarkExtractionConfig
    transition_frames: int = 8
    fps: float = 25.0  # mesmo fps usado por build_dataset_features (etapa [2] de I4)


def synthesize(
    comando: list[str],
    lexicon: dict[str, np.ndarray],
    config: SynthesizerConfig,
) -> SynthesisOutput:
    """`lexicon`: {sinal: sequência (T, feature_dim)}, como devolvido por
    `s6_synthesis.s6_pipeline.load_lexicon`. Lança KeyError se algum
    sinal do comando não estiver no léxico — decisão deliberada: S6 não
    deve inventar um sinal que não existe, melhor falhar alto."""
    if not comando:
        raise ValueError("comando vazio — nada pra sintetizar")

    static_dim = config.landmarks.static_feature_dim
    pieces: list[np.ndarray] = []
    segments: list[SynthesisSegment] = []
    frame_cursor = 0

    def _ms(frame_idx: int) -> int:
        return int(round(frame_idx / config.fps * 1000))

    for i, sinal in enumerate(comando):
        if sinal not in lexicon:
            raise KeyError(f"Sinal '{sinal}' não está no léxico de S6")
        clip = lexicon[sinal][:, :static_dim]

        if i > 0:
            prev_end = pieces[-1][-1]
            transition = generate_transition(prev_end, clip[0], config.transition_frames)
            if len(transition):
                pieces.append(transition)
                segments.append(SynthesisSegment(
                    tipo="transicao", sinal=None,
                    start_frame=frame_cursor, end_frame=frame_cursor + len(transition),
                    inicio_ms=_ms(frame_cursor), fim_ms=_ms(frame_cursor + len(transition)),
                ))
                frame_cursor += len(transition)

        pieces.append(clip)
        segments.append(SynthesisSegment(
            tipo="sinal", sinal=sinal,
            start_frame=frame_cursor, end_frame=frame_cursor + len(clip),
            inicio_ms=_ms(frame_cursor), fim_ms=_ms(frame_cursor + len(clip)),
        ))
        frame_cursor += len(clip)

    stitched_static = np.concatenate(pieces, axis=0)
    stitched = append_hand_velocity(stitched_static, config.landmarks)
    timestamps = [_ms(i) for i in range(len(stitched))]

    return SynthesisOutput(
        comando=comando,
        segments=segments,
        keypoints=sequence_to_keypoint_frames(stitched, config.landmarks),
        marker_layout=describe_marker_layout(config.landmarks),
        frame_timestamps_ms=timestamps,
    )
