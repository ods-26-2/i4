"""
s6_pipeline.py
================
Orquestra S6 ponta a ponta: carrega o léxico de trajetórias (ver
lexicon/build_lexicon.py) e expõe `synthesize(comando)` — o contrato
público que recebe o comando de síntese de A6 e devolve a trajetória +
transições (SynthesisOutput).

Uso:
    python -m s6_synthesis.lexicon.build_lexicon --features-dir dataset/features_v2 --output-dir lexicon
    python -m s6_synthesis.s6_pipeline --lexicon-dir lexicon --comando Aluno Banco Cinco
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    load_landmark_sequence,
)
from s6_synthesis.schema.synthesis_output import SynthesisOutput
from s6_synthesis.synthesis.synthesizer import SynthesizerConfig, synthesize


def load_lexicon(lexicon_dir: str | Path) -> dict[str, np.ndarray]:
    lexicon_dir = Path(lexicon_dir)
    lexicon: dict[str, np.ndarray] = {}
    for path in sorted(lexicon_dir.glob("*.parquet")):
        sequence, _, class_label, _ = load_landmark_sequence(path)
        lexicon[class_label] = sequence
    if not lexicon:
        raise ValueError(f"Nenhum sinal encontrado em {lexicon_dir} — rode build_lexicon.py primeiro")
    return lexicon


class S6SynthesisPipeline:
    """Contrato público de S6: comando (lista de sinais) -> trajetória +
    transições, prontos pra publicar no evento `S6 -> A6`."""

    def __init__(
        self,
        lexicon_dir: str | Path,
        landmarks: Optional[LandmarkExtractionConfig] = None,
        transition_frames: int = 8,
        fps: float = 25.0,
    ):
        self.lexicon = load_lexicon(lexicon_dir)
        self.config = SynthesizerConfig(
            landmarks=landmarks or LandmarkExtractionConfig(),
            transition_frames=transition_frames,
            fps=fps,
        )

    def synthesize(self, comando: list[str]) -> SynthesisOutput:
        return synthesize(comando, self.lexicon, self.config)


def main():
    parser = argparse.ArgumentParser(description="Síntese de trajetória Libras (S6) a partir de um comando de A6.")
    parser.add_argument("--lexicon-dir", default="lexicon")
    parser.add_argument("--comando", nargs="+", required=True, help="Sinais na ordem desejada, ex.: --comando Aluno Banco")
    parser.add_argument("--transition-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    pipeline = S6SynthesisPipeline(
        args.lexicon_dir, transition_frames=args.transition_frames, fps=args.fps
    )
    output = pipeline.synthesize(args.comando)
    print(output.to_json(indent=2))


if __name__ == "__main__":
    main()
