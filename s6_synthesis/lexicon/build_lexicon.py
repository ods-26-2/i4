"""
build_lexicon.py
==================
Monta o léxico de trajetórias de S6 a partir das sequências de
keypoints que I4 já extrai (`dataset/features_v2/*.parquet`, geradas
por `pipeline.feature_extraction.build_dataset_features`). S6 não
precisa de um dataset próprio pra "trajetória" — a sequência de
keypoints de um sinal isolado JÁ É a trajetória de referência pra
executar esse sinal; é o mesmo material que treina o classificador de
I4, só que usado aqui como fonte de movimento em vez de rótulo a prever.

Por sinal, escolhe UMA amostra "canônica" entre as repetições
disponíveis (várias por sinalizador x repetição): a de duração mais
próxima da mediana da classe. É uma heurística simples — não a média
alinhada (precisaria de DTW, fora de escopo aqui) — mas evita escolher
um outlier (execução truncada, muito mais longa/curta que o normal).

O que falta pra S6 de verdade (e isso sim precisaria de dataset
próprio): frases contínuas, pra validar se a transição sintética entre
sinais (ver `synthesis/transition.py`) se parece com transição real.
MINDS-Libras só tem sinais isolados.

Uso:
    python -m s6_synthesis.lexicon.build_lexicon \
        --features-dir dataset/features_v2 --output-dir lexicon
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from pipeline.feature_extraction.landmark_extractor import (
    load_landmark_sequence,
    save_landmark_sequence,
)


def pick_canonical_sample(paths: list[Path]) -> Path:
    """Dentre as repetições de um mesmo sinal, escolhe a de duração (nº
    de frames) mais próxima da mediana da classe."""
    lengths = []
    for p in paths:
        sequence, *_ = load_landmark_sequence(p)
        lengths.append((len(sequence), p))
    lengths.sort()
    median_len = lengths[len(lengths) // 2][0]
    return min(lengths, key=lambda item: abs(item[0] - median_len))[1]


def build_lexicon(features_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(features_dir.glob("*.parquet")):
        _, _, class_label, _ = load_landmark_sequence(path)
        by_class[class_label].append(path)

    saved: dict[str, Path] = {}
    for class_label, paths in sorted(by_class.items()):
        canonical_path = pick_canonical_sample(paths)
        sequence, video_id, _, signer_id = load_landmark_sequence(canonical_path)
        # Reaproveita o MESMO formato parquet de I4 (class/signer_id/
        # video_id/frame_idx/feat_*) — o léxico de S6 é só mais uma
        # instância desse contrato, não um formato novo. signer_id vira
        # "lexicon" e video_id guarda de qual amostra original veio
        # (rastreabilidade: qual vídeo originou o clipe canônico).
        out_path = save_landmark_sequence(
            video_id=f"lexicon-{class_label}",
            class_label=class_label,
            signer_id=f"lexicon(origem={signer_id}:{video_id})",
            sequence=sequence,
            output_dir=output_dir,
        )
        saved[class_label] = out_path
        print(f"{class_label}: {len(sequence)} frames, canônica = {canonical_path.name}")

    print(f"\nLéxico salvo em {output_dir.resolve()} ({len(saved)} sinais)")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Monta o léxico de trajetórias de S6 a partir das features de I4.")
    parser.add_argument("--features-dir", required=True, help="Pasta com os .parquet de I4 (dataset/features_v2)")
    parser.add_argument("--output-dir", default="lexicon")
    args = parser.parse_args()
    build_lexicon(args.features_dir, args.output_dir)


if __name__ == "__main__":
    main()
