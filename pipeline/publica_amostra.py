"""
pipeline/publicar_amostra.py
==============================
Teste do I4 no barramento SEM câmera: pega um .parquet de features já
extraído (dataset/features_v2), reconstrói os keypoints e publica com o
mesmo I4BusPublisher que o ao_vivo.py usa.

Uso (na raiz do repo; deixe o `tail` rodando em outro terminal):
    python -m pipeline.publicar_amostra --features-dir dataset/features_v2
    python -m pipeline.publicar_amostra --parquet dataset/features_v2/XXX.parquet --confidence 0.9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ods_bus.client import BusProducer, ensure_topics, topic_for
from pipeline.bus_publisher import I4BusPublisher
from pipeline.feature_extraction.landmark_extractor import (
    load_landmark_sequence,
    sequence_to_keypoint_frames,
)

FPS = 25.0  # o mesmo do pré-processamento do treino


def main():
    ap = argparse.ArgumentParser(description="Publica um sinal do dataset no barramento.")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--features-dir", default=None, help="usa o primeiro .parquet da pasta")
    ap.add_argument("--confidence", type=float, default=0.87)
    ap.add_argument("--source-id", default="cam0")
    ap.add_argument("--size", default="1280x720", help="LxA do quadro de origem")
    args = ap.parse_args()

    path = Path(args.parquet) if args.parquet else next(iter(sorted(Path(args.features_dir).glob("*.parquet"))), None)
    if path is None:
        raise SystemExit("Informe --parquet ou --features-dir com .parquet")

    sequence, video_id, label, signer_id = load_landmark_sequence(path)
    frames = sequence_to_keypoint_frames(sequence)  # usa a config padrão: precisa bater com a da extração
    width, height = (int(v) for v in args.size.lower().split("x"))

    now_ms = int(time.time() * 1000)
    step = 1000 / FPS
    timestamps = [int(now_ms + i * step) for i in range(len(frames))]

    ensure_topics([topic_for("ods.i4.keypoints"), topic_for("ods.i4.sign")])
    bus = I4BusPublisher(BusProducer(), source_id=args.source_id)
    bus.publish_sign(
        label=label,
        confidence=args.confidence,
        keypoint_frames=frames,
        timestamps_ms=timestamps,
        image_size=(width, height),
        signer_id=signer_id,
    )
    print(f"publicado: {video_id} ({label}), {len(frames)} quadros, sessão {bus.session_id}")


if __name__ == "__main__":
    main()