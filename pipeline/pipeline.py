"""
pipeline.py
============
Orquestra o pipeline ponta a ponta: lê os frames já tratados da memória
compartilhada (escritos pelo server.cpp) e devolve a anotação estruturada
do sinal reconhecido (etapa [5]).

    shm (frames tratados) -> [3] feature_extraction -> [4] model -> [5] schema

(ANTIGO: vídeo -> [2] preprocessing -> [3] -> [4] -> [5])

Cada etapa só é usada pelo seu contrato público (funções/classes
importadas dos respectivos módulos) — o `pipeline.py` não conhece
detalhes internos de nenhuma delas.

Uso:
    python -m libras_pipeline.pipeline --model model.pt
    # ANTIGO: python -m libras_pipeline.pipeline --video caminho.mp4 --model model.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.feature_extraction.landmark_extractor import (
    LandmarkExtractionConfig,
    describe_marker_layout,
    extract_landmarks,
    sequence_to_keypoint_frames,
)
from pipeline.model.predict import SignPredictor

# leitura de vídeo por caminho (fora da shm) — usada por `run_from_video`,
# pra gerar exemplos/testar offline sem depender do server.cpp/câmera ao
# vivo (ver gerar_exemplos_i4.py).
from pipeline.preprocessing.video_preprocessor import PreprocessConfig, extract_frames
# --- NOVO: leitura dos frames na memória compartilhada (produção) ----------
from pipeline.preprocessing.shm_reader import FrameReader, ShmConfig, coletar_frames
from pipeline.schema.i4_output import I4Output
from pipeline.schema.sign_annotation import SignAnnotation


@dataclass
class LibrasPipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    shm: ShmConfig = field(default_factory=ShmConfig)
    landmarks: LandmarkExtractionConfig = field(default_factory=LandmarkExtractionConfig)


class LibrasPipeline:
    """Orquestra as etapas [2]-[5] pra um único sinal (janela de frames da shm)."""

    def __init__(
        self,
        model_path: str | Path,
        config: Optional[LibrasPipelineConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or LibrasPipelineConfig()
        self.predictor = SignPredictor(model_path, device=device)
        self._reader: Optional[FrameReader] = None  # aberto sob demanda (o server pode subir depois)

    def _get_reader(self) -> FrameReader:
        if self._reader is None:
            self._reader = FrameReader(
                self.config.shm.path, ignore_current=self.config.shm.ignore_current
            )
        return self._reader

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _build_output(
        self, frames: list, sequence, relative_stamps: list[int], sinalizador_id: Optional[str]
    ) -> I4Output:
        label, confidence = self.predictor.predict(sequence)
        annotation = SignAnnotation(
            sinal=label,
            confianca=confidence,
            inicio_ms=0,
            fim_ms=int(relative_stamps[-1]) if relative_stamps else 0,
            sinalizador_id=sinalizador_id,
        )
        # Keypoints e definição dos marcadores: antes descartados assim que
        # o classificador terminava de usar `sequence` — agora saem junto
        # da classificação, prontos pro evento `ods.i4.keypoints · sign ·
        # facial_marker` no barramento (S6, A6).
        return I4Output(
            sign=annotation,
            keypoints=sequence_to_keypoint_frames(sequence, self.config.landmarks),
            marker_layout=describe_marker_layout(self.config.landmarks),
            frame_timestamps_ms=relative_stamps,
        )

    # --- Leitura de vídeo por caminho (fora da shm) --------------------------
    # Não é o caminho de produção (que lê da shm, ver `run`), mas o mesmo
    # contrato de saída (I4Output) — serve pra gerar exemplos do payload do
    # barramento e testar offline sem precisar do server.cpp/câmera.
    def run_from_video(self, video_path: str | Path, sinalizador_id: Optional[str] = None) -> I4Output:
        frames = extract_frames(video_path, self.config.preprocess)
        sequence = extract_landmarks(frames, self.config.landmarks)

        fps = self.config.preprocess.target_fps or 25.0
        relative_stamps = [int(round(i / fps * 1000)) for i in range(len(frames))]
        return self._build_output(frames, sequence, relative_stamps, sinalizador_id)

    # --- NOVO: leitura dos frames na memória compartilhada -------------------
    def run(self, sinalizador_id: Optional[str] = None) -> I4Output:
        reader = self._get_reader()
        # frames já tratados pelo server: sem resample/resize/trim aqui
        frames, stamps = coletar_frames(reader, self.config.shm)
        if not frames:
            raise TimeoutError(
                f"Nenhum frame novo em {self.config.shm.path} "
                f"em {self.config.shm.wait_first_s}s"
            )

        sequence = extract_landmarks(frames, self.config.landmarks)
        # tempos reais vindos do server (ts_ms), relativos ao 1º frame da janela
        relative_stamps = [int(ts - stamps[0]) for ts in stamps]
        return self._build_output(frames, sequence, relative_stamps, sinalizador_id)


def main():
    parser = argparse.ArgumentParser(
        description="Roda o pipeline de reconhecimento de sinais em Libras."
    )
    parser.add_argument(
        "--video", default=None,
        help="Caminho de um vídeo (offline/exemplo). Sem isso, lê da shm (produção).",
    )
    parser.add_argument("--model", required=True, help="Caminho do checkpoint treinado (.pt)")
    parser.add_argument("--shm-path", default="/dev/shm/frames_cam")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames por janela/sinal")
    parser.add_argument("--sinalizador-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full-output", action="store_true",
        help="Imprime o I4Output completo (sign + keypoints + marker_layout), não só `sign`",
    )
    args = parser.parse_args()

    config = LibrasPipelineConfig()
    config.shm.path = args.shm_path
    config.shm.num_frames = args.num_frames

    with LibrasPipeline(args.model, config=config, device=args.device) as pipeline:
        if args.video:
            output = pipeline.run_from_video(args.video, sinalizador_id=args.sinalizador_id)
        else:
            output = pipeline.run(sinalizador_id=args.sinalizador_id)
        print(output.to_json(indent=2) if args.full_output else output.sign.to_json(indent=2))


if __name__ == "__main__":
    main()