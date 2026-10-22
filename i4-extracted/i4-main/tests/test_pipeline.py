import numpy as np

from pipeline import pipeline as pipeline_module
from pipeline.feature_extraction.landmark_extractor import LandmarkExtractionConfig
from pipeline.schema.i4_output import I4Output
from pipeline.schema.sign_annotation import SignAnnotation


class _FakePredictor:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path

    def predict(self, sequence):
        return "Aluno", 0.87


def test_pipeline_run_orchestrates_stages_and_builds_i4_output(monkeypatch, tmp_path):
    config = LandmarkExtractionConfig(
        include_face=False, include_hand_local_norm=False, include_hand_velocity=False
    )
    fake_frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 25
    fake_sequence = np.zeros((25, config.feature_dim), dtype=np.float32)
    fake_stamps = [1_000 + i * 40 for i in range(25)]  # 25 frames, 40ms apart -> 960ms de janela

    monkeypatch.setattr(
        pipeline_module, "coletar_frames", lambda reader, cfg: (fake_frames, fake_stamps)
    )
    monkeypatch.setattr(pipeline_module, "extract_landmarks", lambda frames, cfg: fake_sequence)
    monkeypatch.setattr(pipeline_module, "SignPredictor", _FakePredictor)
    # Evita abrir a shm de verdade (FrameReader exige o arquivo do server.cpp).
    monkeypatch.setattr(pipeline_module, "FrameReader", lambda *a, **k: object())

    pipe = pipeline_module.LibrasPipeline(model_path=tmp_path / "model.pt")
    pipe.config.landmarks = config
    output = pipe.run(sinalizador_id="Sinalizador07")

    assert isinstance(output, I4Output)
    assert isinstance(output.sign, SignAnnotation)
    assert output.sign.sinal == "Aluno"
    assert output.sign.confianca == 0.87
    assert output.sign.inicio_ms == 0
    assert output.sign.fim_ms == fake_stamps[-1] - fake_stamps[0]
    assert output.sign.sinalizador_id == "Sinalizador07"

    # keypoints e timestamps: um item por frame, consistentes entre si.
    assert len(output.keypoints) == len(fake_frames)
    assert output.frame_timestamps_ms[0] == 0
    assert output.frame_timestamps_ms[-1] == fake_stamps[-1] - fake_stamps[0]

    # marker_layout é o contrato que descreve `keypoints` — tem que bater
    # com os blocos presentes de fato (face/hand_local/velocity desligados
    # nesta config).
    layout_names = {block["name"] for block in output.marker_layout}
    assert layout_names == {"pose", "left_hand_global", "right_hand_global"}
    assert set(output.keypoints[0].keys()) == layout_names
