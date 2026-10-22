import numpy as np
import pytest

from pipeline.feature_extraction.landmark_extractor import LandmarkExtractionConfig
from s6_synthesis.schema.synthesis_output import SynthesisOutput
from s6_synthesis.synthesis.synthesizer import SynthesizerConfig, synthesize


def _config():
    return LandmarkExtractionConfig(include_face=False, include_hand_local_norm=False)


def _fake_clip(config: LandmarkExtractionConfig, length: int, fill: float) -> np.ndarray:
    # Só a parte estática importa pro lexicon (synthesizer recorta só
    # :static_feature_dim); a "velocidade" fica com qualquer valor —
    # synthesize() a recalcula do zero.
    clip = np.full((length, config.feature_dim), fill, dtype=np.float32)
    return clip


def test_synthesize_single_sign_has_no_transition_segments():
    config = _config()
    lexicon = {"Aluno": _fake_clip(config, 10, 0.1)}
    output = synthesize(["Aluno"], lexicon, SynthesizerConfig(landmarks=config, transition_frames=4))

    assert isinstance(output, SynthesisOutput)
    assert len(output.segments) == 1
    assert output.segments[0].tipo == "sinal"
    assert output.segments[0].sinal == "Aluno"
    assert len(output.keypoints) == 10
    assert len(output.frame_timestamps_ms) == 10


def test_synthesize_two_signs_inserts_transition_segment_between():
    config = _config()
    lexicon = {"Aluno": _fake_clip(config, 10, 0.0), "Banco": _fake_clip(config, 8, 1.0)}
    output = synthesize(["Aluno", "Banco"], lexicon, SynthesizerConfig(landmarks=config, transition_frames=5))

    tipos = [s.tipo for s in output.segments]
    assert tipos == ["sinal", "transicao", "sinal"]
    assert output.segments[0].sinal == "Aluno"
    assert output.segments[1].sinal is None
    assert output.segments[2].sinal == "Banco"

    # tamanho total = clip Aluno (10) + transição (5) + clip Banco (8)
    total_frames = 10 + 5 + 8
    assert len(output.keypoints) == total_frames
    assert len(output.frame_timestamps_ms) == total_frames

    # segmentos cobrem a sequência inteira sem buraco nem sobreposição
    assert output.segments[0].start_frame == 0
    assert output.segments[0].end_frame == output.segments[1].start_frame
    assert output.segments[1].end_frame == output.segments[2].start_frame
    assert output.segments[2].end_frame == total_frames


def test_synthesize_unknown_sign_raises_keyerror():
    config = _config()
    lexicon = {"Aluno": _fake_clip(config, 5, 0.0)}
    with pytest.raises(KeyError):
        synthesize(["SinalQueNaoExiste"], lexicon, SynthesizerConfig(landmarks=config))


def test_synthesize_empty_comando_raises():
    config = _config()
    with pytest.raises(ValueError):
        synthesize([], {}, SynthesizerConfig(landmarks=config))


def test_to_json_roundtrip():
    config = _config()
    lexicon = {"Aluno": _fake_clip(config, 6, 0.0)}
    output = synthesize(["Aluno"], lexicon, SynthesizerConfig(landmarks=config))
    import json

    parsed = json.loads(output.to_json())
    assert parsed["comando"] == ["Aluno"]
    assert parsed["segments"][0]["tipo"] == "sinal"
    assert len(parsed["keypoints"]) == 6
