from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.feature_extraction.landmark_extractor import (
    FACE_DIM_FULL,
    FACE_DIM_REDUCED,
    FACE_SEMANTIC_LANDMARK_INDICES,
    HAND_DIM,
    NUM_HAND_LANDMARKS,
    POSE_DIM,
    LandmarkExtractionConfig,
    _flatten,
    _flatten_by_indices,
    _hand_local_normalize,
    append_hand_velocity,
    describe_marker_layout,
    load_landmark_sequence,
    result_to_feature_vector,
    save_landmark_sequence,
    sequence_to_keypoint_frames,
)


def _lm(x, y, z, visibility=None):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _no_extra_features_config(**overrides):
    """Config com as features novas (hand-local-norm, hand-velocity)
    desligadas, pra testar o esqueleto original isoladamente."""
    return LandmarkExtractionConfig(include_hand_local_norm=False, include_hand_velocity=False, **overrides)


def test_feature_dim_reduced_face_is_default_and_much_smaller_than_full():
    reduced = _no_extra_features_config(include_face=True, face_mode="reduced").feature_dim
    full = _no_extra_features_config(include_face=True, face_mode="full").feature_dim
    no_face = _no_extra_features_config(include_face=False).feature_dim

    assert reduced == POSE_DIM + 2 * HAND_DIM + FACE_DIM_REDUCED
    assert full == POSE_DIM + 2 * HAND_DIM + FACE_DIM_FULL
    assert no_face == POSE_DIM + 2 * HAND_DIM
    assert reduced < full
    assert len(FACE_SEMANTIC_LANDMARK_INDICES) < 478


def test_feature_dim_includes_hand_local_norm_and_velocity_by_default():
    config = LandmarkExtractionConfig(include_face=False)
    assert config.include_hand_local_norm is True
    assert config.include_hand_velocity is True
    # pose + hands(global, local, velocity) x 2 mãos cada
    assert config.feature_dim == POSE_DIM + 3 * 2 * HAND_DIM
    assert config.static_feature_dim == config.feature_dim - config.hand_velocity_dim


def test_hand_local_normalize_centers_on_wrist_and_scales_by_hand_span():
    # pulso em (1, 1, 0); ponta mais distante a 2 unidades de distância em x
    landmarks = [_lm(1, 1, 0)] + [_lm(3, 1, 0)] + [_lm(1, 1, 0)] * (NUM_HAND_LANDMARKS - 2)
    normalized = _hand_local_normalize(landmarks)
    assert normalized[0].x == 0.0 and normalized[0].y == 0.0  # pulso vira a origem
    assert normalized[1].x == 1.0  # ponto mais distante normalizado pra escala 1


def test_hand_local_normalize_passthrough_when_empty():
    assert _hand_local_normalize(None) is None
    assert _hand_local_normalize([]) == []


def test_append_hand_velocity_first_frame_is_zero_and_later_frames_are_deltas():
    config = LandmarkExtractionConfig(include_face=False, include_hand_local_norm=False)
    sequence = np.zeros((3, config.static_feature_dim), dtype=np.float32)
    lo = config.left_hand_global_offset
    sequence[1, lo] = 0.5  # frame 1: mão esquerda andou 0.5 no eixo x
    sequence[2, lo] = 0.2  # frame 2: voltou um pouco

    out = append_hand_velocity(sequence, config)
    assert out.shape == (3, config.feature_dim)
    velocity_lo = config.static_feature_dim  # velocidade começa logo após o estático
    np.testing.assert_allclose(out[0, velocity_lo : velocity_lo + 2], [0.0, 0.0])
    assert out[1, velocity_lo] == pytest.approx(0.5, abs=1e-6)
    assert out[2, velocity_lo] == pytest.approx(-0.3, abs=1e-6)


def test_append_hand_velocity_noop_when_disabled():
    config = LandmarkExtractionConfig(include_hand_velocity=False, include_hand_local_norm=False, include_face=False)
    sequence = np.random.rand(4, config.static_feature_dim).astype(np.float32)
    out = append_hand_velocity(sequence, config)
    assert out is sequence


def test_describe_marker_layout_offsets_cover_feature_dim_exactly():
    config = LandmarkExtractionConfig(include_face=True, face_mode="reduced")
    layout = describe_marker_layout(config)
    covered = sum(block["num_points"] * block["dims_per_point"] for block in layout)
    assert covered == config.feature_dim
    names = [block["name"] for block in layout]
    assert names == [
        "pose", "left_hand_global", "right_hand_global", "face",
        "left_hand_local", "right_hand_local",
        "left_hand_velocity", "right_hand_velocity",
    ]


def test_sequence_to_keypoint_frames_matches_layout_and_point_counts():
    config = _no_extra_features_config(include_face=False)
    sequence = np.random.rand(2, config.feature_dim).astype(np.float32)
    frames = sequence_to_keypoint_frames(sequence, config)
    assert len(frames) == 2
    assert len(frames[0]["pose"]) == 33
    assert set(frames[0]["pose"][0].keys()) == {"x", "y", "z", "visibility"}
    assert len(frames[0]["left_hand_global"]) == 21
    assert set(frames[0]["left_hand_global"][0].keys()) == {"x", "y", "z"}


def test_flatten_by_indices_reads_only_requested_points():
    landmarks = [_lm(i / 100, i / 100, i / 100) for i in range(10)]
    vec = _flatten_by_indices(landmarks, indices=[2, 5], include_visibility=False)
    assert vec.shape == (6,)
    np.testing.assert_allclose(vec, [0.02, 0.02, 0.02, 0.05, 0.05, 0.05], rtol=1e-6)


def test_flatten_by_indices_empty_landmarks_returns_zeros():
    vec = _flatten_by_indices(None, indices=[0, 1, 2], include_visibility=False)
    assert vec.shape == (9,)
    assert np.all(vec == 0.0)


def test_flatten_pads_missing_points_with_zeros():
    vec = _flatten([_lm(0.1, 0.2, 0.3)], num_points=3, include_visibility=False)
    assert vec.shape == (9,)
    np.testing.assert_allclose(vec[:3], [0.1, 0.2, 0.3], rtol=1e-6)
    assert vec[3:].tolist() == [0.0] * 6


def test_flatten_empty_landmarks_returns_zeros():
    vec = _flatten(None, num_points=2, include_visibility=True)
    assert vec.shape == (8,)
    assert np.all(vec == 0.0)


def test_flatten_includes_visibility_when_requested():
    vec = _flatten([_lm(0.1, 0.2, 0.3, visibility=0.75)], num_points=1, include_visibility=True)
    np.testing.assert_allclose(vec, [0.1, 0.2, 0.3, 0.75], rtol=1e-6)


def test_result_to_feature_vector_concatenates_components():
    # result_to_feature_vector devolve o vetor ESTÁTICO (sem velocidade de
    # mão, que depende do frame anterior e é anexada por
    # append_hand_velocity/extract_landmarks) — por isso compara com
    # static_feature_dim, não feature_dim.
    config = LandmarkExtractionConfig(include_face=False)
    result = SimpleNamespace(
        pose_landmarks=[_lm(0.1, 0.1, 0.1, 0.9)] * 33,
        left_hand_landmarks=[_lm(0.2, 0.2, 0.2)] * 21,
        right_hand_landmarks=[_lm(0.3, 0.3, 0.3)] * 21,
        face_landmarks=[],
    )
    vec = result_to_feature_vector(result, config)
    assert vec.shape == (config.static_feature_dim,)


def test_result_to_feature_vector_handles_missing_hand():
    config = LandmarkExtractionConfig(include_face=False)
    result = SimpleNamespace(
        pose_landmarks=[_lm(0.1, 0.1, 0.1, 0.9)] * 33,
        left_hand_landmarks=[],
        right_hand_landmarks=[_lm(0.3, 0.3, 0.3)] * 21,
        face_landmarks=[],
    )
    vec = result_to_feature_vector(result, config)
    assert vec.shape == (config.static_feature_dim,)
    left_hand_slice = vec[POSE_DIM : POSE_DIM + HAND_DIM]
    assert np.all(left_hand_slice == 0.0)


def test_save_and_load_landmark_sequence_roundtrip(tmp_path):
    sequence = np.random.rand(5, 12).astype(np.float32)
    path = save_landmark_sequence(
        video_id="01AcontecerSinalizador01-1.mp4",
        class_label="Acontecer",
        signer_id="Sinalizador01",
        sequence=sequence,
        output_dir=tmp_path,
    )
    loaded_sequence, video_id, class_label, signer_id = load_landmark_sequence(path)
    assert video_id == "01AcontecerSinalizador01-1.mp4"
    assert class_label == "Acontecer"
    assert signer_id == "Sinalizador01"
    np.testing.assert_allclose(loaded_sequence, sequence, rtol=1e-5)
