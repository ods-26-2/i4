from types import SimpleNamespace

import numpy as np

from pipeline.feature_extraction import calculos_ar as ar
from pipeline.feature_extraction.expression_features import (
    BASE_FEATURE_NAMES,
    FACE_OFFSET,
    N_BASE,
    N_FEATURES,
    RollingBaseline,
    add_dynamics,
    base_features,
    compute_baseline,
    face_features,
    head_features,
    sequence_features,
)
from pipeline.feature_extraction.landmark_extractor import (
    FACE_SEMANTIC_LANDMARK_INDICES,
    LandmarkExtractionConfig,
)

# Todos os pontos que o módulo usa precisam estar nos 92 do contrato.
USED_MESH_POINTS = (
    {160, 144, 158, 153, 33, 133, 385, 380, 387, 373, 362, 263}
    | {13, 14, 82, 87, 312, 317, 61, 291, 0, 17}
    | {70, 159, 300, 386, 55, 285, 107, 336}
)


def _mesh(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, size=(478, 3)).astype(np.float32)


def _as_landmarks(mesh):
    return [SimpleNamespace(x=float(r[0]), y=float(r[1]), z=float(r[2])) for r in mesh]


def _pose(nose_x=0.0, nose_y=0.1):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = (nose_x, nose_y, 0.0, 1.0)      # nariz
    pose[2] = (0.05, -0.05, 0.0, 1.0)         # olho esq.
    pose[5] = (-0.05, -0.05, 0.0, 1.0)        # olho dir.
    pose[7] = (0.15, 0.0, 0.0, 1.0)           # orelha esq.
    pose[8] = (-0.15, 0.0, 0.0, 1.0)          # orelha dir.
    return pose


def _row(pose, face):
    config = LandmarkExtractionConfig()
    row = np.zeros(config.feature_dim, dtype=np.float32)
    row[: pose.size] = pose.reshape(-1)
    row[FACE_OFFSET : FACE_OFFSET + face.size] = face.reshape(-1)
    return row


def test_used_points_are_in_reduced_face():
    assert USED_MESH_POINTS <= set(FACE_SEMANTIC_LANDMARK_INDICES)


def test_face_features_match_calculos_ar_for_shared_terms():
    mesh = _mesh()
    lm = _as_landmarks(mesh)
    face92 = mesh[list(FACE_SEMANTIC_LANDMARK_INDICES)]
    got = dict(zip(BASE_FEATURE_NAMES, face_features(face92)))

    assert np.isclose(got["ear_esq"], ar.calcular_ear(ar.PONTOS_OLHO_ESQ, lm, 1, 1), rtol=1e-4)
    assert np.isclose(got["ear_dir"], ar.calcular_ear(ar.PONTOS_OLHO_DIR, lm, 1, 1), rtol=1e-4)
    assert np.isclose(got["mar"], ar.calcular_mar(lm, 1, 1), rtol=1e-4)
    assert np.isclose(got["brow_altura"], ar.calcular_bar(lm, 1, 1)[0], rtol=1e-4)
    assert np.isclose(got["nmar"], ar.calcular_nmar(lm, 1, 1), rtol=1e-4)
    assert np.isclose(got["pup"], ar.calcular_pup(lm, 1, 1), rtol=1e-4)


def test_raised_brows_increase_brow_altura():
    mesh = _mesh(1)
    # sobrancelha (70, 300) acima do olho (159, 386); depois sobe mais (y menor)
    mesh[70] = mesh[159] + (0.0, -0.05, 0.0)
    mesh[300] = mesh[386] + (0.0, -0.05, 0.0)
    base = face_features(mesh[list(FACE_SEMANTIC_LANDMARK_INDICES)])
    raised = mesh.copy()
    raised[[70, 300], 1] -= 0.05
    up = face_features(raised[list(FACE_SEMANTIC_LANDMARK_INDICES)])
    i = BASE_FEATURE_NAMES.index("brow_altura")
    assert up[i] > base[i]


def test_missing_face_is_nan():
    assert np.all(np.isnan(face_features(np.zeros((92, 3), dtype=np.float32))))


def test_head_yaw_is_zero_centered_and_flips_sign():
    yaw0 = head_features(_pose(nose_x=0.0))[0]
    yaw_l = head_features(_pose(nose_x=0.08))[0]
    yaw_r = head_features(_pose(nose_x=-0.08))[0]
    assert abs(yaw0) < 1e-6
    assert yaw_l < 0 < yaw_r or yaw_r < 0 < yaw_l


def test_head_pitch_increases_when_nose_drops():
    up = head_features(_pose(nose_y=0.02))[1]
    down = head_features(_pose(nose_y=0.2))[1]
    assert down > up


def test_sequence_features_shape_and_shake_vs_nod():
    face = _mesh(2)[list(FACE_SEMANTIC_LANDMARK_INDICES)]
    t = np.arange(50)
    shake = np.stack([_row(_pose(nose_x=0.08 * np.sin(t_ / 3)), face) for t_ in t])
    nod = np.stack([_row(_pose(nose_y=0.1 + 0.08 * np.sin(t_ / 3)), face) for t_ in t])

    fs, fn = sequence_features(shake), sequence_features(nod)
    assert fs.shape == fn.shape == (50, N_FEATURES)
    assert not np.isnan(fs).any()
    yaw, pitch = BASE_FEATURE_NAMES.index("yaw"), BASE_FEATURE_NAMES.index("pitch")
    assert fs[:, yaw].std() > 3 * fs[:, pitch].std()   # balanço: yaw varia
    assert fn[:, pitch].std() > 3 * fn[:, yaw].std()   # aceno: pitch varia


def test_missing_frames_are_forward_filled():
    face = _mesh(3)[list(FACE_SEMANTIC_LANDMARK_INDICES)]
    rows = [_row(_pose(), face) for _ in range(5)]
    rows[2] = np.zeros_like(rows[2])  # frame sem detecção
    base = base_features(np.stack(rows))
    assert np.isnan(base[2]).all()
    out = sequence_features(np.stack(rows))
    assert not np.isnan(out).any()


def test_add_dynamics_velocity_scales_with_fps():
    base = np.zeros((3, N_BASE), dtype=np.float32)
    base[:, BASE_FEATURE_NAMES.index("yaw")] = [0.0, 0.1, 0.2]
    out = add_dynamics(base, fps=10.0)
    assert out.shape == (3, N_FEATURES)
    assert np.isclose(out[1, N_BASE + 1], 1.0)  # (0.1 - 0) * 10


def test_compute_baseline_is_median_of_neutral_frames():
    a = np.ones((4, N_BASE), dtype=np.float32)
    b = np.full((4, N_BASE), 3.0, dtype=np.float32)
    assert np.allclose(compute_baseline([a, b]), 2.0)


def test_rolling_baseline_waits_for_min_frames():
    rb = RollingBaseline(seconds=1, fps=10, min_frames=3)
    rb.update(np.ones(N_BASE))
    assert np.allclose(rb.value, 0.0)
    for _ in range(3):
        rb.update(np.full(N_BASE, 2.0))
    assert np.allclose(rb.value, 2.0)