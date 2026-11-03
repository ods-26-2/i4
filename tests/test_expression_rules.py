import tempfile
from pathlib import Path

import numpy as np

from pipeline.feature_extraction.expression_features import FACE_OFFSET, base_features, compute_baseline
from pipeline.feature_extraction.landmark_extractor import FACE_SEMANTIC_LANDMARK_INDICES, LandmarkExtractionConfig
from pipeline.model.expression_rules import (
    CLASS_NAMES,
    RuleConfig,
    RuleExpressionPredictor,
    calibrate,
    oscillation,
    scores_to_probs,
)

FPS = 25
TS = lambda n: [int(i * 1000 / FPS) for i in range(n)]
_F = list(FACE_SEMANTIC_LANDMARK_INDICES)


def _mesh():
    rng = np.random.default_rng(0)
    mesh = rng.uniform(0.1, 0.9, size=(478, 3)).astype(np.float32)
    # rosto em repouso bem definido: sobrancelha 0.05 acima do olho, pontas/arcos separados
    mesh[70], mesh[300] = mesh[159] + (0, -0.05, 0), mesh[386] + (0, -0.05, 0)
    mesh[55], mesh[285] = (0.45, 0.3, 0.0), (0.55, 0.3, 0.0)
    mesh[107], mesh[336] = (0.47, 0.3, 0.0), (0.53, 0.3, 0.0)
    return mesh


def _face(brow_dy=0.0, furrow_dx=0.0):
    m = _mesh()
    m[[70, 300], 1] -= brow_dy                      # >0 = sobrancelha sobe
    m[[55, 107], 0] += furrow_dx                    # >0 = pontas se aproximam
    m[[285, 336], 0] -= furrow_dx
    return m[_F]


def _pose(nose_x=0.01, nose_y=0.1):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[0] = (nose_x, nose_y, 0.0, 1.0)
    pose[2] = (0.05, -0.05, 0.0, 1.0)
    pose[5] = (-0.05, -0.05, 0.0, 1.0)
    pose[7] = (0.15, 0.0, 0.0, 1.0)
    pose[8] = (-0.15, 0.0, 0.0, 1.0)
    return pose


def _seq(n, nose_x=lambda t: 0.01, nose_y=lambda t: 0.1, face=lambda t: _face()):
    rows = []
    for t in range(n):
        row = np.zeros(LandmarkExtractionConfig().feature_dim, dtype=np.float32)
        p = _pose(nose_x(t), nose_y(t))
        row[: p.size] = p.reshape(-1)
        f = face(t)
        row[FACE_OFFSET : FACE_OFFSET + f.size] = f.reshape(-1)
        rows.append(row)
    return np.stack(rows)


def _baseline():
    return compute_baseline([base_features(_seq(50))])


def _markers(seq):
    segs = RuleExpressionPredictor().predict_segments(seq, TS(len(seq)), baseline=_baseline())
    return [s.marker for s in segs]


def test_neutral_face_and_still_head_gives_nothing():
    assert _markers(_seq(60)) == []


def test_head_shake_is_negacao():
    seq = _seq(60, nose_x=lambda t: 0.08 * np.sin(2 * np.pi * t / 10))
    assert _markers(seq) == ["NEGACAO"]


def test_head_nod_is_afirmacao():
    seq = _seq(60, nose_y=lambda t: 0.1 + 0.08 * np.sin(2 * np.pi * t / 10))
    assert _markers(seq) == ["AFIRMACAO"]


def test_raised_brows_are_interrogativa_sn():
    assert _markers(_seq(60, face=lambda t: _face(brow_dy=0.03))) == ["INTERROGATIVA_SN"]


def test_furrowed_lowered_brows_are_interrogativa_qu():
    seq = _seq(60, face=lambda t: _face(brow_dy=-0.02, furrow_dx=0.02))
    assert _markers(seq) == ["INTERROGATIVA_QU"]


def test_marker_has_sensible_time_range_and_confidence():
    seq = _seq(60, face=lambda t: _face(brow_dy=0.03))
    (s,) = RuleExpressionPredictor().predict_segments(seq, TS(60), baseline=_baseline())
    assert s.start_ms == 0 and s.end_ms >= 1500
    assert 0.5 <= s.confidence <= 1.0


def test_short_sequence_is_evaluated_as_one_window():
    seq = _seq(14, face=lambda t: _face(brow_dy=0.05))
    segs = RuleExpressionPredictor().predict_segments(seq, TS(14), baseline=_baseline(), min_duration_ms=100)
    assert [s.marker for s in segs] == ["INTERROGATIVA_SN"]
    assert RuleExpressionPredictor().predict_segments(_seq(5), TS(5)) == []


def test_oscillation_counts_back_and_forth_and_ignores_flat_or_tiny_noise():
    t = np.arange(20)
    amp, rev = oscillation(0.1 * np.sin(2 * np.pi * t / 10))
    assert amp > 0.1 and rev >= 2
    assert oscillation(np.zeros(20)) == (0.0, 0)
    noise = 1e-4 * np.random.default_rng(0).standard_normal(20)
    assert oscillation(noise)[0] < 1e-3  # tremor minúsculo fica bem abaixo de qualquer limiar
    assert oscillation(np.linspace(0, 1, 20))[1] == 0  # subida contínua não tem volta


def test_scores_to_probs_sum_to_one_and_keep_winner_score():
    probs = scores_to_probs(np.array([[0.1, 0.0, 0.8, 0.2], [0.0, 0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert CLASS_NAMES[probs[0].argmax()] == "INTERROGATIVA_SN" and np.isclose(probs[0].max(), 0.8)
    assert CLASS_NAMES[probs[1].argmax()] == "NEUTRO"


def test_calibrate_suggests_thresholds_and_zero_false_alarms_after():
    rng = np.random.default_rng(1)
    seqs = [
        _seq(80, nose_x=lambda t, a=rng.uniform(0.0, 0.02): 0.01 + a * np.sin(t / 3),
             face=lambda t, b=rng.uniform(0.0, 0.004): _face(brow_dy=b * np.sin(t / 5)))
        for _ in range(6)
    ]
    suggested, rate, total = calibrate(seqs)
    assert total > 0
    again, rate_after, _ = calibrate(seqs, suggested)
    assert all(v == 0.0 for k, v in rate_after.items() if k != "NEUTRO")


def test_rule_config_roundtrip():
    cfg = RuleConfig(shake_amp=0.123)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rules.json"
        cfg.save(p)
        assert RuleConfig.load(p).shake_amp == 0.123