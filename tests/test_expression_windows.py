import numpy as np

from pipeline.model.expression_windows import segments_from_probs, sliding_windows, window_starts

CLASSES = ["AFIRMACAO", "NEGACAO", "NEUTRO"]


def _probs(labels, conf=0.9):
    out = np.full((len(labels), 3), (1 - conf) / 2)
    for i, name in enumerate(labels):
        out[i, CLASSES.index(name)] = conf
    return out


def _ts(n, fps=25):
    return [int(i * 1000 / fps) for i in range(n)]


def test_window_starts_and_short_clip():
    assert window_starts(30, 10, 5) == [0, 5, 10, 15, 20]
    assert window_starts(5, 10, 5) == []


def test_sliding_windows_shape():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    w, starts = sliding_windows(x, 10, 5)
    assert w.shape == (3, 10, 2) and starts == [0, 5, 10]
    assert sliding_windows(x[:4], 10, 5)[0].shape == (0, 10, 2)


def test_consecutive_windows_merge_into_one_segment():
    labels = ["NEUTRO", "NEGACAO", "NEGACAO", "NEGACAO", "NEUTRO"]
    starts = [0, 5, 10, 15, 20]
    segs = segments_from_probs(_probs(labels), starts, 10, _ts(40), CLASSES)
    assert len(segs) == 1
    s = segs[0]
    assert s.marker == "NEGACAO" and abs(s.confidence - 0.9) < 1e-9
    assert s.start_ms == _ts(40)[5] and s.end_ms == _ts(40)[24]


def test_one_bad_window_does_not_split_a_marker():
    labels = ["NEGACAO", "NEGACAO", "NEUTRO", "NEGACAO", "NEGACAO"]
    segs = segments_from_probs(_probs(labels), [0, 5, 10, 15, 20], 10, _ts(40), CLASSES)
    assert [s.marker for s in segs] == ["NEGACAO"]


def test_different_markers_stay_separate_and_neutral_never_emitted():
    labels = ["NEGACAO", "NEGACAO", "AFIRMACAO", "AFIRMACAO", "NEUTRO"]
    segs = segments_from_probs(_probs(labels), [0, 5, 10, 15, 20], 10, _ts(40), CLASSES)
    assert [s.marker for s in segs] == ["NEGACAO", "AFIRMACAO"]


def test_low_confidence_and_short_segments_are_dropped():
    assert segments_from_probs(_probs(["NEGACAO"] * 4, conf=0.4), [0, 5, 10, 15], 10, _ts(40), CLASSES) == []
    assert segments_from_probs(_probs(["NEGACAO"]), [0], 5, _ts(40), CLASSES) == []  # 200 ms < 300 ms