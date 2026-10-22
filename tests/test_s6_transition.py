import numpy as np

from s6_synthesis.synthesis.transition import generate_transition


def test_transition_has_requested_length_and_dim():
    end = np.zeros(5, dtype=np.float32)
    start = np.ones(5, dtype=np.float32)
    out = generate_transition(end, start, num_frames=4)
    assert out.shape == (4, 5)


def test_transition_moves_monotonically_from_end_towards_start():
    end = np.array([0.0], dtype=np.float32)
    start = np.array([1.0], dtype=np.float32)
    out = generate_transition(end, start, num_frames=6).ravel()
    assert np.all(np.diff(out) > 0)
    assert out[0] > 0.0  # não repete o frame final do sinal anterior
    assert out[-1] < 1.0  # não repete o frame inicial do próximo sinal


def test_transition_eases_in_and_out_slower_than_the_middle():
    # smoothstep: o passo entre frames no meio é maior que nas pontas.
    end = np.array([0.0], dtype=np.float32)
    start = np.array([1.0], dtype=np.float32)
    out = generate_transition(end, start, num_frames=7).ravel()
    steps = np.diff(out)
    assert steps[len(steps) // 2] > steps[0]
    assert steps[len(steps) // 2] > steps[-1]


def test_zero_frames_returns_empty_with_matching_dim():
    end = np.zeros(3, dtype=np.float32)
    start = np.ones(3, dtype=np.float32)
    out = generate_transition(end, start, num_frames=0)
    assert out.shape == (0, 3)
