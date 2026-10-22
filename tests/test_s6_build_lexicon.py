import numpy as np

from pipeline.feature_extraction.landmark_extractor import load_landmark_sequence, save_landmark_sequence
from s6_synthesis.lexicon.build_lexicon import build_lexicon, pick_canonical_sample


def test_pick_canonical_sample_chooses_median_length(tmp_path):
    lengths = [5, 10, 15, 20, 100]  # mediana = 15; 100 é outlier
    paths = []
    for i, length in enumerate(lengths):
        seq = np.random.rand(length, 4).astype(np.float32)
        path = save_landmark_sequence(f"v{i}", "Aluno", "S1", seq, tmp_path)
        paths.append(path)

    chosen = pick_canonical_sample(paths)
    sequence, *_ = load_landmark_sequence(chosen)
    assert len(sequence) == 15


def test_build_lexicon_one_file_per_class(tmp_path):
    features_dir = tmp_path / "features"
    out_dir = tmp_path / "lexicon"
    features_dir.mkdir()

    for cls in ["Aluno", "Banco"]:
        for rep in range(3):
            seq = np.random.rand(10 + rep, 6).astype(np.float32)
            save_landmark_sequence(f"{cls}{rep}", cls, f"S{rep}", seq, features_dir)

    saved = build_lexicon(features_dir, out_dir)
    assert set(saved.keys()) == {"Aluno", "Banco"}
    for cls, path in saved.items():
        sequence, _, class_label, signer_id = load_landmark_sequence(path)
        assert class_label == cls
        assert signer_id.startswith("lexicon(")
        assert sequence.shape[1] == 6
