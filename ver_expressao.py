import numpy as np
from pipeline.model.dataset import load_all_samples
from pipeline.feature_extraction.expression_features import sequence_features, FEATURE_NAMES

s = load_all_samples(r"C:/Users/jujum/OneDrive/Documents/I4/i4/lexicon")[0]
f = sequence_features(s.sequence)
print(s.label, f.shape)
for n, col in zip(FEATURE_NAMES, f.T):
    print(f"{n:12s} min {col.min():7.3f}  mediana {np.median(col):7.3f}  max {col.max():7.3f}")