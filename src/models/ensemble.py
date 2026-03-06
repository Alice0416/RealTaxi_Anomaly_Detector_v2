import numpy as np

def normalize_minmax(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    lo = float(train_scores.min())
    hi = float(train_scores.max())
    if hi <= lo + 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

def ensemble_mean(*scores: np.ndarray) -> np.ndarray:
    S = np.stack(scores, axis=0)
    return S.mean(axis=0)

def ensemble_max(*scores: np.ndarray) -> np.ndarray:
    S = np.stack(scores, axis=0)
    return S.max(axis=0)