import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

class LOFDetector:
    def __init__(self, n_neighbors: int = 35, contamination: float = 0.1):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,  # allows score on new data
        )
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        # X: (N, W, D) -> (N, W*D)
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray):
        Z = self._flatten_windows(X_train)
        self.model.fit(Z)
        # fit score normalization on train scores
        raw = -self.model.score_samples(Z)  # higher = more anomalous
        self.scaler.fit(raw.reshape(-1, 1))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self._flatten_windows(X)
        raw = -self.model.score_samples(Z)
        s = self.scaler.transform(raw.reshape(-1, 1)).reshape(-1)
        return s  # [0,1]