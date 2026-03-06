# Data utilities for NYC taxi anomaly detection
# Loads raw csv -> adds optional time features -> builds sliding windows -> chronological split
# Split : Chronological split (no shuffle) for temporal data (30 days, 30-min interval). Default uses 60/20/20.
# Inductive bias : daily/weekly cycles (morning/evening demand) encoded using sin/cos time features (cyclic periodicity)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    # X_*: (N, window, D), y_*: (N,), ts_*: (N,)
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ts_train: np.ndarray
    ts_val: np.ndarray
    ts_test: np.ndarray


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cyclic encoding: 23:30 close to 00:30; weekdays wrap around
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    dow = df["timestamp"].dt.dayofweek.astype(float)

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def load_taxi_csv(path: str) -> pd.DataFrame:
    # Required columns: timestamp, value, label (0/1)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"timestamp", "value", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["value"]).sort_values("timestamp").reset_index(drop=True)
    return df


def make_windows(
    df: pd.DataFrame,
    window: int,
    stride: int,
    use_time_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Builds sliding windows for sequence models and window-level labels
    # X: (N, window, D), y: (N,) where y=1 if any anomaly inside the window, ts: window end timestamp
    df2 = df.copy()
    feat_cols = ["value"]

    if use_time_features:
        df2 = _add_time_features(df2)
        feat_cols += ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]

    values = df2[feat_cols].to_numpy(dtype=np.float32)
    labels = df2["label"].to_numpy(dtype=np.int64)
    tss = df2["timestamp"].to_numpy()

    X_list, y_list, ts_list = [], [], []
    for end in range(window - 1, len(df2), stride):
        start = end - window + 1
        X_list.append(values[start:end + 1])
        y_list.append(1 if labels[start:end + 1].max() > 0 else 0)
        ts_list.append(tss[end])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    ts = np.asarray(ts_list)
    return X, y, ts


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> SplitData:
    # Time-ordered split to prevent leakage from future timestamps
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train, y_train, ts_train = X[:n_train], y[:n_train], ts[:n_train]
    X_val, y_val, ts_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val], ts[n_train:n_train + n_val]
    X_test, y_test, ts_test = X[n_train + n_val:], y[n_train + n_val:], ts[n_train + n_val:]

    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test, ts_train, ts_val, ts_test)


def fit_standardizer_on_train(X_train: np.ndarray) -> StandardScaler:
    # Standardize only the value channel (index 0). Keep sin/cos features unchanged.
    scaler = StandardScaler()
    scaler.fit(X_train[:, :, 0].reshape(-1, 1))
    return scaler


def apply_standardizer(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    # Applies the train-fitted scaler to the value channel only
    X2 = X.copy()
    v = scaler.transform(X2[:, :, 0].reshape(-1, 1)).reshape(X2.shape[0], X2.shape[1])
    X2[:, :, 0] = v
    return X2