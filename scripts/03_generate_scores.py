"""
Script 03 – Generate anomaly scores for all models on train / val / test splits.

All deep models (GRU, LSTM, CNN, Transformer, VAE) now use reconstruction MSE
as their anomaly score. Scores are min-max normalised using the training split
so that all models output values in [0, 1] on comparable scales.

Scores are saved to:
  outputs/scores.csv          (test split, all models)
  outputs/scores_splits.pkl   (all splits, all models)
"""

import pickle
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Paths, TrainCfg
from src.models.rnn import GRUAutoencoder
from src.models.lstm import LSTMAutoencoder
from src.models.cnn import CNNAutoencoder
from src.models.transformer import TransformerAutoencoder
from src.models.vae import WindowVAE
from src.models.ensemble import normalize_minmax, ensemble_mean


def recon_score(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Compute per-window reconstruction MSE for a sequence autoencoder.

    Parameters
    ----------
    model : autoencoder that accepts (B, W, D) and returns (B, W, D)
    X : np.ndarray of shape (N, W, D)
    device : torch.device

    Returns
    -------
    scores : np.ndarray of shape (N,) — raw MSE values (not yet normalised)
    """
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        x_hat = model(xb)
        mse = torch.mean((x_hat - xb) ** 2, dim=(1, 2))   # (N,)
    return mse.detach().cpu().numpy()


def vae_raw(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Return per-window reconstruction MSE from the VAE (flattened input)."""
    model.eval()
    Xf = X.reshape(X.shape[0], -1)
    with torch.no_grad():
        xb = torch.tensor(Xf, dtype=torch.float32, device=device)
        x_hat, _, _ = model(xb)
        mse = torch.mean((x_hat - xb) ** 2, dim=1)        # (N,)
    return mse.detach().cpu().numpy()


def main():
    paths = Paths()
    cfg = TrainCfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]
    input_dim = split.X_train.shape[-1]

    # ------------------------------------------------------------------ #
    # Load traditional / incremental models                               #
    # ------------------------------------------------------------------ #
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = pickle.load(f)
    with open(paths.MODEL_DIR / "iso_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(paths.MODEL_DIR / "knn.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(paths.MODEL_DIR / "incremental.pkl", "rb") as f:
        incr = pickle.load(f)

    # ------------------------------------------------------------------ #
    # Load deep autoencoder models                                        #
    # ------------------------------------------------------------------ #
    gru = GRUAutoencoder(
        input_dim=input_dim, hidden=cfg.RNN_HIDDEN,
        layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT,
    ).to(device)
    gru.load_state_dict(torch.load(paths.MODEL_DIR / "rnn.pt", map_location=device))

    lstm = LSTMAutoencoder(
        input_dim=input_dim, hidden=cfg.LSTM_HIDDEN,
        layers=cfg.LSTM_LAYERS, dropout=cfg.LSTM_DROPOUT,
    ).to(device)
    lstm.load_state_dict(torch.load(paths.MODEL_DIR / "lstm.pt", map_location=device))

    cnn = CNNAutoencoder(
        input_dim=input_dim, num_filters=cfg.CNN_FILTERS,
        kernel_size=cfg.CNN_KERNEL, num_layers=cfg.CNN_LAYERS,
        dropout=cfg.CNN_DROPOUT,
    ).to(device)
    cnn.load_state_dict(torch.load(paths.MODEL_DIR / "cnn.pt", map_location=device))

    tf = TransformerAutoencoder(
        input_dim=input_dim, d_model=cfg.TF_D_MODEL,
        nhead=cfg.TF_NHEAD, num_layers=cfg.TF_LAYERS,
        dim_feedforward=cfg.TF_DIM_FF, dropout=cfg.TF_DROPOUT,
    ).to(device)
    tf.load_state_dict(torch.load(paths.MODEL_DIR / "transformer.pt", map_location=device))

    input_dim_vae = split.X_train.reshape(split.X_train.shape[0], -1).shape[1]
    vae = WindowVAE(input_dim=input_dim_vae, hidden=cfg.VAE_HIDDEN, z_dim=cfg.VAE_Z).to(device)
    vae.load_state_dict(torch.load(paths.MODEL_DIR / "vae.pt", map_location=device))

    # ------------------------------------------------------------------ #
    # Pre-compute training-split raw scores for min-max normalisation     #
    # ------------------------------------------------------------------ #
    gru_tr_raw  = recon_score(gru,  split.X_train, device)
    lstm_tr_raw = recon_score(lstm, split.X_train, device)
    cnn_tr_raw  = recon_score(cnn,  split.X_train, device)
    tf_tr_raw   = recon_score(tf,   split.X_train, device)
    vae_tr_raw  = vae_raw(vae, split.X_train, device)

    # ------------------------------------------------------------------ #
    # Score all splits                                                    #
    # ------------------------------------------------------------------ #
    splits_data = {
        "train": (split.X_train, split.y_train, split.ts_train),
        "val":   (split.X_val,   split.y_val,   split.ts_val),
        "test":  (split.X_test,  split.y_test,  split.ts_test),
    }

    all_scores = {}
    for split_name, (X, y, ts) in splits_data.items():
        lof_s  = lof.score(X)
        iso_s  = iso.score(X)
        knn_s  = knn.score(X)
        incr_s = incr.score(X)

        gru_s  = normalize_minmax(gru_tr_raw,  recon_score(gru,  X, device))
        lstm_s = normalize_minmax(lstm_tr_raw, recon_score(lstm, X, device))
        cnn_s  = normalize_minmax(cnn_tr_raw,  recon_score(cnn,  X, device))
        tf_s   = normalize_minmax(tf_tr_raw,   recon_score(tf,   X, device))
        vae_s  = normalize_minmax(vae_tr_raw,  vae_raw(vae, X, device))

        ens_s = ensemble_mean(lof_s, iso_s, knn_s, incr_s, gru_s, lstm_s, cnn_s, tf_s, vae_s)

        all_scores[split_name] = {
            "y":           y,
            "ts":          ts,
            "LOF":         lof_s,
            "IsoForest":   iso_s,
            "KNN":         knn_s,
            "Incremental": incr_s,
            "GRU":         gru_s,
            "LSTM":        lstm_s,
            "CNN":         cnn_s,
            "Transformer": tf_s,
            "VAE":         vae_s,
            "ENS":         ens_s,
        }

    # ------------------------------------------------------------------ #
    # Save test scores CSV                                                #
    # ------------------------------------------------------------------ #
    test = all_scores["test"]
    model_keys = ["LOF", "IsoForest", "KNN", "Incremental", "GRU", "LSTM", "CNN", "Transformer", "VAE", "ENS"]
    df = pd.DataFrame({"timestamp": test["ts"], "label": test["y"]})
    for k in model_keys:
        df[f"{k}_score"] = test[k]

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.OUT_DIR / "scores.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    splits_pkl = {
        split_name: {
            "y":  all_scores[split_name]["y"],
            **{k: all_scores[split_name][k] for k in model_keys},
        }
        for split_name in ["train", "val", "test"]
    }
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump(splits_pkl, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")


if __name__ == "__main__":
    main()
