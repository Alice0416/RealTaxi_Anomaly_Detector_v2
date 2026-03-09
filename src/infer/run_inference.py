import json
import numpy as np
import pandas as pd
import torch

from src.models.lof import LOFDetector
from src.models.rnn import GRUAutoencoder
from src.models.vae import WindowVAE
from src.models.ensemble import normalize_minmax, ensemble_mean


def recon_scores(model, X, device):
    """Reconstruction MSE for any sequence autoencoder (GRU/LSTM/CNN/Transformer)."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        x_hat = model(xb)
        mse = torch.mean((x_hat - xb) ** 2, dim=(1, 2)).detach().cpu().numpy()
    return mse


def vae_scores(model, X, device):
    model.eval()
    Xf = X.reshape(X.shape[0], -1)
    with torch.no_grad():
        xb = torch.tensor(Xf, dtype=torch.float32, device=device)
        x_hat, _, _ = model(xb)
        recon = torch.mean((x_hat - xb) ** 2, dim=1).detach().cpu().numpy()
    return recon  # not yet [0,1]


def run_all_scores(split, cfg, paths):
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.TAB_DIR.mkdir(parents=True, exist_ok=True)
    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # load trained models
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # LOF
    lof = LOFDetector()
    # For simplicity: refit on train windows each time in scripts
    return device
