"""
Shared training loop for all sequence autoencoder models.

Philosophy
----------
All deep models (GRU, LSTM, CNN, Transformer) are trained as unsupervised
sequence autoencoders on NORMAL-ONLY windows. This completely sidesteps the
extreme label scarcity problem (~2% positive rate, ~123 positive windows out
of 6163 training windows) that caused inverted AUROC in classifier training.

Training procedure
------------------
- Keep only windows where y_ep == 0 (no anomaly at the endpoint)
- Loss: MSE between input window and reconstructed window
- Optimizer: Adam with weight_decay
- Scheduler: CosineAnnealingLR
- Grad clipping: max_norm=1.0
- Checkpoint: best val loss (lower = better; val includes both normal and
  anomaly windows so a good checkpoint reconstructs normals well and
  produces high error on anomalies)

At inference, anomaly score = per-window mean MSE (higher = more anomalous).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_reconstruction(
    model: nn.Module,
    X_train: np.ndarray,
    y_ep_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    out_path: str,
    device: torch.device,
) -> tuple:
    """
    Train a sequence autoencoder on normal-only windows.

    Parameters
    ----------
    model : nn.Module
        Autoencoder that accepts (B, W, D) and returns (B, W, D).
    X_train : np.ndarray, shape (N, W, D)
    y_ep_train : np.ndarray, shape (N,)  — endpoint labels (0=normal, 1=anomaly)
    X_val : np.ndarray, shape (M, W, D)  — full val set (used for loss monitoring)
    epochs : int
    lr : float
    batch_size : int
    out_path : str
        Where to save the best model state dict.
    device : torch.device

    Returns
    -------
    out_path : str
    history : dict with keys "train_loss", "val_loss"
    """
    # Train only on normal windows — no label imbalance problem
    normal_mask = y_ep_train == 0
    X_normal = X_train[normal_mask]
    n_total = len(X_train)
    n_normal = int(normal_mask.sum())
    print(f"  Autoencoder training on {n_normal}/{n_total} normal windows "
          f"({100 * n_normal / max(n_total, 1):.1f}%)")

    Xtr = torch.tensor(X_normal, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva), batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_tr = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for (xb,) in pbar:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        train_loss /= max(n_tr, 1)

        model.eval()
        val_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(device)
                x_hat = model(xb)
                loss = criterion(x_hat, xb)
                val_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)
        val_loss /= max(n_va, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)

        scheduler.step()

    return out_path, history
