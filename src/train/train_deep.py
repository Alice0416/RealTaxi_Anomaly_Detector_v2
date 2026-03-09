"""
Generic training loop for deep temporal anomaly classifiers.

Supports: LSTM, CNN, Transformer (and any nn.Module that accepts
(B, W, D) input and returns (B,) logits for binary classification).

Training details
----------------
- Loss: Binary cross-entropy with logits (BCEWithLogitsLoss)
- Optimizer: Adam
- Checkpointing: best val AUROC (not val loss — val loss is a broken signal
  under high pos_weight because each missed anomaly incurs a ~49x penalty,
  causing val loss to diverge monotonically from epoch 1 even as the model
  improves its ranking ability)
- Class imbalance: pos_weight capped at 10 to avoid catastrophic val loss
  inflation while still correcting for the ~2% anomaly base rate
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def _pos_weight(y_train: np.ndarray, device: torch.device) -> torch.Tensor:
    """Compute positive class weight, capped at 10 to prevent val loss explosion."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        return torch.tensor(1.0, device=device)
    raw = float(n_neg) / float(n_pos)
    return torch.tensor(min(raw, 10.0), device=device)


def train_deep_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    out_path: str,
    device: torch.device,
) -> str:
    """
    Train a binary classification model and save the best checkpoint.

    Parameters
    ----------
    model : nn.Module
        Instantiated model (already moved to ``device``).
    X_train, X_val : np.ndarray of shape (N, W, D)
    y_train, y_val : np.ndarray of shape (N,) with values in {0, 1}
    epochs : int
    lr : float
    batch_size : int
    out_path : str
        File path to save the best model state dict.
    device : torch.device

    Returns
    -------
    out_path : str
    """
    pw = _pos_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)
    best_auroc = -1.0

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_tr = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(n_tr, 1)

        # Validation — collect logits for AUROC, loss for monitoring
        model.eval()
        val_loss = 0.0
        n = 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())
        val_loss /= max(n, 1)

        val_probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        val_labels = torch.cat(all_labels).numpy()
        # AUROC requires both classes present; fall back to 0.5 if val has no anomalies
        val_auroc = float(roc_auc_score(val_labels, val_probs)) if val_labels.sum() > 0 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auroc={val_auroc:.4f}")

        # Checkpoint on val AUROC — immune to pos_weight scaling
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), out_path)

        scheduler.step()

    return out_path, history
