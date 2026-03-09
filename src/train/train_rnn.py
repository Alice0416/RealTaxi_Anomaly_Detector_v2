import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models.rnn import GRUAnomalyClassifier

def train_rnn(X_train, y_train, X_val, y_val, cfg, out_path):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.RNN_BATCH, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.RNN_BATCH, shuffle=False, drop_last=False)

    model = GRUAnomalyClassifier(input_dim=X_train.shape[-1], hidden=cfg.RNN_HIDDEN, layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.RNN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.RNN_EPOCHS, eta_min=cfg.RNN_LR / 20)
    # Upweight the minority class, capped at 10 to prevent val loss explosion
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pw = torch.tensor(min(float(n_neg) / max(n_pos, 1), 10.0), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = -1.0
    for epoch in range(cfg.RNN_EPOCHS):
        model.train()
        train_loss = 0.0
        n_tr = 0
        pbar = tqdm(tr_loader, desc=f"RNN epoch {epoch+1}/{cfg.RNN_EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)
            pbar.set_postfix(loss=float(loss.item()))
        train_loss /= max(n_tr, 1)

        model.eval()
        val_loss = 0.0
        n = 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(yb.cpu())
        val_loss /= max(n, 1)

        val_probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        val_labels = torch.cat(all_labels).numpy()
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