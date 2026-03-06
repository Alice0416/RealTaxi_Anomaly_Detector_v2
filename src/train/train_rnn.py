import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    opt = torch.optim.Adam(model.parameters(), lr=cfg.RNN_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = 1e9
    for epoch in range(cfg.RNN_EPOCHS):
        model.train()
        pbar = tqdm(tr_loader, desc=f"RNN epoch {epoch+1}/{cfg.RNN_EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        val_loss /= max(n, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)

    return out_path