import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.vae import WindowVAE

def vae_loss(x, x_hat, mu, logvar, beta: float):
    recon = torch.mean((x_hat - x) ** 2, dim=1)  # per-sample MSE
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return (recon + beta * kl).mean(), recon.detach()

def train_vae(X_train, X_val, cfg, out_path):
    if torch.cuda.is_available() and cfg.DEVICE == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA not available.")

    # VAE uses flattened windows
    Xtr = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
    Xva = torch.tensor(X_val.reshape(X_val.shape[0], -1), dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr), batch_size=cfg.VAE_BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva), batch_size=cfg.VAE_BATCH, shuffle=False)

    model = WindowVAE(input_dim=Xtr.shape[1], hidden=cfg.VAE_HIDDEN, z_dim=cfg.VAE_Z).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.VAE_LR)

    best_val = 1e9
    for epoch in range(cfg.VAE_EPOCHS):
        model.train()
        pbar = tqdm(tr_loader, desc=f"VAE epoch {epoch+1}/{cfg.VAE_EPOCHS}")
        for (xb,) in pbar:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(xb)
            loss, _ = vae_loss(xb, x_hat, mu, logvar, beta=cfg.VAE_BETA)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for (xb,) in va_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = model(xb)
                loss, _ = vae_loss(xb, x_hat, mu, logvar, beta=cfg.VAE_BETA)
                val_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        val_loss /= max(n, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)

    return out_path