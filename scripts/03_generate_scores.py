import pickle
import numpy as np
import pandas as pd
import torch

from src.config import Paths, TrainCfg
from src.models.rnn import GRUAnomalyClassifier
from src.models.vae import WindowVAE
from src.models.ensemble import normalize_minmax, ensemble_mean

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def main():
    paths = Paths()
    cfg = TrainCfg()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    # load LOF
    with open(paths.MODEL_DIR / "lof.pkl", "rb") as f:
        lof = pickle.load(f)

    # load RNN
    rnn = GRUAnomalyClassifier(input_dim=split.X_train.shape[-1], hidden=cfg.RNN_HIDDEN, layers=cfg.RNN_LAYERS, dropout=cfg.RNN_DROPOUT).to(device)
    rnn.load_state_dict(torch.load(paths.MODEL_DIR / "rnn.pt", map_location=device))

    # load VAE
    input_dim_vae = split.X_train.reshape(split.X_train.shape[0], -1).shape[1]
    vae = WindowVAE(input_dim=input_dim_vae, hidden=cfg.VAE_HIDDEN, z_dim=cfg.VAE_Z).to(device)
    vae.load_state_dict(torch.load(paths.MODEL_DIR / "vae.pt", map_location=device))

    def rnn_score(X):
        rnn.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=device)
            logits = rnn(xb).detach().cpu().numpy()
        return sigmoid(logits)

    def vae_raw(X):
        vae.eval()
        Xf = X.reshape(X.shape[0], -1)
        with torch.no_grad():
            xb = torch.tensor(Xf, dtype=torch.float32, device=device)
            x_hat, _, _ = vae(xb)
            recon = torch.mean((x_hat - xb) ** 2, dim=1).detach().cpu().numpy()
        return recon

    # scores on each split
    lof_tr = lof.score(split.X_train)
    lof_va = lof.score(split.X_val)
    lof_te = lof.score(split.X_test)

    rnn_tr = rnn_score(split.X_train)
    rnn_va = rnn_score(split.X_val)
    rnn_te = rnn_score(split.X_test)

    vae_tr_raw = vae_raw(split.X_train)
    vae_va_raw = vae_raw(split.X_val)
    vae_te_raw = vae_raw(split.X_test)

    # normalize VAE raw -> [0,1] using train min/max
    vae_tr = normalize_minmax(vae_tr_raw, vae_tr_raw)
    vae_va = normalize_minmax(vae_tr_raw, vae_va_raw)
    vae_te = normalize_minmax(vae_tr_raw, vae_te_raw)

    # Ensemble on normalized scores (mean)
    ens_tr = ensemble_mean(lof_tr, rnn_tr, vae_tr)
    ens_va = ensemble_mean(lof_va, rnn_va, vae_va)
    ens_te = ensemble_mean(lof_te, rnn_te, vae_te)

    # build one dataframe for test (for plots/table); you can also save all splits if you want
    df = pd.DataFrame({
        "timestamp": split.ts_test,
        "label": split.y_test,
        "LOF_score": lof_te,
        "RNN_score": rnn_te,
        "VAE_score": vae_te,
        "ENS_score": ens_te,
    })
    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = paths.OUT_DIR / "scores.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # save train/val scores too (needed for threshold selection on val)
    with open(paths.OUT_DIR / "scores_splits.pkl", "wb") as f:
        pickle.dump({
            "train": {"y": split.y_train, "LOF": lof_tr, "RNN": rnn_tr, "VAE": vae_tr, "ENS": ens_tr},
            "val":   {"y": split.y_val,   "LOF": lof_va, "RNN": rnn_va, "VAE": vae_va, "ENS": ens_va},
            "test":  {"y": split.y_test,  "LOF": lof_te, "RNN": rnn_te, "VAE": vae_te, "ENS": ens_te},
        }, f)
    print("Saved:", paths.OUT_DIR / "scores_splits.pkl")

if __name__ == "__main__":
    main()