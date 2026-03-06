import pickle
import torch
from src.config import Paths, TrainCfg, LofCfg
from src.models.lof import LOFDetector
from src.train.train_rnn import train_rnn
from src.train.train_vae import train_vae

def main():
    paths = Paths()
    cfg = TrainCfg()
    lof_cfg = LofCfg()

    with open(paths.OUT_DIR / "split.pkl", "rb") as f:
        obj = pickle.load(f)
    split = obj["split"]

    paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # LOF fit
    lof = LOFDetector(n_neighbors=lof_cfg.N_NEIGHBORS, contamination=lof_cfg.CONTAMINATION)
    lof.fit(split.X_train)
    with open(paths.MODEL_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
    print("Saved:", paths.MODEL_DIR / "lof.pkl")

    # RNN train
    rnn_path = str(paths.MODEL_DIR / "rnn.pt")
    train_rnn(split.X_train, split.y_train, split.X_val, split.y_val, cfg, rnn_path)
    print("Saved:", rnn_path)

    # VAE train
    vae_path = str(paths.MODEL_DIR / "vae.pt")
    train_vae(split.X_train, split.X_val, cfg, vae_path)
    print("Saved:", vae_path)

if __name__ == "__main__":
    main()