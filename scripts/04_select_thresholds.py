# Threshold selection, τ
# For each model, choose τ* = argmax_τ F1_val(τ)
# Then apply τ* to test scores: pred = 1 if score > τ*, else 0
# This produces binary prediction columns: LOF_pred, RNN_pred, VAE_pred, ENS_pred

import json
import pickle
import pandas as pd
from src.config import Paths
from src.metrics import best_f1_threshold, apply_threshold

def main():
    paths = Paths()

    with open(paths.OUT_DIR / "scores_splits.pkl", "rb") as f:
        obj = pickle.load(f)
    yv = obj["val"]["y"]

    taus = {}
    stats = {}
    for name in ["LOF", "RNN", "VAE", "ENS"]:
        tau, f1, p, r = best_f1_threshold(yv, obj["val"][name])
        taus[name] = tau
        stats[name] = {"tau": tau, "f1_val": f1, "prec_val": p, "rec_val": r}

    out_json = paths.OUT_DIR / "taus.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved:", out_json)

    # apply thresholds to test scores -> preds.csv
    scores_df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    for name in ["LOF", "RNN", "VAE", "ENS"]:
        scores_df[f"{name}_pred"] = apply_threshold(scores_df[f"{name}_score"].to_numpy(), taus[name])

    out_csv = paths.OUT_DIR / "preds.csv"
    scores_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()