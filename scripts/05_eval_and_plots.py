import json
import pandas as pd

from src.config import Paths
from src.metrics import auroc, auprc
from src.plotting import plot_score_distributions, plot_roc_pr

def main():
    paths = Paths()
    paths.FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(paths.OUT_DIR / "scores.csv")
    y = df["label"].to_numpy()

    score_dict = {
        "LOF": df["LOF_score"].to_numpy(),
        "RNN": df["RNN_score"].to_numpy(),
        "VAE": df["VAE_score"].to_numpy(),
        "ENS": df["ENS_score"].to_numpy(),
    }

    # plots
    plot_score_distributions(paths.FIG_DIR / "fig1_score_dists.png", y, score_dict)
    plot_roc_pr(paths.FIG_DIR / "fig2_roc.png", paths.FIG_DIR / "fig2_pr.png", y, score_dict)

    # table
    rows = []
    for name, s in score_dict.items():
        rows.append({
            "model": name,
            "AUROC": auroc(y, s),
            "AUPRC": auprc(y, s),
        })
    table = pd.DataFrame(rows)
    out_table = paths.TAB_DIR / "table1.csv"
    table.to_csv(out_table, index=False)
    print("Saved:", out_table)
    print("Saved figures in:", paths.FIG_DIR)

if __name__ == "__main__":
    main()