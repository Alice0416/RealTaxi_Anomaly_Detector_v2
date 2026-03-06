import numpy as np
import matplotlib.pyplot as plt
from src.metrics import roc_points, pr_points

def plot_score_distributions(out_path, y_true, score_dict):
    # score_dict: name -> scores
    plt.figure(figsize=(10, 6))
    for name, s in score_dict.items():
        # show anomaly scores only (as overlay) + normal
        s0 = s[y_true == 0]
        s1 = s[y_true == 1]
        plt.hist(s0, bins=60, alpha=0.25, density=True, label=f"{name} normal")
        plt.hist(s1, bins=60, alpha=0.25, density=True, label=f"{name} anomaly")
    plt.xlabel("Normalized score [0,1]")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_roc_pr(out_path_roc, out_path_pr, y_true, score_dict):
    # ROC
    plt.figure(figsize=(7, 6))
    for name, s in score_dict.items():
        fpr, tpr, _ = roc_points(y_true, s)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_roc, dpi=200)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))
    for name, s in score_dict.items():
        prec, rec, _ = pr_points(y_true, s)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_pr, dpi=200)
    plt.close()