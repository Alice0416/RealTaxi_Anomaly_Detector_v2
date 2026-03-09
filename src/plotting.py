import numpy as np
import matplotlib.pyplot as plt
from src.metrics import roc_points, pr_points


def plot_training_curves(out_path: str, histories: dict) -> None:
    """
    Plot train/val loss and val AUROC curves for all deep models.

    Each model gets two vertically stacked panels:
      top   — train loss + val loss
      bottom — val AUROC (the checkpoint criterion)

    Parameters
    ----------
    out_path : str
        File path to save the figure.
    histories : dict
        {model_name: {"train_loss": [...], "val_loss": [...], "val_auroc": [...]}}
    """
    n = len(histories)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    # Two sub-rows per model: loss on top, auroc on bottom
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows * 2, ncols)

    for col, (name, h) in enumerate(histories.items()):
        row = (col // ncols) * 2
        ax_loss = axes[row, col % ncols]
        ax_auc  = axes[row + 1, col % ncols]

        epochs = range(1, len(h["train_loss"]) + 1)

        # --- Loss panel ---
        ax_loss.plot(epochs, h["train_loss"], label="train loss")
        ax_loss.plot(epochs, h["val_loss"],   label="val loss", alpha=0.8)
        ax_loss.set_title(name, fontsize=10, fontweight="bold")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7)

        # --- AUROC panel ---
        has_auroc = "val_auroc" in h and len(h["val_auroc"]) > 0
        if has_auroc:
            best_ep = int(np.argmax(h["val_auroc"])) + 1
            best_auc = max(h["val_auroc"])
            ax_auc.plot(epochs, h["val_auroc"], color="tab:green", label="val AUROC")
            ax_auc.axvline(best_ep, color="red", linestyle="--", linewidth=0.8,
                           label=f"best ep={best_ep} ({best_auc:.3f})")
            ax_auc.set_ylim(0, 1)
            ax_auc.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
            ax_auc.legend(fontsize=7)
        else:
            ax_auc.text(0.5, 0.5, "no AUROC recorded", ha="center", va="center",
                        transform=ax_auc.transAxes, color="grey")
        ax_auc.set_xlabel("Epoch")
        ax_auc.set_ylabel("Val AUROC")

    # Hide unused subplot pairs
    total_models = len(histories)
    for idx in range(total_models, nrows * ncols):
        r = (idx // ncols) * 2
        c = idx % ncols
        axes[r, c].set_visible(False)
        axes[r + 1, c].set_visible(False)

    fig.suptitle("Training curves — loss & val AUROC per model", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves: {out_path}")

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