import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

def auroc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))

def auprc(y_true, y_score):
    return float(average_precision_score(y_true, y_score))

def roc_points(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr

def pr_points(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    return prec, rec, thr

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray):
    # sweep thresholds from PR curve (more stable)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # thr has length = len(prec)-1
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i]), float(prec[i]), float(rec[i])

def apply_threshold(y_score: np.ndarray, tau: float):
    return (y_score >= tau).astype(int)