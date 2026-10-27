"""Evaluation helpers: classification reports, confusion-matrix plots,
model comparison tables."""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)

from .config import ensure_dir

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def classification_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(set(y_true)) > 1 else None,
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 4)
        if len(set(y_true)) > 1 else None,
    }


def save_classification_report(y_true, y_pred_or_prob, path: str, threshold=0.5):
    y_pred = (np.asarray(y_pred_or_prob) >= threshold).astype(int) \
        if np.asarray(y_pred_or_prob).dtype != int else np.asarray(y_pred_or_prob)
    with open(path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=4, zero_division=0))


def plot_confusion(y_true, y_prob, path: str, threshold=0.5, title=""):
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title or "Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def model_comparison_table(rows: list, out_csv: str, out_json: str = None):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return df
