"""Train comment-level hostility classifiers on CHNCI.

RUN:      TF-IDF + Logistic Regression, TF-IDF + Linear SVM (CPU, seeded)
PLANNED:  Chinese BERT/transformer baseline (src/train_transformer.py placeholder,
          requires GPU + transformers; NOT run in this repository)

Labels: majority vote of CHNCI's three annotation strategies (see data_loader).
Splitting: by incident (GroupShuffleSplit 70/15/15) - no incident appears in
more than one split, preventing incident-vocabulary leakage.
Hyperparameters are fixed in configs/experiment_config.yaml and were chosen on
the validation split; the test split is touched exactly once per model.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from .config import load_config, p, ensure_dir
from .preprocessing import preprocess_chnci
from .evaluate import (classification_metrics, model_comparison_table,
                       plot_confusion, save_classification_report)


def _make_pipeline(kind: str, mcfg: dict, seed: int):
    tfidf = TfidfVectorizer(analyzer=mcfg["analyzer"],
                            ngram_range=tuple(mcfg["ngram_range"]),
                            min_df=mcfg["min_df"], max_features=mcfg["max_features"],
                            sublinear_tf=mcfg["sublinear_tf"])
    if kind == "tfidf_lr":
        clf = LogisticRegression(C=mcfg["C"], class_weight=mcfg["class_weight"],
                                 max_iter=2000, solver="liblinear", random_state=seed)
    elif kind == "tfidf_svm":
        base = LinearSVC(C=mcfg["C"], class_weight=mcfg["class_weight"],
                         random_state=seed)
        # wrap for probability outputs (needed for AUC + threshold-based scoring)
        clf = CalibratedClassifierCV(base, cv=3)
    else:
        raise ValueError(kind)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def _select_threshold(pipe, X_val, y_val):
    """Tune the decision threshold on VALIDATION data only (never the test set)."""
    probs = pipe.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.2, 0.81, 0.05):
        f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, round(best_f1, 4)


def train_all(out_dir: str = None) -> pd.DataFrame:
    cfg = load_config()
    seed = cfg["seed"]
    metrics_dir = ensure_dir(out_dir or p(cfg["paths"]["metrics_dir"]))
    models_dir = ensure_dir(p(cfg["paths"]["models_dir"]))
    figs_dir = ensure_dir(p(cfg["paths"]["figures_dir"]))

    d = preprocess_chnci()
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))

    rows = []
    for kind, label in [("tfidf_lr", "TF-IDF + LogisticRegression"),
                        ("tfidf_svm", "TF-IDF + LinearSVM (calibrated)")]:
        pipe = _make_pipeline(kind, cfg["models"][kind], seed)
        pipe.fit(tr["text"], tr["y_majority"])
        thr, val_f1 = _select_threshold(pipe, va["text"], va["y_majority"])

        m_test = classification_metrics(te["y_majority"],
                                        pipe.predict_proba(te["text"])[:, 1], thr)
        m_test.update({"model": label, "threshold": thr, "val_f1_at_threshold": val_f1,
                       "seed": seed, "split": "incident-level 70/15/15"})
        rows.append(m_test)

        # high-confidence (unanimous-label) test subset
        te_hc = te[te["high_conf"]]
        m_hc = classification_metrics(te_hc["y_majority"],
                                      pipe.predict_proba(te_hc["text"])[:, 1], thr)
        rows.append({**m_hc, "model": f"{label}", "threshold": thr,
                     "subset": "test_unanimous", "seed": seed})

        save_classification_report(te["y_majority"],
                                   pipe.predict_proba(te["text"])[:, 1],
                                   f"{metrics_dir}/classification_report_{kind}.txt",
                                   thr)
        plot_confusion(te["y_majority"], pipe.predict_proba(te["text"])[:, 1],
                       f"{figs_dir}/confusion_{kind}.png", thr, label)
        joblib.dump({"pipeline": pipe, "threshold": thr},
                    f"{models_dir}/{kind}.joblib")

    # zero-shot cross-domain check on SWSR (sexist label as hostility proxy)
    from .data_loader import load_swsr_comments, clean_text
    swsr = load_swsr_comments().dropna(subset=["label", "comment_text"])
    swsr["text"] = clean_text(swsr["comment_text"])
    for kind, label in [("tfidf_lr", "TF-IDF + LogisticRegression"),
                        ("tfidf_svm", "TF-IDF + LinearSVM (calibrated)")]:
        bundle = joblib.load(f"{models_dir}/{kind}.joblib")
        m = classification_metrics(swsr["label"],
                                   bundle["pipeline"].predict_proba(swsr["text"])[:, 1],
                                   bundle["threshold"])
        rows.append({**m, "model": label, "split": "swsr_zero_shot",
                     "threshold": bundle["threshold"], "seed": seed})

    table = model_comparison_table(rows, f"{metrics_dir}/model_comparison.csv",
                                   f"{metrics_dir}/model_comparison.json")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    train_all()
