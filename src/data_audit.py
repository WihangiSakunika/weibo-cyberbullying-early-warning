"""Dataset audit: dimensions, schema, missing values, duplicates, label quality,
timestamp quality, per-incident coverage. Writes a machine-readable JSON report
and a human-readable text summary to results/metrics/."""
import json

import numpy as np
import pandas as pd

from .config import load_config, p, ensure_dir
from .data_loader import load_chnci, load_swsr_comments


def audit_chnci(df: pd.DataFrame) -> dict:
    rep = {
        "dataset": "CHNCI",
        "n_rows": int(len(df)),
        "n_incidents": int(df["incident"].nunique()),
        "incidents_by_label": {str(k): int(v) for k, v in
                               df.groupby("incident_label")["incident"].nunique().items()},
        "platform_counts": {str(k): int(v) for k, v in df["platform"].value_counts().items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in
                           ["timestamp", "platform", "content",
                            "label1", "label2", "label3"]},
        "unparseable_timestamps": int(df["ts"].isna().sum()),
        "duplicate_rows_incident_content": int(df.duplicated(subset=["incident", "content"]).sum()),
        "invalid_label_rows": int(df["y_majority"].isna().sum()),
        "time_range": [str(df["ts"].min()), str(df["ts"].max())],
        "y_majority_dist": {str(k): int(v) for k, v in
                            df["y_majority"].value_counts(dropna=False).items()},
        "high_conf_rows": int(df["high_conf"].sum()),
        "cb_rate_by_incident_label": {
            "non_cb": round(float(df.loc[df.incident_label == 0, "y_majority"].mean()), 4),
            "cb": round(float(df.loc[df.incident_label == 1, "y_majority"].mean()), 4),
        },
    }
    per_inc = df.groupby(["incident", "incident_label"]).agg(
        n=("content", "size"),
        span_hours=("ts", lambda s: (s.max() - s.min()).total_seconds() / 3600),
    ).reset_index()
    rep["rows_per_incident_median"] = int(per_inc["n"].median())
    rep["span_hours_median"] = round(float(per_inc["span_hours"].median()), 2)
    return rep, per_inc


def audit_swsr(df: pd.DataFrame) -> dict:
    return {
        "dataset": "SWSR",
        "n_rows": int(len(df)),
        "label_dist": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "category_dist": {str(k): int(v) for k, v in df["category"].value_counts(dropna=False).items()},
        "unparseable_timestamps": int(df["ts"].isna().sum()),
        "time_range": [str(df["ts"].min()), str(df["ts"].max())],
        "n_source_weibos": int(df["weibo_id"].nunique()),
        "comments_per_weibo_median": float(df.groupby("weibo_id").size().median()),
        "duplicate_text_rows": int(df.duplicated(subset=["comment_text"]).sum()),
    }


def run_audit(out_dir: str = None) -> dict:
    cfg = load_config()
    out_dir = ensure_dir(out_dir or p(cfg["paths"]["metrics_dir"]))
    chnci = load_chnci()
    swsr = load_swsr_comments()
    chnci_rep, per_inc = audit_chnci(chnci)
    swsr_rep = audit_swsr(swsr)

    report = {"chnci": chnci_rep, "swsr": swsr_rep,
              "compatibility_notes": [
                  "Different label schemes: CHNCI=cyberbullying (3-strategy majority vote), "
                  "SWSR=sexism (single expert label). Not mergeable without a justified mapping; "
                  "used as separate experiments + cross-domain transfer evaluation.",
                  "SWSR threads have a median of 2 comments -> unsuitable for temporal surge "
                  "analysis; CHNCI (minute-level timestamps, continuous per-incident coverage) "
                  "is the temporal dataset.",
              ]}
    with open(f"{out_dir}/dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    per_inc.to_csv(f"{out_dir}/chnci_per_incident.csv", index=False, encoding="utf-8-sig")

    lines = [
        "=== CHNCI audit ===",
        f"rows={chnci_rep['n_rows']}  incidents={chnci_rep['n_incidents']} "
        f"(CB={chnci_rep['incidents_by_label']['1']}, non-CB={chnci_rep['incidents_by_label']['0']})",
        f"time range: {chnci_rep['time_range'][0]} -> {chnci_rep['time_range'][1]}",
        f"unparseable timestamps: {chnci_rep['unparseable_timestamps']}  "
        f"duplicates: {chnci_rep['duplicate_rows_incident_content']}  "
        f"invalid label rows: {chnci_rep['invalid_label_rows']} (reported, not silently dropped)",
        f"hostile rate: CB incidents {chnci_rep['cb_rate_by_incident_label']['cb']} "
        f"vs non-CB {chnci_rep['cb_rate_by_incident_label']['non_cb']}",
        "",
        "=== SWSR audit ===",
        f"rows={swsr_rep['n_rows']}  sexist={swsr_rep['label_dist'].get('1')} "
        f"non-sexist={swsr_rep['label_dist'].get('0')}  "
        f"time range {swsr_rep['time_range'][0]} -> {swsr_rep['time_range'][1]}",
        f"unparseable timestamps: {swsr_rep['unparseable_timestamps']}",
    ]
    with open(f"{out_dir}/dataset_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    run_audit()
