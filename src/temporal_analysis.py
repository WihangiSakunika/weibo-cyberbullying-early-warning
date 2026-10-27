"""Temporal analysis: aggregate scored comments into hourly windows per incident,
compute hostility counts/rates, fill gaps to a continuous hourly index, and plot
per-incident dynamics. Requires a trained scorer (src/train.py) to score the
full corpus with the frozen model.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from .config import load_config, p, ensure_dir
from .data_loader import load_chnci, clean_text

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def score_corpus(model_dir: str = None) -> pd.DataFrame:
    """Score every CHNCI comment with the frozen tfidf_lr scorer.

    NOTE: the pipeline fitted in train.py is reused here; scoring the full
    corpus (incl. rows outside the train split) mirrors real deployment, where
    the monitor classifies every incoming comment.
    """
    cfg = load_config()
    models_dir = model_dir or p(cfg["paths"]["models_dir"])
    bundle = joblib.load(f"{models_dir}/tfidf_lr.joblib")
    df = load_chnci()
    df["text"] = clean_text(df["content"])
    df["hostility_score"] = bundle["pipeline"].predict_proba(df["text"])[:, 1]
    return df


def hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate (same incident/content/timestamp counts once), aggregate to
    hourly volume + predicted-hostile counts + rate, reindex to a continuous
    hourly range per incident (missing hours: volume 0, rate NaN)."""
    cfg = load_config()["temporal"]
    d = df.drop_duplicates(subset=["incident", "content", "ts"]).copy()
    d["hour"] = d["ts"].dt.floor(cfg["window"])
    agg = d.groupby(["incident", "incident_label", "hour"]).agg(
        volume=("content", "size"),
        hostile_pred=("hostility_score", lambda s: (s >= 0.5).sum()),
    ).reset_index()
    frames = []
    for inc, g in agg.groupby("incident"):
        full = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h")
        g = g.set_index("hour").reindex(full)
        g["incident"], g["incident_label"] = inc, g["incident_label"].iloc[0]
        g[["volume", "hostile_pred"]] = g[["volume", "hostile_pred"]].fillna(0).astype(int)
        g["hostile_rate"] = np.where(g["volume"] > 0,
                                     g["hostile_pred"] / g["volume"], np.nan)
        frames.append(g.rename_axis("hour").reset_index())
    return pd.concat(frames, ignore_index=True)


def compare_incident_types(series: pd.DataFrame) -> pd.DataFrame:
    span = series.groupby(["incident", "incident_label"]).agg(
        hours=("hour", "size"),
        active_hours=("volume", lambda v: int((v > 0).sum())),
        total=("volume", "sum"),
        peak_rate=("hostile_rate", "max"),
        mean_rate=("hostile_rate", "mean"),
    ).reset_index()
    return span


def plot_examples(series: pd.DataFrame, fig_path: str,
                  incidents=("国足0：7日本", "石家庄民宿", "A股疯牛", "iphone16成本拆解")):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, inc in zip(axes.ravel(), incidents):
        g = series[series.incident == inc]
        if g.empty:
            continue
        ax2 = ax.twinx()
        ax.bar(g["hour"], np.log1p(g["volume"]), width=0.04, color="#9ecae1", alpha=.7)
        ax2.plot(g["hour"], g["hostile_rate"].rolling(6, min_periods=2).mean(),
                 color="#cb181d", lw=1.5)
        lab = "CB" if g["incident_label"].iloc[0] == 1 else "non-CB"
        ax.set_title(f"{inc} [{lab}]", fontsize=11)
        ax.set_ylabel("log1p volume"); ax2.set_ylabel("hostility rate (6h MA)", color="#cb181d")
    fig.suptitle("Hourly comment volume (bars) vs predicted hostility rate (red)")
    fig.tight_layout(); fig.savefig(fig_path, dpi=140); plt.close(fig)


def run(models_dir: str = None, out_dir: str = None) -> pd.DataFrame:
    cfg = load_config()
    figs = ensure_dir(p(cfg["paths"]["figures_dir"]))
    out_dir = ensure_dir(out_dir or p(cfg["paths"]["processed_dir"]))
    df = score_corpus(models_dir)
    series = hourly_series(df)
    series.to_parquet(f"{out_dir}/hourly_series.parquet", index=False)
    span = compare_incident_types(series)
    summary = span.groupby("incident_label")[["peak_rate", "mean_rate",
                                              "active_hours", "total"]].mean().round(4)
    summary.to_csv(p(cfg["paths"]["metrics_dir"]) + "/temporal_summary.csv",
                   encoding="utf-8-sig")
    plot_examples(series, f"{figs}/temporal_examples.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, col in [(1, "#cb181d"), (0, "#2171b5")]:
        vals = span[span.incident_label == lab]["peak_rate"].dropna()
        ax.hist(vals, bins=20, alpha=.6, color=col,
                label=f"{'CB' if lab == 1 else 'non-CB'} incidents (n={len(vals)})")
    ax.set_xlabel("peak hourly hostility rate per incident"); ax.set_ylabel("# incidents")
    ax.legend(); ax.set_title("Peak hostility rate: CB vs non-CB incidents")
    fig.tight_layout(); fig.savefig(f"{figs}/peak_rate_hist.png", dpi=140); plt.close(fig)
    print(summary.to_string())
    return series
