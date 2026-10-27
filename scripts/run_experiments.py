"""Run the complete experiment pipeline end-to-end:
audit -> preprocess -> train -> temporal analysis -> early-warning evaluation.
Figures for the alarm timeline are regenerated here.

Usage:  python scripts/run_experiments.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, p, ensure_dir          # noqa: E402
from src.data_audit import run_audit                        # noqa: E402
from src.train import train_all                             # noqa: E402
from src.temporal_analysis import run as temporal_run       # noqa: E402
from src.early_warning_evaluation import run as ew_run      # noqa: E402


def main():
    cfg = load_config()
    print("STEP 1/5 dataset audit");            run_audit()
    print("\nSTEP 2/5 model training");         train_all()
    print("\nSTEP 3/5 temporal analysis");      series = temporal_run()

    print("\nSTEP 4/5 early-warning evaluation"); ew_run(series)

    print("\nSTEP 5/5 alarm timeline figure")
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.early_warning_evaluation import add_smoothed_rate
    from src.cusum_detector import CusumDetector
    from src.threshold_detector import RollingThresholdDetector
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]

    figs = ensure_dir(p(cfg["paths"]["figures_dir"]))
    alarms = pd.read_csv(p(cfg["paths"]["metrics_dir"]) + "/alarm_events.csv",
                         parse_dates=["alarm", "onset", "peak"])
    g = add_smoothed_rate(series)
    examples = ["石家庄民宿", "国足0：7日本", "A股疯牛", "青岛地铁"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, inc in zip(axes.ravel(), examples):
        gg = g[g.incident == inc]
        if gg.empty:
            continue
        ax.plot(gg["hour"], gg["r"], color="#cb181d", lw=1.4,
                label="hostility rate (6h MA)")
        a = alarms[alarms.incident == inc]
        ax.vlines(a[a.method == "cusum_delta0.5_h4"]["alarm"], 0, 1,
                  color="#6a3d9a", lw=1, label="CUSUM alarms")
        ax.vlines(a[a.method == "rolling_mean_2sd"]["alarm"], 0, 1,
                  color="#ff7f00", lw=1, linestyle="--", label="rolling-2σ alarms")
        if len(a):
            ax.axvline(a["onset"].iloc[0], color="gray", ls=":", lw=1,
                       label="surge onset (Q90 proxy)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{inc} [{'CB' if gg['incident_label'].iloc[0]==1 else 'non-CB'}]",
                     fontsize=11)
        ax.legend(fontsize=7)
    fig.suptitle("Early-warning alarms on example incidents")
    fig.tight_layout(); fig.savefig(f"{figs}/alarm_timelines.png", dpi=140)
    plt.close(fig)
    print("all steps complete -> results/metrics, results/figures, data/processed")


if __name__ == "__main__":
    main()
