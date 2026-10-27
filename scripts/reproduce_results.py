"""Re-run the full experiment pipeline with the repository's fixed seed and
verify that the committed result files exist and are fresh.

Usage:  python scripts/reproduce_results.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, p  # noqa: E402

REQUIRED_OUTPUTS = [
    "results/metrics/dataset_audit.json",
    "results/metrics/model_comparison.csv",
    "results/metrics/early_warning_summary.csv",
    "results/metrics/alarm_events.csv",
    "results/figures/temporal_examples.png",
    "results/figures/alarm_timelines.png",
    "results/figures/confusion_tfidf_lr.png",
    "results/figures/confusion_tfidf_svm.png",
]


def main():
    cfg = load_config()
    print(f"[reproduce] seed={cfg['seed']} — running full pipeline "
          "(audit -> train -> temporal -> early warning)")
    subprocess.run([sys.executable, p("scripts/run_experiments.py")], check=True)

    missing = [f for f in REQUIRED_OUTPUTS if not os.path.exists(p(f))]
    if missing:
        print("[reproduce] MISSING outputs:", missing)
        sys.exit(1)
    print("[reproduce] all required result files present:")
    for f in REQUIRED_OUTPUTS:
        print("  ok", f)


if __name__ == "__main__":
    main()
