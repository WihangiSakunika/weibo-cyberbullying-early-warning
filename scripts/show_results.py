"""Display and sanity-check all experiment results in one view.

Usage:
    python scripts/show_results.py            # text summary in terminal
    python scripts/show_results.py --plots    # also open figures (saves an overview grid)

In Google Colab, run after `run_experiments.py`:
    !python scripts/show_results.py --plots
Figures are saved to results/figures/ and can be displayed inline with
IPython.display (see notebook 04 or the snippet in the README).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, p  # noqa: E402


def show(outdir: str = None) -> None:
    cfg = load_config()
    mdir = outdir or p(cfg["paths"]["metrics_dir"])
    fdir = p(cfg["paths"]["figures_dir"])

    def sec(title):
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

    sec("1. DATASET AUDIT (results/metrics/dataset_audit.json)")
    audit_path = os.path.join(mdir, "dataset_audit.json")
    if os.path.exists(audit_path):
        import json
        a = json.load(open(audit_path, encoding="utf-8"))
        for ds in ("chnci", "swsr"):
            d = a[ds]
            print(f"  {ds.upper()}: {d['n_rows']} rows | "
                  f"unparseable timestamps: {d['unparseable_timestamps']}")
    else:
        print("  (not generated yet - run scripts/run_experiments.py)")

    sec("2. MODEL COMPARISON (test split, incident-level)")
    comp_path = os.path.join(mdir, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        cols = [c for c in ["model", "split", "subset", "n", "accuracy",
                            "precision", "recall", "f1", "macro_f1",
                            "roc_auc", "average_precision"] if c in comp.columns]
        print(comp[cols].to_string(index=False))
    else:
        print("  (not generated yet)")

    sec("3. TEMPORAL SUMMARY (CB vs non-CB incidents)")
    tpath = os.path.join(mdir, "temporal_summary.csv")
    if os.path.exists(tpath):
        print(pd.read_csv(tpath).to_string(index=False))
    else:
        print("  (not generated yet)")

    sec("4. EARLY-WARNING EVALUATION")
    epath = os.path.join(mdir, "early_warning_summary.csv")
    if os.path.exists(epath):
        print(pd.read_csv(epath).to_string(index=False))
        print("\n  NOTE: onsets are a Q90 statistical proxy, not external annotations.")
    else:
        print("  (not generated yet)")

    sec("5. FIGURES AVAILABLE (results/figures/)")
    if os.path.isdir(fdir):
        for f in sorted(os.listdir(fdir)):
            print("  ", f)
    else:
        print("  (no figures yet)")

    sec("6. REPRODUCIBILITY CHECK")
    from src.config import CONFIG_PATH
    import yaml
    cfgd = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    print(f"  seed = {cfgd['seed']} | CUSUM delta = "
          f"{cfgd['detectors']['cusum']['delta']}, h = {cfgd['detectors']['cusum']['h']}")
    print("  re-run everything: python scripts/run_experiments.py")


def show_plots():
    """Display all figures inline (Colab/Jupyter)."""
    cfg = load_config()
    fdir = p(cfg["paths"]["figures_dir"])
    try:
        from IPython.display import Image, display
        for f in sorted(os.listdir(fdir)):
            if f.endswith(".png"):
                print(f"\n### {f}")
                display(Image(filename=os.path.join(fdir, f)))
    except ImportError:
        print("Inline display requires Jupyter/IPython; figures are in", fdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true",
                    help="also display figures inline (Jupyter/Colab)")
    args = ap.parse_args()
    show()
    if args.plots:
        show_plots()
