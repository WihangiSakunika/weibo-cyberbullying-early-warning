"""Download the two research datasets from their official GitHub repositories.

- SWSR: MIT licensed; also already committed at data/raw/swsr/ (this script
  refreshes it from upstream).
- CHNCI: NO dataset license -> NOT committed to this repo. Download locally,
  cite Zhu, Zou & Wu (2025) arXiv:2505.20654, do not redistribute.

Usage:
    python scripts/download_data.py              # both
    python scripts/download_data.py --dataset chnci
    python scripts/download_data.py --dataset swsr
"""
import argparse
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, p  # noqa: E402

CHNCI_URL = "https://github.com/zhuyiYZU/CHNCI.git"
SWSR_URL = "https://github.com/aggiejiang/SWSR.git"


def sha256(path: str, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def clone(url: str, dest: str, depth: int = 1) -> bool:
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, ".git")):
        print(f"[skip] already present: {dest}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[git clone] {url} -> {dest}")
    r = subprocess.run(["git", "clone", "--depth", str(depth), url, dest])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["chnci", "swsr"], default=None)
    args = ap.parse_args()
    cfg = load_config()

    want = args.dataset or "both"
    ok = True
    if want in ("swsr", "both"):
        ok &= clone(SWSR_URL, p("data/raw/_upstream_swsr"))
        print("[verify] swsr files:", {f: sha256(p(f'data/raw/_upstream_swsr/SWSR/{f}'))
              for f in ("SexComment.csv", "SexWeibo.csv") if os.path.exists(p(f'data/raw/_upstream_swsr/SWSR/{f}'))})
    if want in ("chnci", "both"):
        ok &= clone(CHNCI_URL, p(cfg["chnci"]["raw_dir"]))
        n = len([f for f in os.listdir(p(os.path.join(cfg["chnci"]["raw_dir"], "dataset", "cyberbullying")))
                 if f.endswith(".csv")]) if os.path.isdir(p(os.path.join(cfg["chnci"]["raw_dir"], "dataset", "cyberbullying"))) else 0
        print(f"[verify] CHNCI cyberbullying incident CSVs found: {n} (expect 47)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
