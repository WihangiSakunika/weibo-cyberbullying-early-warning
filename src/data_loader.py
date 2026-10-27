"""Loaders for the CHNCI and SWSR datasets.

CHNCI  : 95 per-incident CSVs (gb18030) under two split folders
         `cyberbullying/` (47 incidents) and `non-cyberbullying/` (48 incidents).
         Columns: timestamp, platform, label1, label2, label3, diff, content.
         The incident-level cyberbullying label comes from the folder name; the
         comment-level hostile label is the majority vote of the three
         annotation strategies.
SWSR   : SexComment.csv (8,969 comments, label 1=sexist/0=not) - MIT licensed,
         committed at data/raw/swsr/.
"""
import glob
import os

import numpy as np
import pandas as pd

from .config import load_config, p

CHNCI_COLS = ["timestamp", "platform", "label1", "label2", "label3", "diff", "content"]


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    """Read a CSV trying utf-8 first, then gb18030 (CHNCI files are GB18030)."""
    for enc in ("utf-8", "gb18030", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode {path} with utf-8/gb18030")


def load_swsr_comments() -> pd.DataFrame:
    cfg = load_config()["swsr"]
    df = pd.read_csv(p(cfg["comments_file"]))
    df["ts"] = pd.to_datetime(df["date"], format=cfg["timestamp_format"], errors="coerce")
    df["dataset"] = "swsr"
    return df


def load_chnci(raw_dir: str = None, keep_duplicates: bool = True) -> pd.DataFrame:
    """Load all CHNCI incident CSVs.

    Returns one DataFrame with columns:
        incident, incident_label (1=CB incident / 0=non-CB), platform, ts,
        label1, label2, label3, diff, content, y_majority, n_votes, high_conf
    Rows whose three labels are not all in {0,1} get y_majority = NaN.
    """
    cfg = load_config()
    raw_dir = raw_dir or p(cfg["chnci"]["raw_dir"])
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"CHNCI raw data not found at {raw_dir}. "
            "Run `python scripts/download_data.py --dataset chnci` first."
        )
    fmt = cfg["chnci"]["timestamp_format"]
    thr = cfg["labels"]["majority_threshold"]

    frames = []
    for sub, inc_label in (("cyberbullying", 1), ("non-cyberbullying", 0)):
        for path in sorted(glob.glob(os.path.join(raw_dir, "dataset", sub, "*.csv"))):
            df = _read_csv_any_encoding(path)
            missing = set(CHNCI_COLS) - set(df.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            df["incident"] = os.path.basename(path)[:-4]
            df["incident_label"] = inc_label
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CHNCI CSVs found under {raw_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], format=fmt, errors="coerce")

    lab = df[["label1", "label2", "label3"]].apply(pd.to_numeric, errors="coerce")
    votes = (lab == 1).sum(axis=1)
    valid = (lab.isin([0, 1]).sum(axis=1) == 3)
    df["n_votes"] = votes.where(valid)
    df["y_majority"] = np.where(valid & (votes >= thr), 1,
                       np.where(valid & (votes < thr), 0, np.nan))
    df["high_conf"] = valid & ((votes == 3) | (votes == 0))
    df["dataset"] = "chnci"
    if not keep_duplicates:
        df = df.drop_duplicates(subset=["incident", "content"])
    return df


def clean_text(s: pd.Series) -> pd.Series:
    """Minimal lossless cleaning for Chinese social-media text.

    Keeps emotion tags like [流泪] (they carry signal), removes URLs and
    @mentions, collapses whitespace. Never drops rows.
    """
    s = s.fillna("").astype(str)
    s = s.str.replace(r"https?://\S+", " ", regex=True)
    s = s.str.replace(r"@\S+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s
