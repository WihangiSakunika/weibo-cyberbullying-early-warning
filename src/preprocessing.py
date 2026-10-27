"""Preprocessing: text cleaning, incident-level (group) train/val/test splits,
and export of the processed corpus used by every downstream stage."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import load_config, p, ensure_dir
from .data_loader import clean_text


def build_splits(df: pd.DataFrame, cfg: dict = None):
    """Incident-level GroupShuffleSplit: 70/15/15 train/val/test.

    Fitting the vectorizer only on train is handled in src/train.py; this
    function only partitions rows so no incident appears in two splits.
    """
    cfg = cfg or load_config()
    seed = cfg["seed"]
    d = df.dropna(subset=["y_majority", "content"]).copy()
    d = d[d["content"].str.strip() != ""]
    d = d.drop_duplicates(subset=["incident", "content"])
    d["y_majority"] = d["y_majority"].astype(int)
    d["text"] = clean_text(d["content"])
    d = d[d["text"] != ""]                     # only rows whose text emptied out

    groups = d["incident"].values
    gss = GroupShuffleSplit(n_splits=1, train_size=cfg["splits"]["train_frac"],
                            random_state=seed)
    train_idx, rest = next(gss.split(d, groups=groups))
    rest_groups = groups[rest]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_rel, test_rel = next(gss2.split(d.iloc[rest], groups=rest_groups))
    val_idx, test_idx = rest[val_rel], rest[test_rel]

    d["split"] = "train"
    d.loc[d.index[test_idx], "split"] = "test"
    d.loc[d.index[val_idx], "split"] = "val"
    return d


def preprocess_chnci(out_dir: str = None) -> pd.DataFrame:
    cfg = load_config()
    from .data_loader import load_chnci
    d = build_splits(load_chnci())
    out_dir = ensure_dir(out_dir or p(cfg["paths"]["processed_dir"]))
    d.to_parquet(f"{out_dir}/chnci_split.parquet", index=False)
    print(f"preprocessed CHNCI: {len(d)} rows "
          f"(train={int((d.split=='train').sum())}, val={int((d.split=='val').sum())}, "
          f"test={int((d.split=='test').sum())}); "
          f"{len(load_chnci()) - len(d)} rows excluded "
          f"(invalid labels, empty/duplicate text)")
    return d
