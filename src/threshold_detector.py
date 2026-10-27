"""Rolling mean + k*sigma threshold detector (statistical baseline).

Alarm rule: on the smoothed hostility rate r_t (only over active hours),
alarm when r_t > mean(history) + n_sigma * std(history), where history is the
trailing `baseline_window` valid observations. The baseline resets after each
alarm so each alarm is reported separately.
"""
import numpy as np
import pandas as pd

from .config import load_config


class RollingThresholdDetector:
    def __init__(self, baseline_window: int = None, n_sigma: float = None):
        cfg = load_config()["detectors"]["rolling_threshold"]
        self.baseline_window = baseline_window or cfg["baseline_window"]
        self.n_sigma = n_sigma if n_sigma is not None else cfg["n_sigma"]

    def detect(self, g: pd.DataFrame, rate_col: str = "r") -> list:
        """`g` must be time-sorted with an hourly `hour` column and a smoothed
        rate column `rate_col` (NaN on inactive hours). Returns alarm hours."""
        alarms, hist = [], []
        for _, row in g.iterrows():
            r = row[rate_col]
            if pd.isna(r):
                continue
            if len(hist) >= self.baseline_window:
                a = np.asarray(hist)
                mu, sd = a.mean(), a.std()
                if sd > 0 and r > mu + self.n_sigma * sd:
                    alarms.append(row["hour"])
                    hist = []
            hist.append(r)
        return alarms
