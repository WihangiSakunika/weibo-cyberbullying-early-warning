"""CUSUM change detector on the standardized smoothed hostility rate.

S_t = max(0, S_{t-1} + z_t - delta);  alarm when S_t >= h;  reset after alarm.
z_t = (r_t - mu0) / sigma0 with the baseline (mu0, sigma0) estimated from the
first `baseline_points` valid observations of the incident. Defaults
delta=0.5, h=4 follow Page (1954) and the project's feasibility study.
"""
import numpy as np
import pandas as pd

from .config import load_config


class CusumDetector:
    def __init__(self, delta: float = None, h: float = None,
                 baseline_points: int = None):
        cfg = load_config()["detectors"]["cusum"]
        self.delta = delta if delta is not None else cfg["delta"]
        self.h = h if h is not None else cfg["h"]
        self.baseline_points = baseline_points or cfg["baseline_points"]

    def detect(self, g: pd.DataFrame, rate_col: str = "r") -> list:
        alarms = []
        r = g[rate_col].values
        idx = g["hour"].values
        valid = ~pd.isna(r)
        if valid.sum() < self.baseline_points:
            return []
        baseline = r[valid][:self.baseline_points]
        mu0, s0 = baseline.mean(), baseline.std()
        s0 = s0 if s0 > 0 else 1e-6
        S = 0.0
        for t in range(len(r)):
            if pd.isna(r[t]):
                continue
            z = (r[t] - mu0) / s0
            S = max(0.0, S + z - self.delta)
            if S >= self.h:
                alarms.append(idx[t])
                S = 0.0
        return alarms
