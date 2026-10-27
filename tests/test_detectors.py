"""Unit tests for the detectors on synthetic step-change series."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cusum_detector import CusumDetector            # noqa: E402
from src.threshold_detector import RollingThresholdDetector  # noqa: E402


def make_series(before=0.1, after=0.8, n_before=40, n_after=40):
    rng = np.random.default_rng(42)
    rates = np.concatenate([rng.normal(before, 0.03, n_before),
                            rng.normal(after, 0.03, n_after)])
    hours = pd.date_range("2024-01-01", periods=len(rates), freq="h")
    return pd.DataFrame({"hour": hours, "r": rates})


def test_cusum_fires_after_step_change():
    g = make_series()
    alarms = CusumDetector().detect(g)
    assert len(alarms) >= 1
    first = pd.Timestamp(alarms[0])
    assert first > g["hour"].iloc[39]          # not before the change point
    assert first < g["hour"].iloc[55]          # within ~15 h of it


def test_cusum_silent_on_stationary_series():
    g = make_series(before=0.1, after=0.1)
    assert len(CusumDetector().detect(g)) == 0


def test_rolling_threshold_fires_after_step_change():
    g = make_series()
    alarms = RollingThresholdDetector().detect(g)
    assert len(alarms) >= 1
    assert pd.Timestamp(alarms[0]) > g["hour"].iloc[11]  # needs full baseline first


def test_cusum_ignores_nan_gaps():
    g = make_series()
    g.loc[g.index[20:30], "r"] = np.nan        # inactive hours
    alarms = CusumDetector().detect(g)
    assert len(alarms) >= 1
