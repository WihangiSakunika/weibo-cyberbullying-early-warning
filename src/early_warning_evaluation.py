"""Early-warning evaluation: compare the rolling-threshold and CUSUM detectors
on the same ground truth with identical rules.

Surge ground truth (STATISTICAL PROXY - explicitly not an external annotation):
  per incident, on the 6h-smoothed hostility rate over active hours
  (volume >= min_volume):
    onset = first hour the smoothed rate exceeds its own Q90
    peak  = hour of maximum smoothed rate
Metrics (event level):
  - detected: >=1 alarm at/after onset in a CB incident
  - warning lead time = alarm - peak   (negative = before the peak)
  - detection delay   = alarm - onset
  - false alarms: alarms in non-CB incidents (count and per-100-active-hours)
Limitation: because the onset is derived from the same rate series the
detectors monitor, lead/delay numbers are internally consistent but not
externally anchored. Documented in README section 14.
"""
import numpy as np
import pandas as pd

from .config import load_config, p, ensure_dir


def add_smoothed_rate(series: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()["temporal"]
    out = []
    for inc, g in series.sort_values("hour").groupby("incident"):
        g = g.copy()
        ok = g["volume"] >= cfg["min_volume"]
        g["r"] = g["hostile_rate"].where(ok).rolling(6, min_periods=3).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def ground_truth(g: pd.DataFrame, q: float):
    r = g["r"].dropna()
    if len(r) < 12:
        return None, None
    onset_mask = g["r"] > r.quantile(q)
    onset = g.loc[onset_mask.idxmax(), "hour"] if onset_mask.any() else None
    peak = g.loc[g["r"].idxmax(), "hour"]
    return onset, peak


def run(series: pd.DataFrame = None) -> pd.DataFrame:
    from .threshold_detector import RollingThresholdDetector
    from .cusum_detector import CusumDetector

    cfg = load_config()
    if series is None:
        series = pd.read_parquet(p(cfg["paths"]["processed_dir"]) + "/hourly_series.parquet")
    series = add_smoothed_rate(series)
    q = cfg["evaluation"]["surge_quantile"]

    detectors = {"rolling_mean_2sd": RollingThresholdDetector(),
                 "cusum_delta0.5_h4": CusumDetector()}
    rows = []
    for inc, g in series.groupby("incident"):
        onset, peak = ground_truth(g, q)
        if onset is None:
            continue
        lab = int(g["incident_label"].iloc[0])
        active = int((g["volume"] >= cfg["temporal"]["min_volume"]).sum())
        for name, det in detectors.items():
            for alarm in det.detect(g):
                a = pd.Timestamp(alarm)
                rows.append(dict(incident=inc, incident_label=lab, method=name,
                                 alarm=a, onset=onset, peak=peak,
                                 lead_h=(a - peak).total_seconds() / 3600,
                                 delay_h=(a - onset).total_seconds() / 3600,
                                 active_hours=active))
    ev = pd.DataFrame(rows)
    metrics_dir = ensure_dir(p(cfg["paths"]["metrics_dir"]))
    ev.to_csv(f"{metrics_dir}/alarm_events.csv", index=False, encoding="utf-8-sig")

    n_cb = int(series.groupby("incident")["incident_label"].first().eq(1).sum())
    n_ncb = int(series["incident"].nunique()) - n_cb
    summary = []
    for name in detectors:
        m = ev[ev.method == name]
        cb, ncb = m[m.incident_label == 1], m[m.incident_label == 0]
        detected = int(cb[cb.delay_h >= 0]["incident"].nunique())
        summary.append({
            "method": name,
            "cb_incidents_alarmed": f"{cb['incident'].nunique()}/{n_cb}",
            "cb_detected_after_onset": f"{detected}/{n_cb}",
            "median_lead_to_peak_h": round(float(cb["lead_h"].median()), 1) if len(cb) else None,
            "median_detection_delay_h": round(float(cb.loc[cb.delay_h >= 0, "delay_h"].median()), 1)
            if len(cb[cb.delay_h >= 0]) else None,
            "non_cb_incidents_with_fa": f"{ncb['incident'].nunique()}/{n_ncb}",
            "false_alarms_total": int(len(ncb)),
            "fa_per_100_active_hours": round(100 * len(ncb) / ncb["active_hours"].sum(), 2)
            if len(ncb) else 0.0,
        })
    s = pd.DataFrame(summary)
    s.to_csv(f"{metrics_dir}/early_warning_summary.csv", index=False)
    print(s.to_string(index=False))
    print(f"\nNOTE: surge onsets are a statistical proxy (Q{int(q*100)} of each "
          "incident's own smoothed rate), not external annotations.")
    return s
