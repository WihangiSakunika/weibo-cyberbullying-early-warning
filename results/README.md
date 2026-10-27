# Results

All files in this directory are **actual outputs** of
`python scripts/run_experiments.py` (seed 42). Nothing is hand-written.

## metrics/

| File | Contents |
|---|---|
| `dataset_audit.json` / `.txt` | full data-quality report for both datasets |
| `chnci_per_incident.csv` | rows, time span per incident |
| `model_comparison.csv` / `.json` | LR + SVM test metrics, unanimous-subset metrics, SWSR zero-shot |
| `classification_report_tfidf_lr.txt` / `_tfidf_svm.txt` | per-class precision/recall/F1 |
| `temporal_summary.csv` | mean peak/mean hostility rate per incident type |
| `alarm_events.csv` | every alarm with lead/delay vs onset & peak |
| `early_warning_summary.csv` | event-level detector comparison |

## figures/

`confusion_tfidf_lr.png`, `confusion_tfidf_svm.png`,
`temporal_examples.png`, `alarm_timelines.png`, `peak_rate_hist.png`.

Regenerate everything with `python scripts/reproduce_results.py`.
