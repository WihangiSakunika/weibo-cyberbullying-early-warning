# If Hate Arrives in Waves: LLM-Based Early Detection of Cyberbullying Surges on Weibo

Reproducible research repository for thesis work on **early detection of
cyberbullying/hostility surges** on Chinese social media, built on two public
datasets: **CHNCI** (Chinese Cyberbullying Incident Detection, 225,480 comments
across 95 incidents) and **SWSR** (Sina Weibo Sexism Review, 8,969 comments).

---

## 1. Research objectives and questions

- **RQ1** Can comment-level hostility be predicted reliably enough from text to
  serve as a monitoring signal? *(comment-level classification)*
- **RQ2** How do hostility dynamics evolve over the lifetime of a trending
  incident? *(hourly aggregation, CB vs non-CB incidents)*
- **RQ3** Can simple change-detection baselines (rolling threshold, CUSUM)
  raise earlier and more reliable surge warnings than volume-based signals,
  at an acceptable false-alarm cost? *(event-level early-warning evaluation)*

## 2. Repository structure

```text
weibo-cyberbullying-early-warning/
├── README.md                          <- this file
├── requirements.txt                   # Python dependencies (CPU-only baselines)
├── .gitignore                         # keeps CHNCI, processed data, secrets out of git
├── LICENSE                            # MIT for code + dataset-license notes
├── configs/experiment_config.yaml     # single source of truth: seeds, splits, detector params
├── data/
│   ├── README.md                      # dataset descriptions, licenses, download instructions
│   ├── raw/swsr/                      # SWSR committed (MIT license permits redistribution)
│   ├── raw/chnci/                     # gitignored - download with scripts/download_data.py
│   └── processed/                     # generated: split corpus, hourly series
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_surge_detection_evaluation.ipynb
├── src/
│   ├── config.py                      # paths + YAML config loader
│   ├── data_loader.py                 # CHNCI/SWSR loaders, label construction, text cleaning
│   ├── data_audit.py                  # data-quality report (JSON + TXT)
│   ├── preprocessing.py               # cleaning + incident-level 70/15/15 group splits
│   ├── train.py                       # TF-IDF+LR and TF-IDF+SVM training & evaluation
│   ├── train_transformer.py           # PLANNED transformer baseline (not yet implemented)
│   ├── evaluate.py                    # metrics, classification reports, confusion matrices
│   ├── temporal_analysis.py           # hourly aggregation + scoring of full corpus
│   ├── threshold_detector.py          # rolling mean + 2σ baseline detector
│   ├── cusum_detector.py              # CUSUM detector (δ=0.5, h=4)
│   └── early_warning_evaluation.py    # event-level metrics vs Q90 onset proxy
├── tests/
│   ├── test_data_loader.py            # 3 tests (loader, cleaning, encoding)
│   └── test_detectors.py              # 4 tests (CUSUM/rolling on synthetic step changes)
├── results/
│   ├── README.md
│   ├── metrics/                       # audit JSON, model comparison CSV, alarm events, EW summary
│   └── figures/                       # confusion matrices, temporal plots, alarm timelines
└── scripts/
    ├── download_data.py               # clone both official dataset repos (verification included)
    ├── run_experiments.py             # full pipeline: audit -> train -> temporal -> early warning
    └── reproduce_results.py           # re-run + verify all required result files exist
```

## 3. Datasets

| | CHNCI | SWSR |
|---|---|---|
| Source | https://github.com/zhuyiYZU/CHNCI | https://github.com/aggiejiang/SWSR |
| Size | 225,480 comments, 95 incidents (47 CB / 48 non-CB) | 8,969 comments (3,093 sexist), 1,527 weibos, 3,016-term lexicon |
| Fields | `timestamp, platform, label1, label2, label3, diff, content` | `comment_text, label, category, target, date, gender, location, like` |
| Labels | comment-level cyberbullying, 3 annotation strategies; here: majority vote (≥2 of 3) | binary sexism label by trained annotators |
| Timestamps | minute-level, 0 unparseable, 2024-07-05 → 2024-10-24 | minute-level (200 rows nonstandard), 2012 → 2020 |
| License | **None** — use with citation, do **not** redistribute | **MIT** (© 2022 Aiqi Jiang) — committed in `data/raw/swsr/` |
| Role | main classification + temporal/early-warning dataset | cross-domain (zero-shot) external evaluation only |

Full audit numbers: `results/metrics/dataset_audit.json`. Download commands:

```bash
python scripts/download_data.py                # both datasets
python scripts/download_data.py --dataset chnci   # required before running experiments
```

## 4. Dataset compatibility design (not auto-merged)

CHNCI (cyberbullying) and SWSR (sexism) use different label schemes, text
fields and time ranges. The repository therefore uses:

1. **Separate experiments per dataset** (CHNCI for training/temporal analysis,
   SWSR untouched for external evaluation);
2. **Zero-shot transfer evaluation**: models trained on CHNCI, evaluated
   unchanged on SWSR (sexism treated as a hostility proxy);
3. **CHNCI alone for temporal/early-warning analysis** — SWSR threads have a
   median of 2 comments each and cannot support surge detection.

## 5. Installation

```bash
git clone https://github.com/<you>/weibo-cyberbullying-early-warning.git
cd weibo-cyberbullying-early-warning
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -q                           # 7 tests should pass
```

## 6. Google Colab setup

```python
!git clone https://github.com/<you>/weibo-cyberbullying-early-warning.git
%cd weibo-cyberbullying-early-warning
!pip install -r requirements.txt
!python scripts/download_data.py          # fetches CHNCI (not committed, license)
```

Then open the notebooks under `notebooks/` — each has a first cell that
switches to the repo root automatically.

## 7–10. Commands

```bash
# data audit (writes results/metrics/dataset_audit.{json,txt})
python -m src.data_audit

# preprocessing + training + evaluation (writes model_comparison.csv, confusion_*.png)
python -m src.train

# full pipeline, end-to-end (audit -> train -> temporal -> early warning -> figures)
python scripts/run_experiments.py

# re-run everything with the fixed seed and verify outputs exist
python scripts/reproduce_results.py
```

## 11. Early-warning methodology

- Comments are scored with the frozen TF-IDF+LR scorer; per incident, hourly
  `hostile_pred / volume` gives a hostility-rate series (hours with < 5
  comments masked).
- **Detectors**: (a) rolling mean + 2σ of the trailing 12 valid observations of
  the 6-h smoothed rate; (b) CUSUM on the standardized smoothed rate,
  δ = 0.5, h = 4 (σ units), reset after alarm.
- **Surge ground truth (statistical proxy)**: onset = first hour the smoothed
  rate exceeds its own 90th percentile; peak = its maximum.
- **Metrics**: CB-incident detection rate, warning lead time to peak, detection
  delay from onset, false alarms in non-CB incidents (count and per-100-active-hours).
  Identical rules for both methods.

## 12. Actual experimental results (seed 42 — all numbers below were produced by `scripts/run_experiments.py`)

### Model comparison (CHNCI incident-level split 70/15/15, test n = 43,887)

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | AP | SWSR zero-shot AUC |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogisticRegression (thr 0.40) | 0.7032 | 0.6285 | 0.6271 | 0.6278 | **0.6905** | **0.7660** | 0.6807 | 0.6951 |
| TF-IDF + LinearSVM, calibrated (thr 0.35) | 0.6978 | 0.6153 | 0.6479 | 0.6312 | 0.6876 | 0.7617 | 0.6772 | 0.6928 |

On the unanimous-label test subset (n = 33,606): LR AUC 0.7975, SVM AUC 0.7934.
The test split contains only incidents unseen during training.

### Temporal analysis

Cyberbullying incidents sustain **2.0× the hostility rate** of non-CB incidents
(mean hourly rate 0.371 vs 0.187) while lasting longer (162 vs 93 active hours)
and drawing more comments (3,634 vs 871 mean total). Hostility rate and volume
are nearly uncorrelated at all lags (|r| < 0.05, ±12 h), so rate-based
monitoring adds signal that volume alone cannot provide.

### Early-warning evaluation (95 incidents)

| Method | CB incidents alarmed | Detected after onset | Median lead to peak | Median detection delay | Non-CB incidents with ≥1 FA | FA per 100 active hours |
|---|---|---|---|---|---|---|
| Rolling mean+2σ | 35/47 (74%) | 26/47 | **−6.0 h** | 22 h | 21/48 | 2.12 |
| **CUSUM (δ=0.5, h=4)** | **36/47 (77%)** | **36/47** | −3.0 h | **14 h** | 21/48 | **2.04** |

CUSUM detects every incident it alarms *after* the statistical onset, with a
14 h median delay, at the same false-alarm cost. The rolling threshold warns
earlier relative to the peak (−6 h) but 9 of its alarms precede the formal
onset. Full tables: `results/metrics/early_warning_summary.csv`,
`results/metrics/alarm_events.csv`.

## 13. Figures

- `results/figures/temporal_examples.png` — hourly volume vs hostility rate for 2 CB + 2 non-CB incidents
- `results/figures/peak_rate_hist.png`, `results/figures/alarm_timelines.png` — alarm placement vs onset/peak
- `results/figures/confusion_tfidf_lr.png`, `results/figures/confusion_tfidf_svm.png`

## 14. Limitations and known issues

1. **Surge onsets are a statistical proxy** (Q90 of each incident's own smoothed
   rate), not external annotations — lead/delay values are internally
   consistent method comparisons, not externally anchored warnings. Anchoring
   onsets to documented real-world escalation evidence is the top priority.
2. **Label provenance**: CHNCI labels are LLM-assisted with human verification;
   the scorer and the rate series share this provenance. The unanimous-label
   subset and the SWSR transfer partially mitigate this.
3. **One collection window** (2024-07→10, 95 incidents); false-alarm rates
   estimated over ~1,300 non-CB active hours only.
4. **Scorer ceiling**: TF-IDF/LR reaches AUC 0.77; a transformer would likely
   sharpen rate estimates (planned, not yet run).
5. Group-level splits make split sizes approximate (train/val/test =
   140,056/16,870/43,887 rows rather than exactly 70/15/15).

## 15. Reproducibility

- Single seed (42) in `configs/experiment_config.yaml`; every hyperparameter
  (splits, thresholds, detector parameters) is config-driven.
- Thresholds are tuned on the validation split only; the test split is used once.
- `python scripts/reproduce_results.py` re-runs the pipeline and verifies all
  committed result files exist.

## 16. Ethical considerations

The datasets contain hostile, sexist and cyberbullying text collected from
public platforms; comments may contain personally identifying information.
They are used here solely for abuse-detection research. CHNCI carries no open
license — it is downloaded locally, never redistributed via this repository,
and must be cited (Zhu, Zou & Wu 2025, arXiv:2505.20654). SWSR is MIT-licensed
and redistributed with attribution. Deployed early-warning systems could cause
over-moderation harm; the reported false-alarm rates quantify that risk.

## 17. Citation

```bibtex
@article{zhu2025chinese,
  title={Chinese Cyberbullying Detection: Dataset, Method, and Validation},
  author={Zhu, Yi and Zou, Xin and Wu, Xindong},
  journal={arXiv preprint arXiv:2505.20654}, year={2025}}
@article{jiang2022swsr,
  title={SWSR: A Chinese dataset and lexicon for online sexism detection},
  author={Jiang, Aiqi and Yang, Xiaohan and Liu, Yang and Zubiaga, Arkaitz},
  journal={Online Social Networks and Media}, volume={27}, pages={100182}, year={2022}}
```

## 18. Completed vs. planned experiments

| Experiment | Status |
|---|---|
| Dataset audit (CHNCI + SWSR) | ✅ run — `results/metrics/dataset_audit.json` |
| TF-IDF + LogisticRegression | ✅ run — `results/metrics/model_comparison.csv` |
| TF-IDF + LinearSVM (calibrated) | ✅ run |
| SWSR zero-shot transfer | ✅ run |
| Temporal analysis (hourly, 95 incidents) | ✅ run |
| Rolling-2σ + CUSUM early-warning evaluation | ✅ run |
| Unit tests (7) | ✅ passing |
| Transformer baseline (`src/train_transformer.py`) | ⏳ **planned, not implemented/run** |
| Externally anchored surge onsets | ⏳ planned |
