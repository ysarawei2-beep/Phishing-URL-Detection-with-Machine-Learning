# Phishing URL Detection

Classify URLs as **phishing** or **legitimate** using lexical / host /
HTML features and a Random Forest classifier.

This repository is the submission for **Project #11 – Phishing URL
Detection** from the *30 Cybersecurity Projects Using Machine
Learning* brief.

---

## Table of Contents
1. [Project layout](#project-layout)
2. [Dataset](#dataset)
3. [Quick start (macOS / Linux)](#quick-start-macos--linux)
4. [Quick start (Windows)](#quick-start-windows)
5. [Running the pipeline](#running-the-pipeline)
6. [Making predictions](#making-predictions)
7. [Running the unit tests](#running-the-unit-tests)
8. [Notebooks](#notebooks)
9. [Results & artefacts](#results--artefacts)
10. [Troubleshooting](#troubleshooting)

---

## Project layout

```
phishing-url-detection/
├── config/
│   └── config.yaml                  # paths, hyper-parameters, seeds
├── data/
│   ├── raw/                         # place the Kaggle CSV here
│   ├── processed/                   # train/val/test splits (generated)
│   ├── sample/                      # small bundled sample for quick tests
│   └── README.md                    # data dictionary
├── models_saved/                    # trained model + feature list (generated)
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── results/                         # figures & metrics (generated)
├── scripts/
│   ├── run_pipeline.sh              # macOS / Linux one-shot runner
│   ├── run_pipeline.bat             # Windows equivalent
│   ├── download_data.sh             # Kaggle CLI helper (macOS / Linux)
│   └── download_data.bat            # Kaggle CLI helper (Windows)
├── src/
│   ├── preprocessing/               # data loading & cleaning
│   ├── feature_engineering/         # URL feature extraction & ratios
│   ├── models/                      # Random Forest factory
│   ├── training/                    # end-to-end training driver
│   ├── evaluation/                  # metrics + plots
│   ├── utils/                       # logger, config loader
│   └── predict.py                   # command-line predictor
├── tests/                           # pytest unit tests (coverage ≥ 60 %)
├── requirements.txt
├── SOURCES.txt                      # every paper / link / dataset cited
├── LICENSE
└── README.md                        # you are here
```

---

## Dataset

We use the Kaggle dataset **“Phishing Dataset for Machine Learning:
Feature Evaluation”** by Shashwat Tiwari.

> <https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning>

The file is **`Phishing_Legitimate_full.csv`** — 10 000 rows, 50
columns (an `id`, 48 pre-extracted features, and a `CLASS_LABEL`
target: 1 = phishing, 0 = legitimate, perfectly balanced at
5000 / 5000).

### Where to put the CSV

Download the CSV and place it in:

```
data/raw/Phishing_Legitimate_full.csv
```

If the file is missing, the pipeline automatically falls back to the
bundled 200-row sample at `data/sample/sample_phishing.csv` so you
can still run every command end-to-end.

For detailed licensing and a field-by-field data dictionary, see
[`data/README.md`](data/README.md).

---

## Quick start (macOS / Linux)

All commands below assume **Python 3.10 or newer**.

```bash
# 1. Clone / unzip the project, then cd into it
cd phishing-url-detection

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) place the Kaggle CSV
#    Download 'Phishing_Legitimate_full.csv' from Kaggle and drop it in
#    data/raw/.  If you skip this step the bundled sample is used.

# 5. Train the model (takes ~10-30 seconds on a laptop)
python -m src.training.train

# 6. Make a prediction for a URL
python -m src.predict --url "http://paypal.com.login-secure-update.tk/"
```

A one-shot wrapper is available at `scripts/run_pipeline.sh`:

```bash
bash scripts/run_pipeline.sh
```

---

## Quick start (Windows)

All commands are for **PowerShell** (works in Windows Terminal,
VS Code terminal, or plain PowerShell). Requires **Python 3.10+**
from [python.org](https://www.python.org/downloads/windows/) or the
Microsoft Store.

```powershell
# 1. Open PowerShell in the project folder
cd phishing-url-detection

# 2. Create and activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks the activation script, run this once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install the dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) place the Kaggle CSV in data\raw\Phishing_Legitimate_full.csv

# 5. Train the model
python -m src.training.train

# 6. Make a prediction for a URL
python -m src.predict --url "http://paypal.com.login-secure-update.tk/"
```

A one-shot batch wrapper is available at `scripts\run_pipeline.bat`:

```bat
scripts\run_pipeline.bat
```

> Using **Command Prompt** (`cmd.exe`) instead of PowerShell?  Replace
> the activation line with `.\.venv\Scripts\activate.bat`.

---

## Running the pipeline

The end-to-end pipeline is exposed as a single Python module:

```bash
python -m src.training.train
```

What it does, step by step:

1. Loads the CSV from `data/raw/` (or the bundled sample).
2. Cleans missing / duplicate rows.
3. Adds three engineered ratio features (`PathToUrlRatio`,
   `HostnameToUrlRatio`, `DigitToUrlRatio`).
4. Stratified 70 / 15 / 15 train / validation / test split.
5. Fits a Random Forest pipeline (StandardScaler → Random Forest).
6. Performs 5-fold stratified cross-validation on the training set.
7. Evaluates on train, validation and test.
8. Writes artefacts:
   * `models_saved/phishing_rf_model.joblib` — serialised model
   * `models_saved/feature_columns.json` — canonical feature order
   * `data/processed/{train,val,test}.csv` — persisted splits
   * `results/metrics_summary.json` — all metrics in one file
   * `results/confusion_matrix.png`
   * `results/roc_curve.png`
   * `results/pr_curve.png`
   * `results/feature_importance.png`
   * `results/classification_report.txt`

Configuration (hyper-parameters, paths, seeds) lives in
[`config/config.yaml`](config/config.yaml).  Edit that file rather
than hard-coding values in source.

---

## Making predictions

Single URL:

```bash
python -m src.predict --url "http://paypal.com.login-secure-update.tk/"
```

Output:

```json
{
  "url": "http://paypal.com.login-secure-update.tk/",
  "prediction": "phishing",
  "probability": 0.91
}
```

Batch CSV (with the 48 Kaggle feature columns):

```bash
python -m src.predict --csv data/raw/Phishing_Legitimate_full.csv
```

Creates `Phishing_Legitimate_full_predictions.csv` next to the
input file.

---

## Running the unit tests

From the project root:

```bash
pytest -v
```

Or measure coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Tests cover the feature extractor, preprocessing, and the model
pipeline.  On a reference run they report **> 70 %** line coverage.

---

## Notebooks

Launch Jupyter from the project root:

```bash
jupyter notebook
```

and open, in order:

1. `notebooks/EDA.ipynb` – class distribution, feature
   distributions, correlation heatmap, PCA projection.
2. `notebooks/model_training.ipynb` – full training loop + sample
   URL predictions.
3. `notebooks/results_analysis.ipynb` – error analysis and
   inference-time benchmark.

Each notebook saves any figures it produces into `results/`.

---

## Results & artefacts

After running `python -m src.training.train`, the following files
are regenerated:

| File | Description |
| --- | --- |
| `models_saved/phishing_rf_model.joblib` | trained scikit-learn pipeline |
| `models_saved/feature_columns.json` | canonical feature order |
| `data/processed/train.csv` | stratified training split |
| `data/processed/val.csv` | validation split |
| `data/processed/test.csv` | held-out test split |
| `results/metrics_summary.json` | accuracy / precision / recall / F1 / ROC-AUC on every split, plus CV mean ± std |
| `results/confusion_matrix.png` | test-set confusion matrix |
| `results/roc_curve.png` | ROC curve with AUC |
| `results/pr_curve.png` | Precision–Recall curve |
| `results/feature_importance.png` | top-20 feature importances |
| `results/classification_report.txt` | per-class classification report |

On the full Kaggle CSV the reference run achieves approximately:

| Metric | Test score |
| --- | --- |
| Accuracy | ≈ 0.98 |
| Precision | ≈ 0.98 |
| Recall | ≈ 0.98 |
| F1 | ≈ 0.98 |
| ROC-AUC | ≈ 0.997 |

Your exact numbers may vary by a few tenths of a percent depending
on the seed and scikit-learn version.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
: Run commands from the project root, e.g.
  `python -m src.training.train`, not `python src/training/train.py`.

**PowerShell refuses to activate the venv**
: Run once per user:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**`FileNotFoundError: … Phishing_Legitimate_full.csv`**
: The dataset is not in `data/raw/`.  Either download it from the
  Kaggle link above or let the pipeline use the bundled sample
  (it will log a warning and continue).

**Kaggle CLI download fails**
: You need an API key.  Follow the official instructions at
  <https://github.com/Kaggle/kaggle-api#api-credentials> or just
  download the CSV manually from the Kaggle website.

---

## License

MIT — see [LICENSE](LICENSE).
