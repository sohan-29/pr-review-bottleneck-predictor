# Git Repository Code Health & PR Review Bottleneck Predictor

A Python data analytics and visualization web application for a DAV mini project. It explores pull-request review patterns and compares two machine-learning classifiers for identifying review bottlenecks.

## Dataset

`data/pr_review_dataset.csv` contains 1,500 reproducible **synthetic** pull-request records. The generator uses NumPy's seeded random number generator (`seed=42`), and the CSV is included so the application is immediately runnable. Synthetic data is used because the project does not have a real GitHub dataset or credentials; it must not be presented as measured production data.

The binary target `bottleneck` is 1 when review completion takes more than 72 hours. `review_cycle_hours` is used only to derive the target and is excluded from predictors to prevent target leakage. Inputs model information plausibly known when a PR is opened: change size, language, test coverage, static-analysis findings, code complexity/churn, contributor experience, requested reviewers, reviewer workload, labels, and submission timing.

Regenerate the CSV from this directory:

```bash
python data/generate_dataset.py
```

## Features

- Data validation, duplicate-ID removal, invalid outcome filtering, missing-value reporting, and model-pipeline imputation.
- Exploratory visualizations: review-time distribution and threshold, bottleneck rates by language and test-coverage band, code-health signal scatterplot, and numeric correlation matrix.
- Logistic Regression and Random Forest compared on the same stratified 75/25 holdout using accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- Simple model analytics view: select a model to see its metrics, performance graph, confusion matrix, feature importance, and the full dataset together.
- Interactive form to estimate review-bottleneck risk for a new PR.
- Cleaned dataset preview and CSV download.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate` instead.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## DAV methodology

1. **Wrangling:** load the CSV, validate required columns, remove duplicate PR IDs, coerce numeric fields, remove records with invalid targets/outcomes, report remaining missing values, and impute missing predictor values within each training pipeline.
2. **EDA:** inspect class balance, review-cycle distribution, language-specific bottleneck rates, and numeric correlations.
3. **Modeling:** compare a scaled Logistic Regression baseline with a Random Forest using a stratified, fixed-seed holdout. Both fit preprocessing only on training folds and use the same test records.
4. **Evaluation:** compare precision, recall, F1, accuracy, ROC-AUC, and confusion matrices; F1 is used to select the default model shown in the prediction form.

Predictions are educational estimates learned from synthetic examples, not live GitHub risk assessments. More credible conclusions require an appropriately licensed, real-world dataset and careful validation.
