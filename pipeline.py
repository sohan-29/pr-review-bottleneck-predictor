"""Data validation, preprocessing, and model comparison for PR bottlenecks."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "contributor_experience_months",
    "files_changed",
    "lines_added",
    "lines_deleted",
    "test_coverage_percent",
    "lint_findings",
    "complexity_score",
    "code_churn_ratio",
    "is_draft",
    "labels_count",
    "author_prior_prs",
    "reviewers_requested",
    "reviewer_workload",
    "submitted_on_weekend",
]
CATEGORICAL_FEATURES = ["repo_language"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "bottleneck"
REQUIRED_COLUMNS = {"pr_id", "review_cycle_hours", TARGET, *FEATURES}
DATASET_PATH = Path(__file__).resolve().parent / "data" / "pr_review_dataset.csv"


def load_and_clean_data(path: str | Path = DATASET_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load CSV, validate its schema, remove duplicate IDs, and normalize fields."""
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")

    initial_rows = len(data)
    duplicate_count = int(data["pr_id"].duplicated().sum())
    data = data.drop_duplicates(subset="pr_id", keep="first").copy()

    for column in NUMERIC_FEATURES + ["review_cycle_hours", TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype(object).str.strip().replace("", np.nan)

    invalid_target = data[TARGET].isna() | ~data[TARGET].isin([0, 1])
    invalid_duration = data["review_cycle_hours"].isna() | (data["review_cycle_hours"] <= 0)
    invalid_rows = int((invalid_target | invalid_duration).sum())
    data = data.loc[~(invalid_target | invalid_duration)].copy()
    data[TARGET] = data[TARGET].astype(int)

    report = {
        "input_rows": initial_rows,
        "duplicate_ids_removed": duplicate_count,
        "invalid_rows_removed": invalid_rows,
        "usable_rows": len(data),
        "missing_cells_before_model_imputation": int(data[FEATURES].isna().sum().sum()),
    }
    return data, report


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values and scale/encode features inside each model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_and_compare(
    data: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit two classifiers on the same stratified holdout; exclude outcome fields."""
    if len(data) < 20 or data[TARGET].nunique() < 2:
        raise ValueError("At least 20 usable records and both target classes are required.")

    x = data[FEATURES].copy()
    y = data[TARGET].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    models: dict[str, Any] = {}
    rows = []
    for name, classifier in candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        models[name] = model
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
        )

    return models, pd.DataFrame(rows).set_index("Model"), x_test, y_test
