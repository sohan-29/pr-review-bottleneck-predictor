"""Generate a reproducible synthetic pull-request review dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "pr_review_dataset.csv"
RANDOM_SEED = 42
RECORD_COUNT = 1500


def generate_dataset(record_count: int = RECORD_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create PR records with realistic, intentionally imperfect input data."""
    if record_count < 1000:
        raise ValueError("The project dataset must contain at least 1,000 records.")

    rng = np.random.default_rng(seed)
    languages = np.array(["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "Other"])
    language = rng.choice(languages, size=record_count, p=[0.21, 0.19, 0.17, 0.14, 0.10, 0.08, 0.11])
    experience = np.clip(rng.gamma(shape=2.4, scale=15, size=record_count), 1, 120).round().astype(int)
    files_changed = np.clip(rng.lognormal(mean=1.2, sigma=0.9, size=record_count), 1, 100).round().astype(int)
    lines_added = np.clip(rng.lognormal(mean=4.0, sigma=1.1, size=record_count), 1, 12000).round().astype(int)
    lines_deleted = np.clip(rng.lognormal(mean=3.5, sigma=1.2, size=record_count), 0, 10000).round().astype(int)
    test_coverage_percent = np.clip(rng.normal(loc=73, scale=16, size=record_count), 0, 100).round(1)
    lint_findings = np.clip(
        rng.poisson(lam=1.5 + lines_added / 650 + files_changed / 12, size=record_count), 0, 80
    )
    complexity_score = np.clip(
        rng.lognormal(mean=1.4, sigma=0.8, size=record_count) + files_changed / 10, 1, 100
    ).round(1)
    code_churn_ratio = (lines_added / (lines_added + lines_deleted + 1)).round(3)
    draft = rng.binomial(1, 0.22, size=record_count)
    labels_count = np.clip(rng.poisson(lam=1.5, size=record_count), 0, 10)
    author_prior_prs = np.clip(rng.lognormal(mean=2.6, sigma=1.2, size=record_count), 0, 1000).round().astype(int)
    reviewers_requested = np.clip(rng.poisson(lam=2.2, size=record_count) + 1, 1, 8)
    reviewer_workload = np.clip(rng.gamma(shape=2.0, scale=4.0, size=record_count), 0, 50).round().astype(int)
    weekend = rng.binomial(1, 0.28, size=record_count)

    log_delay = (
        3.35
        + 0.24 * np.log1p(files_changed)
        + 0.11 * np.log1p(lines_added + lines_deleted)
        + 0.035 * reviewer_workload
        + 0.11 * reviewers_requested
        + 0.25 * weekend
        + 0.018 * lint_findings
        + 0.09 * complexity_score
        - 0.006 * test_coverage_percent
        - 0.012 * np.log1p(author_prior_prs)
        - 0.012 * experience
        - 0.28 * draft
        + rng.normal(0, 0.65, size=record_count)
    )
    review_cycle_hours = np.clip(np.exp(log_delay), 1, 240).round(1)
    bottleneck = (review_cycle_hours > 72).astype(int)

    data = pd.DataFrame(
        {
            "pr_id": [f"PR-{index:05d}" for index in range(1, record_count + 1)],
            "repo_language": language,
            "contributor_experience_months": experience,
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "test_coverage_percent": test_coverage_percent,
            "lint_findings": lint_findings,
            "complexity_score": complexity_score,
            "code_churn_ratio": code_churn_ratio,
            "is_draft": draft,
            "labels_count": labels_count,
            "author_prior_prs": author_prior_prs,
            "reviewers_requested": reviewers_requested,
            "reviewer_workload": reviewer_workload,
            "submitted_on_weekend": weekend,
            "review_cycle_hours": review_cycle_hours,
            "bottleneck": bottleneck,
        }
    )

    # Missing values make the data-cleaning step visible and testable.
    for column, fraction in {
        "repo_language": 0.015,
        "contributor_experience_months": 0.02,
        "reviewer_workload": 0.04,
    }.items():
        missing_rows = rng.choice(record_count, size=max(1, int(record_count * fraction)), replace=False)
        data.loc[missing_rows, column] = np.nan

    return data


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT, index=False)
    print(f"Saved {len(dataset):,} PR records to {OUTPUT}")
    print(f"Bottleneck prevalence: {dataset['bottleneck'].mean():.1%}")
