"""Interactive DAV lab dashboard for pull-request review bottlenecks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline

from pipeline import DATASET_PATH, FEATURES, load_and_clean_data, train_and_compare


st.set_page_config(
    page_title="PR Review Bottleneck Predictor",
    page_icon="🔎",
    layout="wide",
)

NAVY = "#17324D"
BLUE = "#2878B5"
ORANGE = "#F28E2B"
GREEN = "#2A9D8F"
RED = "#D9534F"


@st.cache_data
def get_data() -> tuple[pd.DataFrame, dict]:
    return load_and_clean_data(DATASET_PATH)


@st.cache_resource
def get_models(data: pd.DataFrame):
    return train_and_compare(data)


def styled_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


@st.cache_data
def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include="number").drop(columns=["bottleneck"], errors="ignore")
    return numeric.corr()


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return the strongest model signals with readable feature names."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    values = (
        classifier.feature_importances_
        if hasattr(classifier, "feature_importances_")
        else np.abs(classifier.coef_[0])
    )

    importance = pd.DataFrame({"Feature": names, "Importance": values})
    importance["Feature"] = (
        importance["Feature"]
        .str.replace(r"^(numeric|categorical)__", "", regex=True)
        .str.replace("_", " ")
    )
    return importance.nlargest(10, "Importance").sort_values("Importance")


data, cleaning = get_data()
models, scores, x_test, y_test = get_models(data)
prevalence = float(data["bottleneck"].mean() * 100)
best_model_name = str(scores["F1"].idxmax())

st.title("PR Review Bottleneck Predictor")
st.caption("Explore code health, compare review-delay models, and estimate risk for a pull request.")
st.info(
    f"Using **{len(data):,} synthetic teaching records** (not live GitHub data). "
    "A review bottleneck means review takes more than 72 hours."
)

tab_overview, tab_models, tab_eda, tab_predict = st.tabs(
    ["Project overview", "Model analytics & data", "Explore data", "Try a prediction"]
)

with tab_overview:
    st.subheader("What this project does")
    kpis = st.columns(4)
    kpis[0].metric("Usable PR records", f"{len(data):,}")
    kpis[1].metric("Bottleneck rate", f"{prevalence:.1f}%")
    kpis[2].metric("Median review time", f"{data['review_cycle_hours'].median():.1f} hours")
    kpis[3].metric("Best holdout F1", f"{scores.loc[best_model_name, 'F1']:.3f}", best_model_name)

    st.markdown("#### How to use")
    st.write(
        "1. Open **Model analytics & data** and select Logistic Regression or Random Forest. "
        "Its scores, graphs, and the complete dataset appear together."
    )
    st.write(
        "2. Open **Try a prediction** to enter PR details and estimate the chance of a review taking over 72 hours."
    )
    st.caption(
        "Both models are tested on the same stratified holdout set. Review duration defines the target, "
        "but is never used as a model input."
    )

with tab_eda:
    st.subheader("Exploratory data analysis")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data["review_cycle_hours"], bins=30, color=BLUE, alpha=0.88, edgecolor="white")
        ax.axvline(72, color=RED, linestyle="--", linewidth=2, label="72-hour bottleneck threshold")
        ax.set_xlabel("Review cycle (hours)")
        ax.set_ylabel("PR count")
        ax.legend(frameon=False)
        styled_axis(ax, "Review cycle time distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        language_rates = (
            data.groupby("repo_language", observed=True)["bottleneck"].mean().mul(100).sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(language_rates.index, language_rates.values, color=GREEN)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Bottleneck PRs (%)")
        styled_axis(ax, "Bottleneck rate by repository language")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        coverage_bins = pd.cut(data["test_coverage_percent"], bins=[-1, 40, 60, 80, 101])
        coverage_rates = data.groupby(coverage_bins, observed=False)["bottleneck"].mean().mul(100)
        bars = ax.bar(coverage_rates.index.astype(str), coverage_rates.values, color=ORANGE)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlabel("Test coverage (%)")
        ax.set_ylabel("Bottleneck PRs (%)")
        styled_axis(ax, "Review bottlenecks by test coverage")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        scatter = ax.scatter(
            data["complexity_score"],
            data["test_coverage_percent"],
            c=data["bottleneck"],
            cmap=ListedColormap([BLUE, ORANGE]),
            alpha=0.55,
            s=20,
        )
        ax.set_xlabel("Code complexity score")
        ax.set_ylabel("Test coverage (%)")
        styled_axis(ax, "Code health signals (orange = bottleneck)")
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=["Not bottleneck", "Bottleneck"],
            frameon=False,
            loc="best",
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    corr = correlation_matrix(data)
    st.markdown("#### Numeric feature correlation")
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation", shrink=0.8)
    ax.set_title("Correlation matrix (outcome duration excluded)", loc="left", color=NAVY, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_models:
    st.subheader("Model results and dataset")
    chosen = st.radio(
        "Choose a model to see its results",
        list(models),
        index=list(models).index(best_model_name),
        horizontal=True,
    )
    st.caption(f"Evaluated on the same {len(y_test):,}-record holdout set (25% of the data).")

    model_metrics = scores.loc[chosen]
    metric_help = {
        "Accuracy": "Share of all holdout PRs classified correctly.",
        "Precision": "Of PRs flagged as bottlenecks, how many were bottlenecks.",
        "Recall": "Of actual bottlenecks, how many the model detected.",
        "F1": "Balance between bottleneck precision and recall.",
        "ROC-AUC": "How well the model ranks bottleneck risk across thresholds.",
    }
    metric_columns = st.columns(5)
    for column, metric_name in zip(metric_columns, scores.columns):
        column.metric(metric_name, f"{model_metrics[metric_name]:.3f}", help=metric_help[metric_name])

    predictions = models[chosen].predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metric_chart, matrix_chart = st.columns(2)
    with metric_chart:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            model_metrics.index,
            model_metrics.values,
            color=[BLUE, GREEN, ORANGE, RED, NAVY],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Score (higher is better)")
        styled_axis(ax, f"{chosen}: model performance")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with matrix_chart:
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["On time", "Bottleneck"]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d"
        )
        ax.set_title("Correct and incorrect predictions", loc="left", color=NAVY, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        "F1 balances missed bottlenecks and false alarms. ROC-AUC measures how well the model ranks "
        "bottleneck risk across probability thresholds."
    )

    importance = get_feature_importance(models[chosen])
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(importance["Feature"], importance["Importance"], color=GREEN)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    styled_axis(ax, f"Most influential features — {chosen}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Feature importance explains what influenced the selected model, not proof that a feature causes delays. "
        "Logistic Regression uses absolute coefficients; Random Forest uses split-based importance."
    )

    st.markdown("#### Compare both models")
    st.dataframe(scores.style.format("{:.3f}"), width="stretch")

    st.markdown("#### Dataset records")
    st.caption(
        f"All {len(data):,} cleaned records are shown below. Scroll the table to inspect more rows. "
        "Review duration defines the target but is not given to the models."
    )
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download all records (CSV)", csv, "pr_review_dataset.csv", "text/csv")
    st.dataframe(data, height=420, hide_index=True, width="stretch")
    st.caption(
        f"Cleaning: {cleaning['duplicate_ids_removed']} duplicate IDs and "
        f"{cleaning['invalid_rows_removed']} invalid outcome rows removed; "
        f"{cleaning['missing_cells_before_model_imputation']} missing feature values are imputed during model training."
    )
with tab_predict:
    st.subheader("Estimate bottleneck risk for a new pull request")
    st.caption("Enter details known when the PR is opened. Fields describing its eventual review outcome are not requested.")
    with st.form("prediction_form"):
        one, two, three = st.columns(3)
        with one:
            repo_language = st.selectbox("Repository language", ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "Other"])
            contributor_experience_months = st.slider("Contributor experience (months)", 1, 120, 24)
            files_changed = st.slider("Files changed", 1, 100, 8)
            lines_added = st.number_input("Lines added", 0, 12000, 120, step=10)
            test_coverage_percent = st.slider("Test coverage (%)", 0, 100, 75)
        with two:
            lines_deleted = st.number_input("Lines deleted", 0, 10000, 40, step=10)
            lint_findings = st.slider("Lint findings", 0, 80, 2)
            complexity_score = st.slider("Code complexity score", 1, 100, 12)
            code_churn_ratio = st.slider("Code churn ratio", 0.0, 1.0, 0.75, step=0.01)
            is_draft = st.checkbox("PR is a draft")
            labels_count = st.slider("Number of labels", 0, 10, 2)
            author_prior_prs = st.number_input("Author's previous PRs", 0, 1000, 20, step=1)
        with three:
            reviewers_requested = st.slider("Reviewers requested", 1, 8, 2)
            reviewer_workload = st.slider("Current reviewer workload (open PRs)", 0, 50, 8)
            submitted_on_weekend = st.checkbox("Submitted on a weekend")
            prediction_model = st.selectbox("Prediction model", list(models), index=list(models).index(best_model_name))

        submitted = st.form_submit_button("Estimate review bottleneck risk", type="primary")

    if submitted:
        new_pr = pd.DataFrame(
            [
                {
                    "repo_language": repo_language,
                    "contributor_experience_months": contributor_experience_months,
                    "files_changed": files_changed,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "test_coverage_percent": test_coverage_percent,
                    "lint_findings": lint_findings,
                    "complexity_score": complexity_score,
                    "code_churn_ratio": code_churn_ratio,
                    "is_draft": int(is_draft),
                    "labels_count": labels_count,
                    "author_prior_prs": author_prior_prs,
                    "reviewers_requested": reviewers_requested,
                    "reviewer_workload": reviewer_workload,
                    "submitted_on_weekend": int(submitted_on_weekend),
                }
            ],
            columns=FEATURES,
        )
        model = models[prediction_model]
        probability = float(model.predict_proba(new_pr)[0, 1])
        result_columns = st.columns([1, 2])
        with result_columns[0]:
            st.metric("Estimated bottleneck probability", f"{probability:.1%}")
        with result_columns[1]:
            if probability >= 0.5:
                st.warning("Higher estimated risk: this PR is more likely to take over 72 hours to review.")
            else:
                st.success("Lower estimated risk: this PR is less likely to take over 72 hours to review.")
        st.progress(int(round(probability * 100)))
        st.caption(
            f"Model: {prediction_model}. This is an educational estimate from synthetic data, "
            "not a guarantee or a live GitHub prediction."
        )
