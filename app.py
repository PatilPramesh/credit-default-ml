"""
Streamlit app - Machine Learning Assignment 2

Dataset : Default of Credit Card Clients (UCI)
Task    : Binary classification - will a client default on payment next month?

Features:
- Upload a test CSV (features + true label column "default")
- Pick one of 5 trained classification models from a dropdown
- View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- View confusion matrix and full classification report
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="\U0001F4B3",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_COL = "default"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

# Exact feature columns / order the models were trained on. Uploaded files
# are re-indexed against this list so extra columns (e.g. a stray ID) or a
# different column order don't break inference.
FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


@st.cache_resource
def load_model(model_name: str):
    filename = MODEL_FILES[model_name]
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


@st.cache_resource
def load_all_models():
    return {name: load_model(name) for name in MODEL_FILES}


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("\U0001F4B3 Credit Card Default Prediction")
st.markdown(
    """
This app demonstrates **5 classification models** trained to predict whether a
credit card client will **default on payment next month**, using the
[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset.

**How to use:**
1. Upload the `test_data.csv` file (or any CSV with the same columns, including the true `default` label).
2. Select a model from the dropdown.
3. View predictions, evaluation metrics, confusion matrix, ROC curve, and a side-by-side comparison of all 5 models.
"""
)

with st.sidebar:
    st.header("\u2699\ufe0f Controls")
    uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

    sample_path = os.path.join(BASE_DIR, "test_data.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                "\U0001F4E5 Download sample test_data.csv",
                data=f,
                file_name="test_data.csv",
                mime="text/csv",
            )

    model_name = st.selectbox("Select a classification model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown(
        "**Models available:**\n"
        "- Logistic Regression\n"
        "- Decision Tree\n"
        "- kNN\n"
        "- Naive Bayes\n"
        "- Random Forest (Ensemble)"
    )

if uploaded_file is None:
    st.info("\u2b06\ufe0f Upload a CSV file from the sidebar to get started (use the provided `test_data.csv`).")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.subheader("\U0001F4C4 Uploaded Data Preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

has_target = TARGET_COL in data.columns

missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
if missing_cols:
    st.error(
        "The uploaded CSV is missing required feature column(s): "
        f"`{', '.join(missing_cols)}`.\n\n"
        f"Expected columns: `{', '.join(FEATURE_COLUMNS)}`"
        + (f" (+ optional `{TARGET_COL}`)" if not has_target else "")
    )
    st.stop()

# Re-index to the exact training column order; ignores extra columns
# (e.g. a stray ID column) and tolerates any column ordering in the upload.
X = data[FEATURE_COLUMNS].copy()
if has_target:
    y_true = data[TARGET_COL]

if not has_target:
    st.warning(
        f"No `{TARGET_COL}` column found in the uploaded file. "
        "Predictions will be shown, but evaluation metrics require the true label column."
    )

try:
    model = load_model(model_name)
except Exception as e:
    st.error(f"Failed to load model '{model_name}': {e}")
    st.stop()

try:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Prediction failed.\n\nError: {e}")
    st.stop()

st.subheader(f"\U0001F52E Predictions using {model_name}")
result_df = X.copy()
result_df["Predicted_Default"] = y_pred
result_df["Default_Probability"] = y_proba.round(4)
if has_target:
    result_df[TARGET_COL] = y_true.values
st.dataframe(result_df.head(20), use_container_width=True)

if has_target:
    st.subheader("\U0001F4CA Evaluation Metrics")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    st.subheader("\U0001F9EE Confusion Matrix & Classification Report")
    col_left, col_right = st.columns(2)

    with col_left:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig)

    with col_right:
        report = classification_report(
            y_true, y_pred, target_names=["No Default", "Default"], output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    col_roc, col_imp = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curve**")
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax_roc, name=model_name)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax_roc.set_title(f"ROC Curve - {model_name}")
        ax_roc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_roc)

    with col_imp:
        clf = model.named_steps.get("clf", model)
        if hasattr(clf, "feature_importances_"):
            st.markdown("**Feature Importance**")
            importance = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
            importance = importance.sort_values(ascending=True).tail(10)
            fig_imp, ax_imp = plt.subplots(figsize=(4, 3.5))
            importance.plot.barh(ax=ax_imp, color="teal")
            ax_imp.set_title(f"Top 10 Features - {model_name}")
            st.pyplot(fig_imp)
        elif hasattr(clf, "coef_"):
            st.markdown("**Feature Coefficients (importance proxy)**")
            importance = pd.Series(clf.coef_[0], index=FEATURE_COLUMNS)
            importance = importance.reindex(importance.abs().sort_values().tail(10).index)
            fig_imp, ax_imp = plt.subplots(figsize=(4, 3.5))
            importance.plot.barh(ax=ax_imp, color="teal")
            ax_imp.set_title(f"Top 10 Feature Coefficients - {model_name}")
            st.pyplot(fig_imp)
        else:
            st.info("Feature importance is not directly available for this model (e.g. kNN, Naive Bayes).")

    st.markdown("---")
    st.subheader("\U0001F3C6 Compare All 5 Models on This Dataset")
    st.caption("Runs every trained model on the uploaded data so you can see how they stack up side by side.")

    all_models = load_all_models()
    comparison_rows = {}
    for name, mdl in all_models.items():
        pred = mdl.predict(X)
        proba = mdl.predict_proba(X)[:, 1]
        comparison_rows[name] = compute_metrics(y_true, pred, proba)

    comparison_df = pd.DataFrame(comparison_rows).T.round(4)
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, color="lightgreen"),
        use_container_width=True,
    )

    fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
    comparison_df[["Accuracy", "AUC", "F1 Score", "MCC"]].plot.bar(ax=ax_cmp)
    ax_cmp.set_title("Model Comparison on Uploaded Data")
    ax_cmp.set_ylabel("Score")
    ax_cmp.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig_cmp)
else:
    st.info("Upload a file that includes the true `default` column to see metrics and the confusion matrix.")

st.markdown("---")
st.caption(
    "Machine Learning Assignment 2 | Models: Logistic Regression, Decision Tree, "
    "kNN, Naive Bayes, Random Forest (Ensemble)"
)
