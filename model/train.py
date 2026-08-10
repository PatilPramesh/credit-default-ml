"""
Training script for Machine Learning Assignment 2.

Dataset : Default of Credit Card Clients (UCI)
Task    : Binary classification - predict whether a client will default
          on their credit card payment next month.

This script:
1. Loads and cleans the raw dataset.
2. Splits it into train/test sets (stratified, 80/20).
3. Trains 5 classification models (each wrapped in a preprocessing pipeline).
4. Evaluates every model on the held-out test set using 6 metrics.
5. Saves the trained pipelines (model/*.joblib) so the Streamlit app can
   load them directly at inference time.
6. Saves the test split as test_data.csv (features + true label) for the
   assignment submission and for uploading into the Streamlit app.
7. Saves a metrics comparison table as CSV/JSON for the README.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "default_of_credit_card_clients.xls")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_CSV_PATH = os.path.join(BASE_DIR, "model", "metrics_comparison.csv")
METRICS_JSON_PATH = os.path.join(BASE_DIR, "model", "metrics_comparison.json")

TARGET_COL = "default"
RANDOM_STATE = 42


def load_data():
    df = pd.read_excel(RAW_DATA_PATH, header=1)
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": TARGET_COL})

    # A few EDUCATION/MARRIAGE codes fall outside the documented categories
    # (0, 5, 6 for EDUCATION and 0 for MARRIAGE). Fold them into "others".
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def build_pipelines():
    """Each model gets its own pipeline so the app can call .predict directly
    on raw feature columns without worrying about separate scalers."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("clf", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)),
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"Dataset shape: {df.shape} | Features: {X.shape[1]} | Rows: {df.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    pipelines = build_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        print(f"  {metrics}")

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(MODEL_DIR, filename))
        print(f"  Saved -> model/{filename}")

    # Save comparison table
    results_df = pd.DataFrame(results).T
    results_df.index.name = "ML Model Name"
    results_df = results_df.round(4)
    results_df.to_csv(METRICS_CSV_PATH)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Comparison Table ===")
    print(results_df.to_string())

    # Save test split (features + true label) for submission + Streamlit app
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved test data -> {TEST_DATA_PATH} ({test_df.shape[0]} rows)")


if __name__ == "__main__":
    main()
