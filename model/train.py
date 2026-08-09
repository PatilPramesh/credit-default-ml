"""
Training script for Machine Learning Assignment 2.

Dataset : Default of Credit Card Clients (UCI)
Task    : Binary classification - predict whether a client will default
          on their credit card payment next month.

Stage 3: Adds K-Nearest Neighbors and Naive Bayes classifiers alongside
Logistic Regression and Decision Tree.
"""

import os

import joblib
import pandas as pd
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    for name, pipeline in build_pipelines().items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        print(f"  {metrics}")

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(MODEL_DIR, filename))
        print(f"  Saved -> model/{filename}")


if __name__ == "__main__":
    main()
