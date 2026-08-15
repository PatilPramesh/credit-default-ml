"""
Smoke tests for the trained model artifacts and test dataset.

These run in CI on every push so a broken model file, a schema change in
test_data.csv, or a scikit-learn version mismatch is caught *before*
Streamlit Community Cloud auto-deploys a broken app.
"""

import os

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
TARGET_COL = "default"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@pytest.fixture(scope="module")
def test_data():
    assert os.path.exists(TEST_DATA_PATH), "test_data.csv is missing from the repo root"
    df = pd.read_csv(TEST_DATA_PATH)
    assert TARGET_COL in df.columns, f"test_data.csv must contain a '{TARGET_COL}' column"
    assert df.shape[0] > 0, "test_data.csv has no rows"
    return df


@pytest.mark.parametrize("model_name,filename", MODEL_FILES.items())
def test_model_loads_and_predicts(model_name, filename, test_data):
    path = os.path.join(MODEL_DIR, filename)
    assert os.path.exists(path), f"Missing model artifact: model/{filename}"

    model = joblib.load(path)
    X = test_data.drop(columns=[TARGET_COL])
    y_true = test_data[TARGET_COL]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    assert len(y_pred) == len(test_data)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    # Sanity thresholds - not tuned for peak performance, just to catch a
    # broken/corrupted/mismatched model artifact (e.g. random predictions
    # would fail these).
    assert acc > 0.6, f"{model_name} accuracy too low ({acc:.4f}) - possible broken model"
    assert auc > 0.6, f"{model_name} AUC too low ({auc:.4f}) - possible broken model"


def test_all_five_models_present():
    for filename in MODEL_FILES.values():
        assert os.path.exists(os.path.join(MODEL_DIR, filename)), f"Missing model/{filename}"


def test_requirements_file_exists():
    assert os.path.exists(os.path.join(BASE_DIR, "requirements.txt"))


def test_app_file_exists():
    assert os.path.exists(os.path.join(BASE_DIR, "app.py"))
