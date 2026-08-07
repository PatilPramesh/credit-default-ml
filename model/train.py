"""
Training script for Machine Learning Assignment 2.

Dataset : Default of Credit Card Clients (UCI)
Task    : Binary classification - predict whether a client will default
          on their credit card payment next month.

Stage 1: Data loading, cleaning, and train/test split.
Model training is added in subsequent commits.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "default_of_credit_card_clients.xls")

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


def main():
    print("Loading data...")
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"Dataset shape: {df.shape} | Features: {X.shape[1]} | Rows: {df.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


if __name__ == "__main__":
    main()
