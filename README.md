# Credit Card Default Prediction — ML Assignment 2

## a. Problem Statement

Credit card issuers need to identify customers who are likely to **default on
their payment next month** so that risk can be managed proactively (e.g. credit
limit adjustments, early collections outreach). This project builds and
compares five supervised classification models that predict, from a client's
demographic details and their last six months of billing/payment history,
whether they will **default on their credit card payment next month**
(binary classification: `1` = default, `0` = no default).

## b. Dataset Description

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Domain:** Consumer credit risk (Taiwan, 2005)
- **Instances:** 30,000 clients
- **Features:** 23 (after dropping the `ID` column)
  - Demographics: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - Repayment status (last 6 months): `PAY_0`, `PAY_2`–`PAY_6`
  - Bill statement amounts (last 6 months): `BILL_AMT1`–`BILL_AMT6`
  - Previous payment amounts (last 6 months): `PAY_AMT1`–`PAY_AMT6`
- **Target column:** `default` (renamed from `default payment next month`) — `1` = default, `0` = no default
- **Class balance:** ~77.9% no-default, ~22.1% default (moderately imbalanced)
- **Train/Test split:** 80% train (24,000 rows) / 20% test (6,000 rows), stratified on the target
- **Test data file:** [`test_data.csv`](./test_data.csv) — the held-out 20% test split (features + true label), used for the Streamlit app demo and included in this repository as required.

## c. GitHub Repository Link

> **https://github.com/PatilPramesh/credit-default-ml**

Repository contains: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and the `model/` folder with training code and all 5 saved model files.

## d. Models Used

All 5 models below were trained **on the same dataset** (same train/test split) inside [`model/train.py`](./model/train.py). Each model is wrapped in a `scikit-learn` `Pipeline` (with `StandardScaler` where relevant) and saved as a `.joblib` file used directly by the Streamlit app.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8083 | 0.7079 | 0.6903 | 0.2419 | 0.3583 | 0.3276 |
| Decision Tree | 0.8157 | 0.7432 | 0.6467 | 0.3670 | 0.4683 | 0.3885 |
| kNN | 0.8060 | 0.7319 | 0.6146 | 0.3293 | 0.4289 | 0.3476 |
| Naive Bayes (Gaussian) | 0.7518 | 0.7248 | 0.4504 | 0.5539 | 0.4968 | 0.3376 |
| Random Forest (Ensemble) | 0.8170 | 0.7728 | 0.6667 | 0.3451 | 0.4548 | 0.3860 |

*(Metrics computed on the 6,000-row held-out test set; reproducible by re-running `model/train.py`. Full raw values are also saved in `model/metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.69) but by far the lowest recall (0.24) — it is conservative and only flags a client as "default" when very confident, so it misses most actual defaulters. Its linear decision boundary struggles to capture the non-linear interactions between repayment-status and bill-amount features. |
| Decision Tree | Good balance of precision and recall, giving the second-best F1 (0.47) and MCC (0.39). It captures non-linear splits (e.g. on `PAY_0`, the most recent repayment status) well, but a single tree is prone to overfitting, which is why `max_depth` was restricted during training. |
| kNN | Middle-of-the-pack performance across all metrics. Being distance-based, it is sensitive to feature scaling (handled via `StandardScaler`) and to the relatively high dimensionality (23 features), which dilutes the usefulness of raw Euclidean distance. |
| Naive Bayes (Gaussian) | Lowest accuracy (0.75) and MCC, but the **highest recall (0.55)** — it catches more true defaulters than any other model, at the cost of many more false positives. This happens because its core assumption (feature independence) is violated here, since bill amounts across months are highly correlated. |
| Random Forest (Ensemble) | **Best overall performer** — highest Accuracy (0.817) and AUC (0.773), and second-highest MCC. By averaging many decorrelated decision trees, it reduces the overfitting/variance seen in the single Decision Tree while still modeling non-linear feature interactions well. |
| **Overall Winner** | **Random Forest (Ensemble)** — it has the best AUC and Accuracy and a strong, well-balanced MCC, making it the most reliable model for this dataset. Decision Tree is a close second (best MCC) and a good lightweight alternative. Naive Bayes is the model of choice only if recall (catching defaulters) matters more than overall accuracy/precision. |

## Project Structure

```
credit-default-ml/
├── app.py                        # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv                 # held-out test set (features + true label)
├── .gitignore
├── data/
│   └── default_of_credit_card_clients.xls   # raw dataset from UCI
└── model/
    ├── train.py                  # data prep, training, evaluation, model saving (script version)
    ├── train_and_evaluate.ipynb  # same pipeline as a notebook, with EDA + charts (for BITS Lab run)
    ├── metrics_comparison.csv    # comparison table (raw values)
    ├── metrics_comparison.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## Streamlit App Features

Beyond the minimum required features (CSV upload, model dropdown, metrics display, confusion matrix), the app also includes:

- **Robust CSV handling** — validates required columns and re-orders/ignores extras (e.g. a stray `ID` column) instead of crashing on upload
- **Classification report** alongside the confusion matrix
- **ROC curve** for the selected model
- **Feature importance / coefficient chart** (Random Forest, Decision Tree, Logistic Regression)
- **Side-by-side comparison of all 5 models** on the uploaded data, with the best score per metric highlighted
- **One-click sample data download** button in the sidebar

## How to Run Locally

```bash
pip install -r requirements.txt

# (Optional) retrain all models from scratch — script or notebook, both produce identical artifacts
python model/train.py
# or open and run model/train_and_evaluate.ipynb

# Launch the Streamlit app
streamlit run app.py
```

Then upload `test_data.csv` in the app's sidebar and select a model from the dropdown.

## Live Streamlit App

> **`<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`**

### Deployment Notes

- Streamlit Community Cloud does **not** read `runtime.txt` for Python version selection — set the Python version explicitly via **Advanced settings** during deploy (Python 3.11 or 3.12 recommended) or from the app's **Settings** afterward.
- `requirements.txt` uses minimum-version pins (`>=`) rather than exact pins, so pip can resolve wheels that are actually compatible with whatever Python version Cloud provisions — this avoids the "missing/incompatible dependency" deployment failures the assignment warns about.
- Model files are committed directly to `model/` so the app loads pre-trained pipelines instantly instead of retraining on every run (faster cold start on the free tier).

## BITS Virtual Lab Execution

> Screenshot of this assignment being executed on BITS Virtual Lab is included in the submitted PDF. `model/train_and_evaluate.ipynb` was executed end-to-end with no errors and is a good candidate to run/screenshot on the lab.
