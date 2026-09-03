"""
train_model.py
---------------
One-time, offline script. Not part of the API.

What it does:
1. Builds a small synthetic loan-approval dataset using the same 5 features
   the API accepts (income, loan_amount, credit_history, employment_years,
   existing_debt). Synthetic data is used so this script runs anywhere with
   no external CSV download / no internet dependency, and the approval rule
   is transparent (good for explaining the model in an interview).
2. Trains a LogisticRegression classifier.
3. Saves the trained model to model.joblib.

Run it with:
    python model/train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

RANDOM_SEED = 42
N_SAMPLES = 2000


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Create a synthetic-but-plausible loan applicant dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    income = rng.normal(60000, 20000, n_samples).clip(10000, 200000)
    loan_amount = rng.normal(18000, 9000, n_samples).clip(1000, 60000)
    credit_history = rng.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    employment_years = rng.exponential(4, n_samples).clip(0, 40)
    existing_debt = rng.normal(9000, 6000, n_samples).clip(0, 50000)

    # A simple, human-readable "ground truth" rule (plus noise) drives the
    # label. This keeps the relationship between features and outcome
    # something you can explain feature-by-feature:
    #   - higher income                -> more likely approved
    #   - higher loan_amount           -> less likely approved
    #   - good credit_history          -> strongly more likely approved
    #   - more employment_years        -> more likely approved
    #   - higher existing_debt         -> less likely approved
    score = (
        0.00004 * income
        - 0.00006 * loan_amount
        + 1.8 * credit_history
        + 0.08 * employment_years
        - 0.00007 * existing_debt
        - 1.5  # bias term so approvals aren't trivially all 1s
    )
    noise = rng.normal(0, 0.6, n_samples)
    probability = 1 / (1 + np.exp(-(score + noise)))
    approved = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "income": income,
            "loan_amount": loan_amount,
            "credit_history": credit_history,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
            "approved": approved,
        }
    )


def main():
    df = generate_dataset()

    features = [
        "income",
        "loan_amount",
        "credit_history",
        "employment_years",
        "existing_debt",
    ]
    X = df[features]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Quick sanity-check metrics printed to console.
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")

    print("\nFeature coefficients (higher = pushes toward approval):")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"  {feature:<18} {coef:+.5f}")

    joblib.dump(model, "model/model.joblib")
    print("\nSaved trained model to model/model.joblib")


if __name__ == "__main__":
    main()
