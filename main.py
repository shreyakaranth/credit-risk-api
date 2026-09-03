"""
main.py
-------
FastAPI app for the Loan/Credit Risk Predictor API.

Stateless prediction service:
- No database.
- Model is trained offline (see model/train_model.py) and loaded once here.
- Two endpoints: /health and /predict.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import LoanApplication, PredictionResponse, HealthResponse

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"

app = FastAPI(
    title="Loan/Credit Risk Predictor API",
    description="Given an applicant's financial details, returns an approval probability.",
    version="1.0.0",
)

# Loaded once at startup, reused for every request.
model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python model/train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
def health():
    """Proves the service is up and the model loaded correctly."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """Takes applicant features, returns an approval probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Keep column order identical to training time.
    input_df = pd.DataFrame(
        [
            {
                "income": application.income,
                "loan_amount": application.loan_amount,
                "credit_history": application.credit_history,
                "employment_years": application.employment_years,
                "existing_debt": application.existing_debt,
            }
        ]
    )

    probability = model.predict_proba(input_df)[0][1]

    return {"approved_probability": round(float(probability), 4)}
