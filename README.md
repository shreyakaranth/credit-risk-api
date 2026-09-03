# Loan/Credit Risk Predictor API

A small, stateless backend API that wraps a trained classifier. Given an
applicant's financial details, it returns an approval probability.

No database — this is a pure prediction service: load model → predict → return.

## Folder structure

```
credit-risk-api/
├── main.py              # FastAPI app + /predict, /health
├── schemas.py            # Pydantic request/response models
├── model/
│   ├── train_model.py    # one-time script: trains + saves model
│   └── model.joblib       # saved trained model
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model (run once, offline)

```bash
python model/train_model.py
```

This generates a small synthetic loan-approval dataset (2,000 rows) using
the same 5 features the API accepts, trains a `LogisticRegression`
classifier, prints accuracy/AUC + feature coefficients, and saves the
result to `model/model.joblib`.

Only 5 features are used on purpose — it keeps the model something you can
explain feature-by-feature (e.g. "higher existing debt lowers approval
odds") instead of fumbling through 20+ columns.

## 2. Run the API

```bash
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.
Interactive docs: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /health`

Confirms the service and model loaded correctly.

```json
{"status": "ok"}
```

### `POST /predict`

Request body:

```json
{
  "income": 65000,
  "loan_amount": 20000,
  "credit_history": 1,
  "employment_years": 4.5,
  "existing_debt": 8000
}
```

Response:

```json
{"approved_probability": 0.82}
```

## Example curl call

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "income": 65000,
    "loan_amount": 20000,
    "credit_history": 1,
    "employment_years": 4.5,
    "existing_debt": 8000
  }'
```

## Model features

| Feature             | Meaning                              | Effect on approval        |
|----------------------|----------------------------------------|-----------------------------|
| `income`            | Applicant's annual income              | Higher → more likely approved |
| `loan_amount`        | Requested loan amount                  | Higher → less likely approved |
| `credit_history`     | 1 = good history, 0 = poor/none        | 1 → strongly more likely approved |
| `employment_years`   | Years in current employment            | Higher → more likely approved |
| `existing_debt`      | Applicant's existing outstanding debt  | Higher → less likely approved |
