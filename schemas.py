"""
Pydantic models for the /predict endpoint.

Kept intentionally small (5 features) so the whole request/response
contract fits on one screen and is easy to reason about in an interview.
"""

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """Incoming applicant data for a prediction request."""

    income: float = Field(..., gt=0, description="Applicant's annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    credit_history: int = Field(
        ..., ge=0, le=1, description="1 = good credit history, 0 = poor/none"
    )
    employment_years: float = Field(
        ..., ge=0, description="Years in current employment"
    )
    existing_debt: float = Field(
        ..., ge=0, description="Applicant's existing outstanding debt"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "income": 65000,
                "loan_amount": 20000,
                "credit_history": 1,
                "employment_years": 4.5,
                "existing_debt": 8000,
            }
        }


class PredictionResponse(BaseModel):
    """What /predict sends back."""

    approved_probability: float = Field(
        ..., description="Model's predicted probability of loan approval (0-1)"
    )


class HealthResponse(BaseModel):
    status: str
