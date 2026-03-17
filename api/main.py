from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using a trained ML pipeline",
    version="1.0"
)

# -----------n nm------------------
# Load trained pipeline + threshold
# -----------------------------
model = joblib.load("models/model.pkl")

with open("models/threshold.json") as f:
    threshold = json.load(f)["threshold"]

# -----------------------------
# Input Schema using Pydantic
# -----------------------------
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame([customer.model_dump()])

        # 🔥 Always use probability
        probability = model.predict_proba(input_df)[0][1]

        # 🔥 Apply optimized threshold
        prediction = 1 if probability >= threshold else 0

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 3),
            "threshold_used": round(float(threshold), 3)
        }

    except Exception as e:
        return {"error": str(e)}