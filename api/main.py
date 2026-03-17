from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)
print("NEW VERSION DEPLOYED")
# -----------------------------
# Load model + threshold (SAFE WAY)
# -----------------------------
@app.on_event("startup")
def load_artifacts():
    global model, threshold

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")
    threshold_path = os.path.join(BASE_DIR, "..", "models", "threshold.json")

    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading threshold...")
    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    print("Startup complete")

# -----------------------------
# Input Schema
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
# Health Check
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
        input_df = pd.DataFrame([customer.model_dump()])

        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability >= threshold else 0

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 3),
            "threshold_used": round(float(threshold), 3)
        }

    except Exception as e:
        return {"error": str(e)}
