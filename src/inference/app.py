from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Create the FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0")

# Load the model when the API starts
MODEL_PATH = "models/best_churn_model.pkl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model not found. Please run training first!")
    print("   Run: python src\\training\\train.py")
    exit(1)

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# Define what data the API expects
class CustomerData(BaseModel):
    tenure_months: float
    monthly_charges: float
    num_support_tickets: float
    satisfaction_score: float
    contract_years: float

@app.get("/")
def root():
    """Home endpoint - check if API is running"""
    return {"message": "Churn Prediction API is running!", "status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerData):
    """
    Predict if a customer will churn
    """
    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        'tenure_months': customer.tenure_months,
        'monthly_charges': customer.monthly_charges,
        'num_support_tickets': customer.num_support_tickets,
        'satisfaction_score': customer.satisfaction_score,
        'contract_years': customer.contract_years
    }])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability[1], 3),
        "explanation": "Customer WILL churn" if prediction == 1 else "Customer will STAY"
    }