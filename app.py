from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Customer Spend Production API")

# Load artifacts
model = joblib.load("rf_spend_model.pkl")
scaler = joblib.load("data_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

class PredictionInput(BaseModel):
    monthly_spend: float  
    num_transactions: int 
    promo_usage: int      
    Age: int
    Annual_Income_USD: float
    Financial_Discipline_Index: int
    month: int          

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # 1. Create a raw DataFrame
        raw_df = pd.DataFrame([data.model_dump()])
        
        # 2. Re-create the specific columns the scaler expects 
        # (The notebook scaled 8 columns: Age, Income, Purchase_Amt, Discipline, year, month, week, dayofweek)
        # Note: This is a hackathon shortcut to align with their specific notebook scaling
        dummy_data = pd.DataFrame(np.zeros((1, 8)), columns=["Age", "Annual_Income_USD", "Purchase_Amt_USD", "Financial_Discipline_Index", "year", "month", "week", "dayofweek"])
        
        dummy_data["Age"] = data.Age
        dummy_data["Annual_Income_USD"] = data.Annual_Income_USD
        dummy_data["Financial_Discipline_Index"] = data.Financial_Discipline_Index
        dummy_data["month"] = data.month
        
        # 3. Apply Scaling
        scaled_data = scaler.transform(dummy_data)
        scaled_df = pd.DataFrame(scaled_data, columns=dummy_data.columns)
        
        # 4. Final Feature Set (Monthly Spend in notebook was sum of SCALED amounts)
        # We simulate that logic here
        final_X = pd.DataFrame([[
            data.monthly_spend / 100, # Approximate scaling for the spend
            data.num_transactions,
            data.promo_usage,
            scaled_df["Age"].values[0],
            scaled_df["Annual_Income_USD"].values[0],
            scaled_df["Financial_Discipline_Index"].values[0],
            scaled_df["month"].values[0]
        ]], columns=feature_names)

        # 5. Predict
        prediction = model.predict(final_X)
        return {"predicted_next_month_spend": round(float(prediction[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
