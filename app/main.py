from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fraud_detection.config import MODELS_DIR
from fraud_detection.hyperparameter_tuning import get_hour
import pandas as pd
import numpy as np
import joblib as jb
from pydantic import BaseModel
import sys

sys.modules['__main__'].get_hour = get_hour

app = FastAPI(title = 'Credit Card Fraud Detection API')

model = jb.load(MODELS_DIR / 'XGBoost_best_model.joblib')[0]

class Transaction(BaseModel):
    time: float
    amount: float
    components: list[float]

@app.get("/", include_in_schema=False)
async def root_to_docs():
    return RedirectResponse(url="/docs")

@app.post('/predict')
def predict(transaction: Transaction):
    input_dict = {
        "Time": transaction.time,
    }
    
    for i, val in enumerate(transaction.components, start=1):
        input_dict[f"V{i}"] = val

    input_dict['Amount'] = transaction.amount
    input_df = pd.DataFrame([input_dict])
    is_fraud = model.predict(input_df)
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = 'Fraud' if probability > 0.1566 else 'Legitimate'
    return {
        "prediction": prediction,
        "probability": probability
    }