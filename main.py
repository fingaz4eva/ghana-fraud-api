
from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("fraud_model.pkl")

@app.post("/score")
async def score_transaction(transaction: dict, request: Request):
    df = pd.DataFrame([transaction])
    fraud_prob = model.predict_proba(df)[0][1]
    if fraud_prob < 0.3:
        risk = "LOW"
    elif fraud_prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    
    client_host = request.client.host
    print(f"Incoming transaction from {client_host}: {transaction} -> fraud_score={fraud_prob:.2f}, risk={risk}")
    
    return {"fraud_score": float(fraud_prob), "risk_level": risk}
