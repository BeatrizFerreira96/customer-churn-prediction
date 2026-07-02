from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("src/churn_model.pkl")

from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("templates/index.html")


from pydantic import BaseModel

class CustomerInput(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Internet_Service: str
    Contract: str
    Payment_Method: str
    Monthly_Charges: float
    
import pandas as pd

@app.post("/predict")
def predict(customer: CustomerInput):

    input_data = pd.DataFrame([{
        "Gender": customer.Gender,
        "Senior Citizen": customer.Senior_Citizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "Tenure Months": customer.Tenure_Months,
        "Internet Service": customer.Internet_Service,
        "Contract": customer.Contract,
        "Payment Method": customer.Payment_Method,
        "Monthly Charges": customer.Monthly_Charges
    }])

    pred = model.predict(input_data)[0]

    prob = model.predict_proba(input_data)[0][1]

    confidence = max(prob, 1 - prob)

    return {
        "prediction": "churn" if pred == 1 else "stay",
        "churn_probability": round(float(prob), 3),
        "confidence": round(float(confidence), 3)
    }