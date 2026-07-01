from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("src/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn API running"}


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
    Total_Charges: float
    
@app.post("/predict")
def predict(customer: CustomerInput):

    return {
        "message": "Prediction endpoint working"
    }