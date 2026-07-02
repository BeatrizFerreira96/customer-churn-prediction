from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import joblib
import shap

app = FastAPI()

model = joblib.load("src/churn_model.pkl")
feature_names = joblib.load(
    "src/feature_names.pkl"
)

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]


background = pd.DataFrame(
    [[0] * len(feature_names)],
    columns=feature_names
)

explainer = shap.LinearExplainer(
    classifier,
    background
)


@app.get("/")
def home():
    return FileResponse("templates/index.html")




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
    
    feature_mapping = {
    "num__Tenure Months": "Customer Tenure",
    "num__Monthly Charges": "Monthly Charges",

    "cat__Dependents_No": "No Dependents",
    "cat__Dependents_Yes": "Has Dependents",

    "cat__Partner_No": "No Partner",
    "cat__Partner_Yes": "Has Partner",

    "cat__Senior Citizen_No": "Not Senior Citizen",
    "cat__Senior Citizen_Yes": "Senior Citizen",

    "cat__Contract_Month-to-month":
        "Month-to-month Contract",

    "cat__Contract_One year":
        "One-year Contract",

    "cat__Contract_Two year":
        "Two-year Contract",

    "cat__Internet Service_Fiber optic":
        "Fiber Optic Internet",

    "cat__Internet Service_DSL":
        "DSL Internet",

    "cat__Internet Service_No":
        "No Internet Service",

    "cat__Payment Method_Electronic check":
        "Electronic Check",

    "cat__Payment Method_Credit card (automatic)":
        "Automatic Credit Card",

    "cat__Payment Method_Bank transfer (automatic)":
        "Automatic Bank Transfer",

    "cat__Payment Method_Mailed check":
        "Mailed Check"
    }
    
    
    pred = model.predict(input_data)[0]

    prob = model.predict_proba(input_data)[0][1]

    confidence = max(prob, 1 - prob)
    
    
    processed = preprocessor.transform(input_data)

    shap_values = explainer(processed)

    customer_shap = shap_values.values[0]
    
    shap_df = pd.DataFrame({
    "feature": feature_names,
    "shap": customer_shap
    })
    
    shap_df["feature"] = (
    shap_df["feature"]
    .map(feature_mapping)
    .fillna(shap_df["feature"])
    )
    # Create abs_shap FIRST
    shap_df["abs_shap"] = abs(shap_df["shap"])

# Then filter
    shap_df = shap_df[
    shap_df["abs_shap"] > 0.05
]

# Then create positive/negative tables
    top_positive = (
    shap_df
    .sort_values("shap", ascending=False)
    .head(3)
)

    top_negative = (
    shap_df
    .sort_values("shap")
    .head(3)
)

# Then create chart data
    shap_chart = (
    shap_df
    .sort_values("abs_shap", ascending=False)
    .head(6)
)


    
    top_positive["shap"] = top_positive["shap"].round(3)
    top_negative["shap"] = top_negative["shap"].round(3)
    
    return {
        "prediction": "churn" if pred == 1 else "stay",
        "churn_probability": round(float(prob), 3),
        "confidence": round(float(confidence), 3),
        "top_positive": top_positive.to_dict("records"),
        "top_negative": top_negative.to_dict("records"),
        "shap_chart": shap_chart.to_dict("records")
    }