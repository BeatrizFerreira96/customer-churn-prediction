from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse

import io
import pandas as pd
import joblib
import shap

app = FastAPI()

models = {
    "logistic": joblib.load("src/logistic_model.pkl"),
    "random_forest": joblib.load("src/random_forest_model.pkl"),
    "xgboost": joblib.load("src/xgboost_model.pkl")
}
feature_names = joblib.load(
    "src/feature_names.pkl"
)
explainers = {}

for name, pipeline in models.items():

    classifier = pipeline.named_steps["classifier"]

    if name == "logistic":

        background = pd.DataFrame(
            [[0] * len(feature_names)],
            columns=feature_names
        )

        explainers[name] = shap.LinearExplainer(
            classifier,
            background
        )

    else:

        explainers[name] = shap.TreeExplainer(
            classifier
        )
        
        

EXPECTED_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Internet Service",
    "Contract",
    "Payment Method",
    "Monthly Charges"
]

@app.get("/")
def home():
    return FileResponse("templates/index.html")




class CustomerInput(BaseModel):
    model: str = "logistic"
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Internet_Service: str
    Contract: str
    Payment_Method: str
    Monthly_Charges: float
    
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    
    missing_cols = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
        raise HTTPException(
        status_code=400,
        detail={
            "error": "Missing columns",
            "missing": missing_cols
        }
    )

    # Check for missing values
    if df[EXPECTED_COLUMNS].isnull().any().any():

        missing = (
        df[EXPECTED_COLUMNS]
        .isnull()
        .sum()
    )

        missing = missing[missing > 0]

        raise HTTPException(
        status_code=400,
        detail={
            "error": "Missing values detected",
            "missing_values": missing.to_dict()
        }
    )
    
    processed = model.named_steps["preprocessor"].transform(df)
    

    classifier = model.named_steps["classifier"]
    predictions = classifier.predict(processed)
    probabilities = classifier.predict_proba(processed)[:, 1]
    
    
    total_customers = len(df)

    predicted_churners = int(sum(predictions))

    churn_rate = round(
    predicted_churners / total_customers * 100,
    1
    )

    avg_probability = round(
    probabilities.mean() * 100,
    1
    )
    
    df["prediction"] = [
        "Churn" if p == 1 else "No Churn"
        for p in predictions
    ]

    df["churn_probability"] = probabilities.round(4)

    output = io.StringIO()

    df.to_csv(output, index=False)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=predictions.csv"
        }
    )

@app.post("/predict")
def predict(customer: CustomerInput):

    selected_model = models[customer.model]
    
    explainer = explainers[customer.model]
    
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
    
    
    pred = selected_model.predict(input_data)[0]

    prob = selected_model.predict_proba(input_data)[0][1]

    confidence = max(prob, 1 - prob)
    
    preprocessor = selected_model.named_steps["preprocessor"]
    processed = preprocessor.transform(input_data)

    

    print(f"Using model: {customer.model}")
    print(type(selected_model))

    shap_values = explainer(processed)

    print(type(shap_values))

    if hasattr(shap_values, "values"):
        print("Values shape:", shap_values.values.shape)
    else:
        print("No .values attribute")
        
    if shap_values.values.ndim == 3:
        customer_shap = shap_values.values[0, :, 1]
    else:
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

    positive_features = shap_df[shap_df["shap"] > 0]
    negative_features = shap_df[shap_df["shap"] < 0]
# Then create positive/negative tables
    top_positive = (
    positive_features
    .sort_values("shap", ascending=False)
    .head(3)
)

    top_negative = (
    negative_features
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
        "model": customer.model,
        "prediction": "churn" if pred == 1 else "stay",
        "churn_probability": round(float(prob), 3),
        "confidence": round(float(confidence), 3),
        "top_positive": top_positive.to_dict("records"),
        "top_negative": top_negative.to_dict("records"),
        "shap_chart": shap_chart.to_dict("records")
    }
    
@app.post("/batch_summary")
async def batch_summary(file: UploadFile = File(...)):
    
    df = pd.read_csv(file.file)
    
    missing_cols = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:
       
        raise HTTPException(
        status_code=400,
        detail={
            "error": "Missing columns",
            "missing": missing_cols
        }
    )
     # Check for missing values
    if df[EXPECTED_COLUMNS].isnull().any().any():

        missing = (
        df[EXPECTED_COLUMNS]
        .isnull()
        .sum()
    )

        missing = missing[missing > 0]

        raise HTTPException(
        status_code=400,
        detail={
            "error": "Missing values detected",
            "missing_values": missing.to_dict()
        }
    )
        
    processed = model.named_steps["preprocessor"].transform(df)

    import numpy as np
    
   

# Now continue with prediction
    predictions = model.named_steps["classifier"].predict(processed)
    probabilities = model.named_steps["classifier"].predict_proba(processed)[:, 1]

    total_customers = len(df)

    predicted_churners = int(sum(predictions))

    churn_rate = round(
        predicted_churners / total_customers * 100,
        1
    )

    avg_probability = round(
        probabilities.mean() * 100,
        1
    )

    return {
        "total_customers": total_customers,
        "predicted_churners": predicted_churners,
        "churn_rate": churn_rate,
        "average_probability": avg_probability
    }