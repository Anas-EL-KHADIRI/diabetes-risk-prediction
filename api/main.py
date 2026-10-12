import os
import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI
from pydantic import BaseModel

FEATURES = [
    "Glucose",
    "Insulin",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction"
]

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

model_uri = "models:/diabetes-risk-classifier@champion"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Diabetes Risk Prediction API")


class PatientData(BaseModel):
    Glucose: float
    Insulin: float
    BMI: float
    Age: float
    DiabetesPedigreeFunction: float


@app.get("/")
def home():
    return {"message": "Diabetes Risk Prediction API is running"}


@app.post("/predict")
def predict(data: PatientData):
    patient_df = pd.DataFrame([data.model_dump()])[FEATURES].astype("float64")
    prediction = model.predict(patient_df)

    return {
        "prediction": str(prediction[0])
    }