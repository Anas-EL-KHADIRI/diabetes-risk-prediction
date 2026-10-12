import os
import pandas as pd
import mlflow
import mlflow.pyfunc

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

model_uri = "models:/diabetes-risk-classifier@champion"
model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")

input_data = pd.DataFrame([{
    "Glucose": 170.0,
    "Insulin": 150.0,
    "BMI": 30.0,
    "Age": 55.0,
    "DiabetesPedigreeFunction": 0.9
}])

prediction = model.predict(input_data)
print("Prediction:", prediction[0])