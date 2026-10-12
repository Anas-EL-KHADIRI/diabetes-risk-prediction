
import os
import streamlit as st
import requests

# use env var so it works in Docker too
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Diabetes Risk Prediction")
st.write("Enter the patient information below to predict diabetes risk.")

with st.form("patient_form"):
    glucose = st.number_input("Glucose", min_value=0.0, value=110.0, step=1.0)
    insulin = st.number_input("Insulin", min_value=0.0, value=170.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, value=30.0, step=0.1)
    age = st.number_input("Age", min_value=1, value=26, step=1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.4, step=0.01)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    patient_data = {
        "Glucose": float(glucose),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "Age": float(age),
        "DiabetesPedigreeFunction": float(pedigree)
    }

    try:
        response = requests.post(
            API_URL,
            json=patient_data,
            timeout=10
        )
        response.raise_for_status()

        result = response.json()["prediction"]

        st.subheader("Prediction Result")

        if result == "high risk":
            st.error("High Risk")
            st.markdown(
                "**Recommendations :** Consult a doctor quickly."
            )
        else:
            st.success("Low Risk")
            st.markdown(
                "**Recommendations :** Keep a balanced diet and regular physical activity. "
            )

    except requests.RequestException as error:
        st.error(f"Could not connect to the API: {error}")