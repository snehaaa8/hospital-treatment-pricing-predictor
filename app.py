import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Load your trained pipeline (you'll need to save it first!)
model = joblib.load("hospital_charge_model.pkl")

st.set_page_config(page_title="Hospital Charge Predictor", layout="centered")
st.title("Hospital Treatment Charge Predictor")

st.markdown("Fill out the patient info below to estimate total hospital charges:")

# --- Patient Input Form ---
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
diagnosis_code = st.selectbox("Diagnosis Code", ["I10", "E11", "J45", "C50", "K21"])  # Modify as per your dataset
procedure_code = st.selectbox("Procedure Code", ["0U5B7ZZ", "3E0P3MZ", "0WQF0ZZ"])
treatment_type = st.selectbox("Treatment Type", ["Surgery", "Medical Therapy", "Observation", "Emergency Care", "Rehabilitation"])
insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private Insurance", "Uninsured"])
age = st.slider("Age", 18, 90, 45)
length_of_stay = st.slider("Length of Stay (days)", 1, 15, 3)

# Predict Button
if st.button("Predict Hospital Charges"):
    new_patient = pd.DataFrame([{
        "gender": gender,
        "race": race,
        "diagnosis_code": diagnosis_code,
        "procedure_code": procedure_code,
        "treatment_type": treatment_type,
        "insurance_type": insurance_type,
        "age": age,
        "length_of_stay": length_of_stay
    }])
    
    predicted = model.predict(new_patient)
    st.success(f"Estimated Hospital Charges: **${round(predicted[0], 2):,.2f}**")
