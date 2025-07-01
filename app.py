# Import required libraries for the Streamlit web application
import streamlit as st
import pandas as pd
#import joblib
from sklearn.pipeline import Pipeline

# Load the pre-trained machine learning model from the saved pickle file
# This model was trained on healthcare data to predict hospital charges
model = pd.read_pickle("hospital_charge_model.pkl")

# Configure the Streamlit page settings
st.set_page_config(page_title="Hospital Charge Predictor", layout="centered")

# Custom CSS to change fonts and apply color theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Roboto:wght@300;400;500;700&display=swap');

    /* Main title styling with Bebas Neue, bold, white */
    .main .block-container h1 {
        font-family: 'Bebas Neue', cursive, sans-serif;
        font-size: 3rem;
        letter-spacing: 2px;
        color: #fff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 1rem;
    }

    /* All text bold and white */
    .main .block-container, .main .block-container p, .main .block-container span, .main .block-container label, .main .block-container h2, .main .block-container h3, .main .block-container h4, .main .block-container h5, .main .block-container h6 {
        font-family: 'Roboto', sans-serif;
        color: #fff !important;
        font-weight: bold !important;
    }

    /* Custom slider color to #A56ABD */
    /* For webkit browsers (Chrome, Edge, Safari) */
    input[type="range"]::-webkit-slider-thumb {
        background: #A56ABD;
        border: 2px solid #fff;
    }
    input[type="range"]::-webkit-slider-runnable-track {
        background: #A56ABD;
    }
    /* For Firefox */
    input[type="range"]::-moz-range-thumb {
        background: #A56ABD;
        border: 2px solid #fff;
    }
    input[type="range"]::-moz-range-track {
        background: #A56ABD;
    }
    /* For IE */
    input[type="range"]::-ms-thumb {
        background: #A56ABD;
        border: 2px solid #fff;
    }
    input[type="range"]::-ms-fill-lower {
        background: #A56ABD;
    }
    input[type="range"]::-ms-fill-upper {
        background: #A56ABD;
    }
    /* Slider value color */
    .stSlider > div[data-testid="stSlider"] span {
        color: #A56ABD !important;
    }
</style>
""", unsafe_allow_html=True)

# Display the main title and description
st.title("Hospital Treatment Charge Predictor")
st.markdown("Fill out the patient info below to estimate total hospital charges:")

# --- Patient Input Form Section ---
# Create interactive form elements for user input

# Demographic information
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])

# Medical information
diagnosis_code = st.selectbox("Diagnosis Code", ["I10", "E11", "J45", "C50", "K21"])  # ICD-10 diagnosis codes
procedure_code = st.selectbox("Procedure Code", ["0U5B7ZZ", "3E0P3MZ", "0WQF0ZZ"])  # ICD-10-PCS procedure codes
treatment_type = st.selectbox("Treatment Type", ["Surgery", "Medical Therapy", "Observation", "Emergency Care", "Rehabilitation"])

# Financial information
insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private Insurance", "Uninsured"])

# Numerical inputs with sliders for better user experience
age = st.slider("Age", 18, 90, 45)  # Range: 18-90, default: 45
length_of_stay = st.slider("Length of Stay (days)", 1, 15, 3)  # Range: 1-15 days, default: 3

# Prediction Button and Logic
if st.button("Predict Hospital Charges"):
    # Create a DataFrame with the user input data
    # This matches the format expected by the trained model
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
    
    # Use the loaded model to make a prediction on the new patient data
    predicted = model.predict(new_patient)
    
    # Display the prediction result with proper formatting
    # Round to 2 decimal places and add thousand separators for readability
    st.success(f"Estimated Hospital Charges: **${round(predicted[0], 2):,.2f}**")
