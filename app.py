import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Path to your final trained model (pipeline)
MODEL_PATH = r"C:\Users\pc\Documents\Basel BME\Programming\Python\SPRINTS Heart Disease Project\python\models\final_model.pkl"

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient data below to predict the likelihood of heart disease.")

# Load final model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file not found! Please make sure final_model.pkl exists.")
    st.stop()

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Column names must match training!
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Collect features into a DataFrame
features = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]], columns=columns)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = None

    if prediction == 1:
        if probability is not None:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {probability:.2f})")
        else:
            st.error("⚠️ High Risk of Heart Disease")
    else:
        if probability is not None:
            st.success(f"✅ Low Risk of Heart Disease (Confidence: {1 - probability:.2f})")
        else:
            st.success("✅ Low Risk of Heart Disease")
