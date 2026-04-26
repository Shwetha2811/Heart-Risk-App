# ================= 1. IMPORT =================
import streamlit as st
import numpy as np
import joblib

# ================= 2. LOAD FILES =================
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# ================= 3. UI =================
st.set_page_config(page_title="Heart Risk Predictor", layout="centered")

st.title("❤️ Heart Attack Risk Predictor")
st.write("Enter patient details to predict heart disease risk")

# ================= 4. INPUT FIELDS =================
age = st.number_input("Age", 1, 120, step=1)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])

resting_bp = st.number_input("Resting Blood Pressure", 50, 250)

cholesterol = st.number_input("Cholesterol", 0, 600)

fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

max_hr = st.number_input("Max Heart Rate", 60, 220)

exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])

oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ================= 5. ENCODING =================
try:
    sex = encoders['Sex'].transform([sex])[0]
    chest_pain = encoders['ChestPainType'].transform([chest_pain])[0]
    rest_ecg = encoders['RestingECG'].transform([rest_ecg])[0]
    exercise_angina = encoders['ExerciseAngina'].transform([exercise_angina])[0]
    st_slope = encoders['ST_Slope'].transform([st_slope])[0]
except Exception as e:
    st.error("Encoding error. Make sure encoders.pkl matches training.")
    st.stop()

# ================= 6. PREDICTION =================
if st.button("Predict"):

    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, rest_ecg, max_hr,
                            exercise_angina, oldpeak, st_slope]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ================= 7. OUTPUT =================
    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {probability:.2f}")
