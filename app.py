.import streamlit as st
import numpy as np
import joblib

# === Load Model and Scaler ===
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

# === Streamlit App ===
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Dropout Prediction App")
st.write("Enter student details below to predict the likelihood of dropout.")

# === Input Fields ===
adm_grade = st.number_input("Curricular units 1st sem (grade)", min_value=0.0, max_value=20.0, value=10.0)
daytime = st.selectbox("Daytime/Evening attendance", ["Daytime", "Evening"])
scholarship = st.selectbox("Scholarship holder", ["Yes", "No"])
age_enroll = st.slider("Age at enrollment", min_value=15, max_value=60, value=20)
gdp_val = st.number_input("GDP value", min_value=0.0, max_value=100000.0, value=20000.0)

# Convert categorical inputs
daytime_val = 1 if daytime == "Daytime" else 0
scholar_val = 1 if scholarship == "Yes" else 0

# === Predict Button ===
if st.button("🔍 Predict Dropout"):
    data = np.array([[adm_grade, daytime_val, scholar_val, age_enroll, gdp_val]])
    data_scaled = scaler.transform(data)

    prob = model.predict_proba(data_scaled)[0][1]
    result = "Likely to Dropout 😔" if prob >= 0.5 else "Likely to Continue 🎓"

    st.subheader("📊 Prediction Result")
    st.metric(label="Dropout Probability", value=f"{prob:.2%}")
    st.success(result if prob < 0.5 else result, icon="🎓" if prob < 0.5 else "⚠️")
