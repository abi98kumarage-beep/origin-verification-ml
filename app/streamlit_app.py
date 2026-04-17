import streamlit as st
import numpy as np
import joblib
import os

st.title("🌍 Origin Prediction System")

# =========================
# LOAD MODEL (WITH ERROR HANDLING)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# =========================
# INPUTS
# =========================
N = st.number_input("Nitrogen", 0, 140, 50)
P = st.number_input("Phosphorus", 0, 140, 50)
K = st.number_input("Potassium", 0, 200, 50)
temperature = st.number_input("Temperature", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity", 0.0, 100.0, 60.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall", 0.0, 300.0, 100.0)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = model.predict(data)
    probs = model.predict_proba(data)

    st.write("Prediction:", pred[0])
    st.write("Confidence:", np.max(probs))