import streamlit as st
import numpy as np
import joblib

st.title("Placement Predictor App")

scaler = joblib.load("standard_scale_placement.pkl")
model = joblib.load("log_reg_placement.pkl")

x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")
x3 = st.number_input("Feature 3")
x4 = st.number_input("Feature 4")

if st.button("Predict"):
    arr = np.array([[x1, x2, x3, x4]])
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    st.success(f"Prediction = {pred}")
