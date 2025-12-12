import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.markdown("""
    <h1 style="text-align:center; color:#2C3E50;">🎓 Placement Prediction App</h1>
    <p style="text-align:center; font-size:18px; color:#34495E;">
        Enter student details below to predict whether they will get placed.
    </p>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
scaler = joblib.load("standard_scale_placement.pkl")
model = joblib.load("log_reg_placement.pkl")

EXPECTED_FEATURES = scaler.n_features_in_   # Should be 5

# -------------------- FEATURE NAMES --------------------
FEATURE_NAMES = [
    "Age",
    "CGPA Score",
    "Interview Score",
    "Gender (1 = Male, 0 = Female)",
    "Experience (Years)"
]

# -------------------- UI FORM --------------------
st.markdown("### 📌 Enter Student Details")

inputs = []

col1, col2 = st.columns(2)

for i, feature in enumerate(FEATURE_NAMES):
    if i % 2 == 0:
        with col1:
            val = st.number_input(f"{feature}", value=0.0, step=1.0 if "Age" in feature else 0.1)
            inputs.append(val)
    else:
        with col2:
            val = st.number_input(f"{feature}", value=0.0, step=1.0 if "Gender" in feature else 0.1)
            inputs.append(val)

# -------------------- PREDICT BUTTON --------------------
st.markdown("---")

if st.button("🔮 Predict Placement", use_container_width=True):
    arr = np.array([inputs])

    # Validate feature count
    if arr.shape[1] != EXPECTED_FEATURES:
        st.error(f"Expected {EXPECTED_FEATURES} features but got {arr.shape[1]}")
    else:
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]

        # -------------------- RESULT CARD --------------------
        if pred == 1:
            st.markdown("""
                <div style="background-color:#D4EFDF; padding:20px; border-radius:10px; text-align:center;">
                    <h2 style="color:#1D8348;">🎉 Prediction: Student is likely to get PLACED!</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color:#FADBD8; padding:20px; border-radius:10px; text-align:center;">
                    <h2 style="color:#922B21;">❌ Prediction: Student may NOT get placed.</h2>
                </div>
            """, unsafe_allow_html=True)

        # Input summary
        st.markdown("### 📊 Entered Values")
        st.write({FEATURE_NAMES[i]: inputs[i] for i in range(len(FEATURE_NAMES))})

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
