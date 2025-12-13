import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<h1 style="text-align:center; color:#2C3E50;">🎓 Placement Prediction App</h1>
<p style="text-align:center; color:#566573;">
Manual typing allowed • Validation on submit
</p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
scaler = joblib.load("standard_scale_placement.pkl")
model = joblib.load("log_reg_placement.pkl")

# ---------------- INPUT UI ----------------
st.markdown("### 📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=21, step=1)
    cgpa = st.number_input("CGPA Score (0–10)", value=7.0, step=0.1)
    gender = st.number_input("Gender (0 = Female, 1 = Male)", value=0, step=1)

with col2:
    interview_score = st.number_input("Interview Score (0–10)", value=6.0, step=0.1)
    experience = st.number_input("Experience (Years)", value=0, step=1)

# ---------------- PREDICT ----------------
st.markdown("---")

if st.button("🔮 Predict Placement", use_container_width=True):

    errors = []

    # 🔒 STRICT VALIDATION (but manual typing allowed)
    if not (0 <= cgpa <= 10):
        errors.append("❌ CGPA Score must be between 0 and 10")

    if not (0 <= interview_score <= 10):
        errors.append("❌ Interview Score must be between 0 and 10")

    if gender not in [0, 1]:
        errors.append("❌ Gender must be 0 (Female) or 1 (Male)")

    # If errors → show & stop
    if errors:
        for e in errors:
            st.error(e)
        st.warning("⚠️ Fix the errors above to get prediction")
        st.stop()

    # ---------------- MODEL PREDICTION ----------------
    input_data = np.array([[age, cgpa, interview_score, gender, experience]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    # ---------------- RESULT ----------------
    if prediction == 1:
        st.success("🎉 RESULT: STUDENT IS LIKELY TO BE PLACED")
    else:
        st.error("❌ RESULT: STUDENT MAY NOT BE PLACED")

    # ---------------- SUMMARY ----------------
    st.markdown("### 📊 Input Summary")
    st.table({
        "Feature": ["Age", "CGPA", "Interview Score", "Gender", "Experience"],
        "Value": [
            age,
            cgpa,
            interview_score,
            "Female (0)" if gender == 0 else "Male (1)",
            experience
        ]
    })

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✔ Manual typing enabled | ✔ Error shown | ✔ Prediction blocked on invalid input")
