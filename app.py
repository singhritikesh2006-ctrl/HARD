import streamlit as st
import pandas as pd
import joblib
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Risk App", layout="centered")

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background:
            linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.9)),
            url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Text */
    body, p, div {{
        color: white;
    }}

    /* Glass Cards */
    .card {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(15px);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(0,247,255,0.2);
        transition: 0.3s;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }}

    .card:hover {{
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 0 25px rgba(0,247,255,0.6);
    }}

    /* Title */
    .main-title {{
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        color: #00f7ff;
        text-shadow: 0 0 20px rgba(0,247,255,0.8);
    }}

    .subtitle {{
        text-align: center;
        font-size: 22px;
        color: #9edfff;
        margin-bottom: 40px;
    }}

    /* Button */
    div.stButton > button {{
        background: linear-gradient(90deg, #00f7ff, #0051ff);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px;
        border: none;
        transition: 0.3s;
        box-shadow: 0 0 15px rgba(0,247,255,0.6);
    }}

    div.stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0,247,255,0.9);
    }}

    </style>
    """, unsafe_allow_html=True)

# ---------------- APPLY BACKGROUND ----------------
set_bg("heart.png")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- LOAD MODEL ----------------
rf2 = joblib.load("rf_model_dataset2.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":

    st.markdown('<div class="main-title">❤️ Heart Risk AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict • Prevent • Protect</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">🧠<br><b>AI Model</b><br>Accurate prediction</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">⚡<br><b>Instant Result</b><br>Fast analysis</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">💡<br><b>Smart Advice</b><br>Health guidance</div>', unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

# ---------------- FORM PAGE ----------------
elif st.session_state.page == "form":

    col_back, col_reset = st.columns(2)

    with col_back:
        if st.button("🔙 Back"):
            st.session_state.page = "home"
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset"):
            st.rerun()

    st.markdown("## ❤️ Enter Your Health Details")

    name = st.text_input("👤 Patient Name")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", 1, 100)
        bp = st.number_input("🩺 Blood Pressure", 80, 200)
        cholesterol = st.number_input("🧪 Cholesterol", 100, 400)
        bmi = st.number_input("⚖️ BMI", 10.0, 40.0)

    with col2:
        heart_rate = st.number_input("❤️ Heart Rate", 40, 200)
        diabetes = st.selectbox("🩸 Diabetes", ["No", "Yes"])
        sex = st.selectbox("👤 Gender", ["Male", "Female"])
        exercise = st.selectbox("🏃 Exercise Level", ["Low", "Moderate", "High"])

    cp_level = st.selectbox("💢 Chest Pain", ["No Pain", "Mild", "Moderate", "Severe"])

    if st.button("🔍 Predict Heart Risk"):

        sex_val = 1 if sex == "Male" else 0
        diabetes_val = 1 if diabetes == "Yes" else 0
        cp = {"No Pain":0,"Mild":1,"Moderate":2,"Severe":3}[cp_level]
        exercise_level = {"Low":0,"Moderate":1,"High":2}[exercise]

        trestbps = bp
        chol = cholesterol
        fbs = diabetes_val
        restecg = 1 if bp > 140 else 0

        thalach = heart_rate + (5 if exercise_level==2 else -5 if exercise_level==0 else 0)
        exang = 1 if (exercise_level == 0 and heart_rate > 140) or cp >= 2 else 0
        oldpeak = round((bmi / 10) + (exercise_level * 0.5), 2)
        slope = 2 if heart_rate > 140 else 1

        risk_count = sum([bp>140, cholesterol>240, diabetes_val==1, exercise_level==0])
        ca = min(risk_count, 3)
        thal = 3 if cholesterol>260 else 2 if cholesterol>200 else 1

        user_input = pd.DataFrame([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]],
            columns=['age','sex','chest_pain_type','resting_blood_pressure',
                     'cholestoral','fasting_blood_sugar','rest_ecg',
                     'Max_heart_rate','exercise_induced_angina',
                     'oldpeak','slope','vessels_colored_by_flourosopy','thalassemia'])

        user_scaled = scaler.transform(user_input)
        risk_prob = rf2.predict_proba(user_scaled)[0][1]
        risk_percent = risk_prob * 100

        if risk_percent < 30:
            category, color = "Low Risk", "green"
        elif risk_percent < 60:
            category, color = "Medium Risk", "orange"
        elif risk_percent < 80:
            category, color = "High Risk", "red"
        else:
            category, color = "Critical", "darkred"

        st.markdown(f"## ❤️ Risk Score: {risk_percent:.2f}%")
        st.progress(int(risk_percent))
        st.markdown(f"## ⚠️ Category: :{color}[{category}]")

        st.markdown("### 🩺 Recommended Actions")

        if category == "Low Risk":
            st.success("Maintain healthy lifestyle 👍")
        elif category == "Medium Risk":
            st.warning("Improve lifestyle + monitor health ⚠️")
        elif category == "High Risk":
            st.error("Consult doctor immediately ❗")
        else:
            st.error("Emergency 🚨 Seek medical help")
