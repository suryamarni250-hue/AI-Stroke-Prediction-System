import streamlit as st
import joblib
import numpy as np
import base64
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Brain Stroke AI", layout="wide")

# ----------------------------
# BACKGROUND
# ----------------------------
def add_bg():
    with open("brain.jpg", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        h1 {{
            text-align: center;
            font-size: 55px !important;
            color: white !important;
            text-shadow: 0 0 10px black;
        }}

        label {{
            font-size: 18px !important;
            font-weight: bold !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/stroke_model.pkl")

st.markdown("<h1>🧠 Brain Stroke Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# INPUT LAYOUT
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Hypertension = st.selectbox("Hypertension (0/1)", [0,1])
    Heart_Disease = st.selectbox("Heart_Disease (0/1)", [0,1])
    Ever_Married = st.selectbox("Ever_Married", ["Yes", "No"])
    Work_Type = st.selectbox("Work_Type",
                             ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    Residence = st.selectbox("Residence", ["Urban", "Rural"])
    Average_Glucose = st.number_input("Average_Glucose")
    BMI = st.number_input("BMI")

with col2:
    Smoking_Status = st.selectbox("Smoking_Status",
                                   ["Never", "Formerly", "Currently", "Unknown"])
    Physical_Activity = st.number_input("Physical_Activity")
    Alcohol_Intake = st.number_input("Alcohol_Intake")
    Stress_Level = st.number_input("Stress_Level")
    Blood_Pressure = st.number_input("Blood_Pressure")
    Cholesterol = st.number_input("Cholesterol")
    Family_History = st.selectbox("Family_History", ["Yes", "No"])
    MRI_Result = st.number_input("MRI_Result")

# Encoding maps
sex_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
family_map = {"Yes": 1, "No": 0}
residence_map = {"Urban": 1, "Rural": 0}

work_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "Children": 3,
    "Never_worked": 4
}

smoking_map = {
    "Never": 0,
    "Formerly": 1,
    "Currently": 2,
    "Unknown": 3
}

st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1,2,1])

with c2:
    predict_button = st.button("🚀 ANALYZE RISK", use_container_width=True)

# ----------------------------
# ANALYSIS SECTION
# ----------------------------
if predict_button:

    input_data = np.array([[
        sex_map[Sex],
        Hypertension,
        Heart_Disease,
        married_map[Ever_Married],
        work_map[Work_Type],
        residence_map[Residence],
        Average_Glucose,
        BMI,
        smoking_map[Smoking_Status],
        Physical_Activity,
        Alcohol_Intake,
        Stress_Level,
        Blood_Pressure,
        Cholesterol,
        family_map[Family_History],
        MRI_Result
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = int(probability * 100)

    # ----------------------------
    # RESULT
    # ----------------------------
    st.subheader("📊 Stroke Risk Percentage")
    st.progress(risk_percent)
    st.write(f"### Risk Level: {risk_percent}%")

    result_text = "HIGH RISK OF STROKE" if prediction[0] == 1 else "LOW RISK OF STROKE"

    if prediction[0] == 1:
        st.error("⚠ HIGH RISK OF STROKE")
    else:
        st.success("✅ LOW RISK OF STROKE")

    # ----------------------------
    # GRAPH 1: Key Health Comparison
    # ----------------------------
    st.subheader("📈 Health Parameter Comparison")

    parameters = ['Glucose', 'BP', 'Cholesterol', 'Stress']
    values = [Average_Glucose, Blood_Pressure, Cholesterol, Stress_Level]
    normal = [100, 120, 200, 5]

    fig1, ax1 = plt.subplots()
    x = np.arange(len(parameters))
    ax1.bar(x - 0.2, values, 0.4, label='Patient')
    ax1.bar(x + 0.2, normal, 0.4, label='Normal')
    ax1.set_xticks(x)
    ax1.set_xticklabels(parameters)
    ax1.legend()
    st.pyplot(fig1)

    # ----------------------------
    # GRAPH 2: Risk Indicator Pie
    # ----------------------------
    st.subheader("📉 Risk Distribution")

    fig2, ax2 = plt.subplots()
    ax2.pie([risk_percent, 100-risk_percent],
            labels=['Risk', 'Safe'],
            autopct='%1.1f%%')
    st.pyplot(fig2)

    # ----------------------------
    # PDF REPORT
    # ----------------------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Brain Stroke Risk Report", styles['Heading1']))
    elements.append(Spacer(1, 0.3 * inch))

    report_data = [
        f"Sex: {Sex}",
        f"Glucose Level: {Average_Glucose}",
        f"Blood Pressure: {Blood_Pressure}",
        f"Cholesterol: {Cholesterol}",
        f"Stress Level: {Stress_Level}",
        f"Stroke Risk Percentage: {risk_percent}%",
        f"Result: {result_text}"
    ]

    for item in report_data:
        elements.append(Paragraph(item, styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)

    st.download_button(
        label="📄 Download Medical Report",
        data=buffer.getvalue(),
        file_name="Stroke_Risk_Report.pdf",
        mime="application/pdf"
    )
