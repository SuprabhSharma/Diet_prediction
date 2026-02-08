import streamlit as st
import pandas as pd
import joblib


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Diet Health Predictor",
    page_icon="🥗",
    layout="wide"
)


# ---------- CLEAN BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#f8fbf9,#eef6f2);
}
.health-title {
    font-size:60px;
    font-weight:800;
    color:#0b7a55;
    text-align:center;
}
.health-sub {
    font-size:22px;
    text-align:center;
    color:#444;
}
.stButton>button {
    background: linear-gradient(90deg,#0b7a55,#34c38f);
    color:white;
    font-size:18px;
    border-radius:10px;
    height:3em;
    width:100%;
}
.result-box {
    padding:20px;
    border-radius:14px;
    background:white;
    border-left:6px solid #34c38f;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
model = joblib.load("diet_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")


# ---------- HEADER ----------
st.markdown('<p class="health-title">🥗 DIET HEALTH PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="health-sub">Personalized nutrition recommendation based on your health data</p>', unsafe_allow_html=True)

st.divider()


# ---------- NUMERIC INPUTS ----------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 90, 30)
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
    height = st.number_input("Height (cm)", 140, 210, 170)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

with col2:
    calories = st.number_input("Daily Calories", 1000, 4000, 2000)
    cholesterol = st.number_input("Cholesterol", 100, 300, 180)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    glucose = st.number_input("Glucose", 60, 200, 100)

with col3:
    exercise = st.number_input("Exercise Hours", 0.0, 20.0, 3.0)
    adherence = st.number_input("Diet Adherence %", 0.0, 100.0, 60.0)
    imbalance = st.number_input("Nutrient Imbalance Score", 0.0, 10.0, 3.0)

st.divider()


# ---------- CATEGORICAL INPUTS ----------
col4, col5, col6 = st.columns(3)

with col4:
    gender = st.selectbox("Gender", ["Male", "Female"])
    disease = st.selectbox("Disease Type", ["None", "Obesity"])

with col5:
    severity = st.selectbox("Severity", ["Low", "Medium"])
    activity = st.selectbox("Activity", ["Low", "Moderate"])

with col6:
    allergy = st.selectbox("Allergy", ["None", "Peanuts"])
    cuisine = st.selectbox("Cuisine", ["Indian", "Italian", "Mexican"])
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

st.divider()


# ---------- PREDICTION ----------
if st.button("🔍 Predict Diet Recommendation"):

    input_dict = {
        "Age": age,
        "Weight_kg": weight,
        "Height_cm": height,
        "BMI": bmi,
        "Daily_Caloric_Intake": calories,
        "Cholesterol_mg/dL": cholesterol,
        "Blood_Pressure_mmHg": bp,
        "Glucose_mg/dL": glucose,
        "Weekly_Exercise_Hours": exercise,
        "Adherence_to_Diet_Plan": adherence,
        "Dietary_Nutrient_Imbalance_Score": imbalance,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Disease_Type_None": 1 if disease == "None" else 0,
        "Disease_Type_Obesity": 1 if disease == "Obesity" else 0,
        "Severity_Low": 1 if severity == "Low" else 0,
        "Severity_Medium": 1 if severity == "Medium" else 0,
        "Physical_Activity_Level_Low": 1 if activity == "Low" else 0,
        "Physical_Activity_Level_Moderate": 1 if activity == "Moderate" else 0,
        "Dietary_Restrictions_None": 1,
        "Allergies_Peanuts": 1 if allergy == "Peanuts" else 0,
        "Preferred_Cuisine_Indian": 1 if cuisine == "Indian" else 0,
        "Preferred_Cuisine_Italian": 1 if cuisine == "Italian" else 0,
        "Preferred_Cuisine_Mexican": 1 if cuisine == "Mexican" else 0,
        "BMI_Category_Normal": 1 if bmi_category == "Normal" else 0,
        "BMI_Category_Overweight": 1 if bmi_category == "Overweight" else 0,
        "BMI_Category_Obese": 1 if bmi_category == "Obese" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Apply scaling
    num_cols = [
        "Age","Weight_kg","Height_cm","BMI",
        "Daily_Caloric_Intake","Cholesterol_mg/dL",
        "Blood_Pressure_mmHg","Glucose_mg/dL",
        "Weekly_Exercise_Hours","Adherence_to_Diet_Plan",
        "Dietary_Nutrient_Imbalance_Score"
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict diet
    prediction = model.predict(input_df)
    diet = encoder.inverse_transform(prediction)[0]

    st.markdown(
        f"<div class='result-box'><h2>Recommended Diet: {diet}</h2></div>",
        unsafe_allow_html=True
    )

    # ---------- DIET GUIDANCE ----------
    diet_guidance = {

        "Low_Fat": {
            "eat": """🥗 Recommended Foods:
• Green vegetables, fruits  
• Whole grains  
• Boiled eggs  
• Low-fat dairy""",
            "avoid": """❌ Avoid:
• Fried food  
• Butter/ghee excess  
• High salt pickles"""
        },

        "Low_Carb": {
            "eat": """🥗 Recommended Foods:
• Leafy vegetables  
• Paneer/tofu/eggs  
• Nuts & seeds""",
            "avoid": """❌ Avoid:
• Rice, sugar, sweets  
• Bakery foods"""
        },

        "Balanced": {
            "eat": """🥗 Recommended Foods:
• Vegetables & fruits  
• Dal, paneer, eggs  
• Whole grains""",
            "avoid": """❌ Avoid:
• Junk food  
• Excess salt/sugar"""
        }
    }

    st.subheader("🥦 Diet Guidance")
    st.markdown(diet_guidance[diet]["eat"])
    st.markdown(diet_guidance[diet]["avoid"])
