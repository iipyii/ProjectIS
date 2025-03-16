import streamlit as st
import pickle
import pandas as pd
import numpy as np

# โหลดโมเดลจากไฟล์ .pkl
model_path = "MLModel.pkl"  # แก้ไขชื่อไฟล์ตามที่ใช้จริง
with open(model_path, "rb") as file:
    model = pickle.load(file)

# ชื่อคอลัมน์ของข้อมูล
columns = [
    "Gender", "age", "currentSmoker", "cigsPerDay", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

def predict_heart_stroke(input_data, threshold=0.6):  # ปรับ threshold
    input_df = pd.DataFrame([input_data])

    # ตรวจสอบและเรียงลำดับคอลัมน์ให้ตรงกับตอนเทรน
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    # ใช้ predict_proba() เพื่อดูค่าความน่าจะเป็น
    probabilities = model.predict_proba(input_df)[:, 1]  # ดึงเฉพาะโอกาสเป็น '1'
    
    # ถ้า probability > threshold → พยากรณ์เป็น '1'
    prediction = (probabilities >= threshold).astype(int)
    return prediction[0]


# สร้างหน้าเว็บด้วย Streamlit
st.title("Heart Stroke Prediction App")
st.write("กรอกข้อมูลเพื่อทำนายความเสี่ยงของโรคหลอดเลือดสมอง")

# ส่วนรับข้อมูลจากผู้ใช้
user_input = {}

for col in columns:
    if col == "Gender":
        gender_option = st.selectbox("Gender", ["Male", "Female"], index=0)
        user_input[col] = 1 if gender_option == "Male" else 0  # แปลง Male = 1, Female = 0
    elif col in ["currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]:
        user_input[col] = st.selectbox(f"{col}", [0, 1], index=0)
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, format="%f")


# เมื่อกดปุ่มให้ทำการทำนาย
if st.button("Predict"):
    prediction = predict_heart_stroke(user_input)
    st.write(f"### ผลการพยากรณ์: {'มีความเสี่ยง' if prediction == 1 else 'ไม่มีความเสี่ยง'}")