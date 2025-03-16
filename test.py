import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@100;200;300;400;500;600;700;800;900&display=swap');

    /* ใช้ฟอนต์ Kanit กับทุกส่วนของ Streamlit */
    html, body, div, span, h1, h2, h3, h4, h5, h6, p, a, button, input, textarea, select, label {
        font-family: 'Kanit', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- ฟังก์ชันเพื่อโหลดโมเดลทั้งสองประเภท ----
# โมเดล Machine Learning (ตัวอย่าง Logistic Regression)
@st.cache_data
def load_ml_model():
    with open("MLModel.pkl", "rb") as file:
        return pickle.load(file)

# โมเดล Neural Network (ตัวอย่างใช้ Keras)
@st.cache_data
def load_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights("NNModel.h5")  # โหลดน้ำหนักของโมเดล
    return model

# หน้าแรก: อธิบายขั้นตอนการเตรียมข้อมูล (Data Preprocessing)
def page_1():
    st.title("แนวทางการพัฒนาโมเดล Machine Learning")
    st.header('About Dataset "Heart Disease Dataset"')
    st.markdown("[Visit Heart Disease Dataset](https://www.kaggle.com/datasets/mirzahasnine/heart-disease-dataset/data)")
    st.write("""
    โรคหัวใจคืออะไร?

    คำว่า “โรคหัวใจ” หมายถึงกลุ่มของโรคที่เกี่ยวข้องกับหัวใจหลายประเภท โดยประเภทที่พบมากที่สุดในสหรัฐอเมริกาคือ โรคหลอดเลือดหัวใจตีบ (Coronary Artery Disease - CAD) ซึ่งส่งผลต่อการไหลเวียนของเลือดไปยังหัวใจ หากเลือดไหลเวียนลดลง อาจทำให้เกิด ภาวะหัวใจวาย ได้

    อาการของโรคหัวใจมีอะไรบ้าง?

    บางครั้งโรคหัวใจอาจไม่แสดงอาการ (“เงียบ”) และไม่ได้รับการวินิจฉัยจนกว่าผู้ป่วยจะมีอาการของ ภาวะหัวใจวาย ภาวะหัวใจล้มเหลว หรือ ภาวะหัวใจเต้นผิดจังหวะ เมื่อเกิดเหตุการณ์เหล่านี้ อาการที่อาจพบ ได้แก่:
    
    ภาวะหัวใจวาย (Heart attack): อาการเจ็บหน้าอกหรือรู้สึกไม่สบาย, ปวดบริเวณหลังส่วนบนหรือคอ, อาการอาหารไม่ย่อย, อาการแสบร้อนกลางอก, คลื่นไส้หรืออาเจียน, อ่อนเพลียอย่างรุนแรง, ไม่สบายตัวบริเวณส่วนบนของร่างกาย, เวียนศีรษะ และหายใจลำบาก

    ภาวะหัวใจเต้นผิดจังหวะ (Arrhythmia): รู้สึกหัวใจเต้นผิดปกติหรือเต้นแรงผิดจังหวะ (ใจสั่น)

    ภาวะหัวใจล้มเหลว (Heart failure): หายใจลำบาก, อ่อนเพลีย, หรือมีอาการบวมที่เท้า, ข้อเท้า, ขา, ท้อง, หรือเส้นเลือดบริเวณคอ
    """)
    st.subheader("ตัวอย่างข้อมูลใน dataset")
    df = pd.read_csv("heart_disease.csv") 
    st.dataframe(df.head(5))
    st.subheader("Features ของ Dataset")
    st.write("""
    age: อายุของผู้ป่วย (Numerical)

    Gender: เพศของผู้ป่วย (Categorical)

        Male: เพศชาย (แปลงเป็น 1)

        Female: เพศหญิง (แปลงเป็น 0)  
             
    currentSmoker: สถานะการสูบบุหรี่ในปัจจุบัน (Categorical)

        1: สูบบุหรี่

        0: ไม่สูบบุหรี่

    cigsPerDay: จำนวนบุหรี่ที่สูบต่อวัน (Numerical)

    BPMeds: การใช้ยาลดความดันโลหิต (Categorical)

        1: ใช้ยา

        0: ไม่ใช้ยา
    
    prevalentStroke: ประวัติการเป็นโรคหลอดเลือดสมอง (Categorical)

        1: มีประวัติ

        0: ไม่มีประวัติ

    prevalentHyp: ประวัติการเป็นโรคความดันโลหิตสูง (Categorical)

        1: มีประวัติ

        0: ไม่มีประวัติ

    diabetes: ประวัติการเป็นโรคเบาหวาน (Categorical)

        1: เป็นเบาหวาน

        0: ไม่เป็นเบาหวาน 
    totChol: ระดับคอเลสเตอรอลรวมในเลือด (Numerical)

    sysBP: ความดันโลหิต systolic (Numerical)

    diaBP: ความดันโลหิต diastolic (Numerical)

    BMI: ดัชนีมวลกาย (Body Mass Index) (Numerical)

    heartRate: อัตราการเต้นของหัวใจ (Numerical)

    glucose: ระดับน้ำตาลในเลือด (Numerical)
    
    Heart_ stroke: ผลลัพธ์การวินิจฉัยโรคหัวใจหรือหลอดเลือดสมอง (Categorical)

        1: มีโรคหัวใจหรือหลอดเลือดสมอง

        0: ไม่มีโรคหัวใจหรือหลอดเลือดสมอง
    """)
    
    st.header("การเตรียมข้อมูล")
    st.subheader("การจัดการกับข้อมูลที่ไม่สมบูรณ์ (Missing Data)")
    st.write("""
    การแทนที่ค่า NA ด้วย NaN เพื่อให้ง่ายต่อการจัดการกับข้อมูลที่ไม่สมบูรณ์
    
    การตรวจสอบค่าที่หายไป
    
    การเติมค่าที่หายไปด้วยค่าเฉลี่ย (Mean Imputation) 
    คอลัมน์ที่เป็นตัวเลข (BPMeds, glucose, BMI, cigsPerDay, totChol, heartRate) ที่มีค่าหายไปถูกเติมด้วยค่าเฉลี่ยของคอลัมน์นั้นๆ
    
    การเติมค่าที่หายไปด้วยค่าที่พบมากที่สุด (Mode Imputation) 
    คอลัมน์ที่เป็น categorical (education, prevalentStroke) ที่มีค่าหายไปถูกเติมด้วยค่าที่พบมากที่สุด (mode) ของคอลัมน์นั้นๆ
    """)
    st.subheader("การแปลงข้อมูล (Data Transformation)")
    st.write("""
    การแปลงค่าคอลัมน์ categorical เป็นตัวเลข
    คอลัมน์ที่เป็น categorical (Gender, education, prevalentStroke, Heart_ stroke) ถูกแปลงเป็นตัวเลขเพื่อให้สามารถใช้ในโมเดล Machine Learning ได้
    
    การปรับสเกลข้อมูล (Scaling)
    คอลัมน์ที่เป็นตัวเลขถูกปรับสเกลให้อยู่ในช่วง 0 ถึง 1 โดยใช้ MinMaxScaler
    """)
    st.subheader("การจัดการกับ Skewness")    
    st.write("""
    การตรวจสอบ Skewness
    ตรวจสอบความเบ้ (Skewness) ของข้อมูลในคอลัมน์ที่เป็นตัวเลข
    
    การปรับ Skewness ด้วยการแปลงข้อมูล
    คอลัมน์ที่มีความเบ้สูงถูกแปลงโดยใช้ log1p (logarithm of 1 plus the value) เพื่อลดความเบ้
    คอลัมน์ glucose ถูกแปลงโดยใช้ Box-Cox Transformation ซึ่งเหมาะสำหรับข้อมูลที่เป็นบวกทั้งหมด
    """)
    st.subheader("การจัดการกับข้อมูลที่ไม่สมดุล (Imbalanced Data)")    
    st.write("""
    การใช้ SMOTE (Synthetic Minority Over-sampling Technique)
    
    การใช้ ADASYN (Adaptive Synthetic Sampling)
    """)
    st.subheader("การเลือกคุณสมบัติ (Feature Selection)")    
    st.write("""
    การใช้ SelectKBest : เลือกคุณสมบัติที่สำคัญที่สุด 10 อันดับโดยใช้ SelectKBest และ f_classif
    
    การใช้ RFE (Recursive Feature Elimination): ใช้ RFE เพื่อเลือกคุณสมบัติที่สำคัญที่สุด 10 อันดับโดยใช้ RandomForest เป็นโมเดล
    """)
    
    st.header("Model ที่ใช้ในการ train")
    st.subheader("1. Random Forest Classifier")    
    st.write("""
    วัตถุประสงค์ : ใช้สำหรับการจำแนกประเภท (Classification) โดยใช้ Decision Trees หลายต้น (Ensemble Method)

    พารามิเตอร์ที่ปรับ:

    n_estimators=100: จำนวนต้นไม้ใน Random Forest

    random_state=42: เพื่อให้ผลลัพธ์สามารถทำซ้ำได้
    """)
    st.subheader("2. K-Nearest Neighbors (KNN) Classifier")   
    st.write("""
    วัตถุประสงค์: ใช้สำหรับการจำแนกประเภทโดยใช้ระยะทางระหว่างจุดข้อมูล (Distance-based Method)

    พารามิเตอร์ที่ปรับ:

    n_neighbors=5: จำนวนเพื่อนบ้าน (Neighbors) ที่ใช้ในการทำนาย
    """)
    
    st.subheader("3. Logistic Regression")   
    st.write("""
    วัตถุประสงค์: ใช้สำหรับการจำแนกประเภทโดยใช้การถดถอยโลจิสติก

    พารามิเตอร์ที่ปรับ:

    class_weight='balanced': เพื่อจัดการกับข้อมูลที่ไม่สมดุล

    solver='liblinear': อัลกอริทึมที่ใช้ในการปรับโมเดล
    """)
    

# หน้าที่สอง: อธิบายทฤษฎีของอัลกอริธึมที่พัฒนา (Machine Learning และ Neural Network)
def page_2():
    st.title("แนวทางการพัฒนาโมเดล  Neural Networks")
    st.header('About Dataset "Fashion MNIST Dataset"')
    st.markdown("[Visit Fashion MNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)")
    st.write("""
    Fashion-MNIST เป็นชุดข้อมูลที่ประกอบด้วยภาพสินค้าของ Zalando ซึ่งประกอบด้วยชุดข้อมูลฝึกสอน (training set) จำนวน 60,000 ตัวอย่าง และชุดข้อมูลทดสอบ (test set) จำนวน 10,000 ตัวอย่าง แต่ละตัวอย่างเป็นภาพขาวดำขนาด 28x28 พิกเซล ที่มีป้ายกำกับ (label) เป็นหนึ่งใน 10 ประเภท

    Zalando ตั้งใจให้ Fashion-MNIST เป็นตัวแทนที่สามารถใช้แทนชุดข้อมูล MNIST ดั้งเดิมที่มีตัวเลขเขียนด้วยลายมือได้โดยตรง เพื่อใช้เป็นมาตรฐานในการทดสอบอัลกอริธึม Machine Learning เนื่องจาก Fashion-MNIST มีโครงสร้างของภาพและการแบ่งชุดข้อมูลฝึกกับทดสอบเหมือนกับ MNIST

    ชุดข้อมูล MNIST ดั้งเดิมนั้นเป็นชุดข้อมูลที่มีตัวเลขเขียนด้วยลายมือ ซึ่งเป็นที่นิยมในหมู่ผู้ทำงานด้าน AI/ML/Data Science และมักใช้เป็นเกณฑ์มาตรฐานในการประเมินอัลกอริธึมต่าง ๆ มีคำกล่าวที่ว่า:

        "ถ้าอัลกอริธึมของคุณใช้ไม่ได้กับ MNIST แสดงว่ามันจะใช้ไม่ได้เลย"
        
        "แต่ถ้าใช้ได้กับ MNIST ก็ยังอาจใช้ไม่ได้กับชุดข้อมูลอื่น ๆ อยู่ดี"
    ด้วยเหตุนี้ Zalando จึงพัฒนาชุดข้อมูล Fashion-MNIST ขึ้นมาเพื่อใช้แทน MNIST

    เนื้อหา (Content)
    
    แต่ละภาพมีขนาด 28 x 28 พิกเซล รวมทั้งหมด 784 พิกเซล
    
    แต่ละพิกเซลมีค่า ระดับความสว่าง (grayscale) ที่เป็นจำนวนเต็มระหว่าง 0 ถึง 255 โดยค่าที่มากขึ้นหมายถึงสีที่เข้มขึ้น
    
    ข้อมูลชุดฝึก (training set) และข้อมูลชุดทดสอบ (test set) มี 785 คอลัมน์
    
    คอลัมน์แรกเป็น ป้ายกำกับ (label) ซึ่งบอกว่าสินค้านั้นเป็นประเภทใด
    
    คอลัมน์ที่เหลือเป็น ค่าพิกเซลของภาพ

    """)
    
    st.subheader("Features ของ Dataset")
    st.write("""
    Image : รูปภาพต่างๆ
    
    Label :
    
        0 T-shirt/top
        
        1 Trouser
        
        2 Pullover
        
        3 Dress
        
        4 Coat
        
        5 Sandal
        
        6 Shirt
        
        7 Sneaker
        
        8 Bag
        
        9 Ankle boot
    """)
    
    st.header("การเตรียมข้อมูล")
    st.subheader("Download ชุดข้อมูล Fashion-MNIST")
    st.write("""
    โหลดชุดข้อมูล Fashion-MNIST ซึ่งมีภาพขาวดำขนาด 28x28 พิกเซล
    
    แบ่งออกเป็น ชุดข้อมูลฝึก (Training Set) จำนวน 60,000 ภาพ และ ชุดข้อมูลทดสอบ (Test Set) จำนวน 10,000 ภาพ
    
    train_labels และ test_labels เป็นค่าหมวดหมู่ (Class Labels) ระหว่าง 0-9 ที่บอกว่าแต่ละภาพเป็นหมวดหมู่ใด
    """)
    st.subheader("ปรับขนาดค่าพิกเซล (Normalization)")
    st.write("""
    แปลงค่าพิกเซลจากช่วง 0-255 ให้เป็น 0-1 โดยการหารด้วย 255 เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น
    """)
    st.subheader("ขยายมิติของภาพให้รองรับ Conv2D Layer")    
    st.write("""
    จากเดิมที่ train_images และ test_images มีขนาด (จำนวนภาพ, 28, 28)
    
    ใช้ np.expand_dims() เพื่อเพิ่มมิติที่ 4 ให้เป็น (จำนวนภาพ, 28, 28, 1)
    
    มิติเพิ่มเติมนี้จำเป็นสำหรับการป้อนข้อมูลให้ Convolutional Neural Network (CNN) ที่ใช้กับภาพ
    """)
    
    st.header("Model ที่ใช้ในการ train")
    st.subheader("1. Convolutional Neural Network (CNN)")    
    st.write("""
    Conv2D(32, (3,3), activation='relu') → ใช้ 32 ฟิลเตอร์ขนาด 3x3 กับ Activation Function ReLU
    
    MaxPooling2D((2,2)) → ลดขนาดภาพลงด้วย Max Pooling ขนาด 2x2
    
    ซ้ำอีก 2 ชั้น แต่เพิ่มจำนวนฟิลเตอร์เป็น 64 เพื่อให้โมเดลสามารถเรียนรู้ Feature ที่ซับซ้อนขึ้น
    """)
    

# หน้า 3: Demo การทำงานของโมเดล Machine Learning (Logistic Regression)
def page_3():
    st.title("Heart Stroke Prediction")
    st.write("Please enter the information to predict the risk of Heart Stroke.")
    
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

        probabilities = model.predict_proba(input_df)[:, 1]  # ดึงเฉพาะโอกาสเป็น '1'
        
        prediction = (probabilities >= threshold).astype(int)
        return prediction[0]
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

# หน้า 4: Demo การทำงานของโมเดล Neural Network
def page_4():
    
    model = load_model('CNN_fashion_mnist.keras')

    # ชื่อของคลาสที่โมเดลทำนาย
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Ankle boot', 'Bag']

    # สร้างหน้าต่าง Streamlit
    st.title('Fashion MNIST Image Prediction')
    st.write("Upload an image to get the model's prediction.")

    # การอัพโหลดภาพจากผู้ใช้
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if uploaded_file is not None:
        # เปิดภาพที่ผู้ใช้อัพโหลด
        img = Image.open(uploaded_file)
        
        # แปลงเป็นขาวดำ (grayscale) และปรับขนาดเป็น 28x28
        img = img.convert('L')  # แปลงเป็น grayscale
        img = img.resize((28, 28))  # ปรับขนาดให้เป็น 28x28
        
        
        # แสดงภาพที่อัพโหลดก่อนการแปลง
        st.image(uploaded_file, caption="Original Uploaded Image", use_container_width=True)
        # แสดงภาพที่ผู้ใช้อัพโหลด
        st.image(img, caption="Uploaded Image", use_container_width=True)
       

        # ทำการเตรียมภาพให้เหมาะสมกับโมเดล
        img_array = np.array(img) / 255.0  # Normalize ค่าพิกเซลให้อยู่ในช่วง [0, 1]
        img_array = np.expand_dims(img_array, axis=-1)  # เพิ่มมิติช่องสี (28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติ batch (1, 28, 28, 1)
        
        # ทำนายผลจากโมเดล (โมเดลที่คุณมี)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        
        # แสดงผลการทำนาย
        st.write(f"Prediction: {class_names[predicted_class]}")

# ---- การเลือกหน้าที่จะแสดงใน Streamlit ----
pages = {
    "Machine Learning Steps Development": page_1,
    "Neural Network Steps Development": page_2,
    "Machine Learning": page_3,
    "Neural Network": page_4
}


page_selection = st.sidebar.radio("Menu", list(pages.keys()))


pages[page_selection]()
