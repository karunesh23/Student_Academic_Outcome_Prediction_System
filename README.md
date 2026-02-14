# 🎓 Student Dropout & Graduate Prediction using XGBoost

## 📌 Project Overview
This project predicts whether a student will:

- 🎯 Dropout
- 🎓 Graduate

The prediction is done using the **XGBoost Classification algorithm**.

The goal is to help educational institutions identify students at risk of dropping out and support them in time.

---

## 🌐 Live Streamlit App

👉 **Try the Web App Here:**  
🔗 https://your-streamlit-app-link.streamlit.app  

*(Replace with your actual deployed link)*

---

## 📊 Dataset
Dataset used: **AcademicSuccess.csv**

The dataset contains:
- Academic performance data
- Student background information
- Demographic details

### 🎯 Target Variable:
- 0 → Dropout  
- 1 → Graduate  

---

## 🛠 Technologies Used
- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit (Web Application)

---

## 🤖 Model Used
- XGBoost Classifier  
- PCA (Dimensionality Reduction)

### 📁 Saved Files:
- dropout_xgb_model.pkl  
- scaler.pkl  
- pca.pkl  
- feature_columns.pkl  

---

## 📂 Project Structure

Student_Dropout_Prediction_ML/
│
├── app.py # Streamlit Web App
├── train_model.py # Model training script
├── predict.py # Prediction script
├── preprocessing.py # Data preprocessing + PCA
├── AcademicSuccess.csv # Dataset
├── requirements.txt # Dependencies
├── README.md # Project Documentation
│
├── dropout_xgb_model.pkl # Trained model
├── scaler.pkl # StandardScaler object
├── pca.pkl # PCA object
├── feature_columns.pkl # Feature column list
│
└── predict/ # Screenshots folder
├── 1.png
├── 2.png
└── 3.png

---

## 📸 Application Screenshots

### 🖥️ Home Page
![Home Page](predict/1.png)

### 📊 Prediction Output
![Prediction](predict/2.png)

### 📈 Model Insights
![Insights](predict/3.png)

---

## 🚀 How to Run the Project Locally

```bash
git clone https://github.com/karunesh23/Student_Dropout_Prediction_ML.git
cd Student_Dropout_Prediction_ML
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Contact
Have questions, suggestions, or want to collaborate?
Feel free to reach out. I would be happy to connect and discuss ideas, projects, or opportunities.
📧 **Email:** karuneshbansal84@gmail.com  
💼 **LinkedIn:**  
https://www.linkedin.com/in/karunesh-bansal-780828380  

I look forward to connecting with you! 🚀

