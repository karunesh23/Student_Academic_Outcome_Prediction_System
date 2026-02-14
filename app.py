import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Academic Success Predictor",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🎓 Academic Success / Graduate & Dropout Prediction")
st.markdown("### XGBoost Model with Hyperparameter Tuning")
st.markdown("---")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("dropout_xgb_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# -------------------------------
# Sidebar Upload
# -------------------------------
st.sidebar.header("📌 Upload Student Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        # Remove target if exists
        if "Target" in df.columns:
            X = df.drop(columns=["Target"])
        else:
            X = df.copy()

        # One-hot encoding
        X = pd.get_dummies(X, drop_first=True)

        # Align columns with training features
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Convert to numeric
        X = X.astype(float)

        # Predict button
        if st.button("🔍 Predict Dropout Risk"):

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            result = pd.DataFrame({
                "Prediction_Code": predictions,
                "Probability_of_Continuing": np.round(probabilities, 3)
            })

            result["Prediction"] = result["Prediction_Code"].replace({
                0: "High Risk (Dropout)",
                1: "Low Risk (Continue)"
            })

            st.subheader("📈 Prediction Results")
            st.dataframe(result[["Prediction", "Probability_of_Continuing"]])

            # Summary Metrics
            st.subheader("📊 Summary Statistics")

            high_risk = (predictions == 0).sum()
            low_risk = (predictions == 1).sum()

            col1, col2 = st.columns(2)
            col1.metric("🚨 High Risk Students", int(high_risk))
            col2.metric("✅ Low Risk Students", int(low_risk))

            st.success("✅ Prediction Completed Successfully!")

            # -------------------------------
            # Feature Importance
            # -------------------------------
            st.subheader("📊 Feature Importance")

            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(feature_importance_df["Feature"],
                    feature_importance_df["Importance"])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top 15 Important Features")

            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin prediction.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Karunesh | Machine Learning Deployment Project")
