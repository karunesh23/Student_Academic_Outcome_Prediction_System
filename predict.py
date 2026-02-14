import pandas as pd
import joblib

# Load saved objects
model = joblib.load("dropout_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load new data
df = pd.read_csv("AcademicSuccess.csv")
X_new = df.drop(columns=["Target"])

# One-hot encoding
X_new = pd.get_dummies(X_new, drop_first=True)

# Align columns
X_new = X_new.reindex(columns=feature_columns, fill_value=0)

# Scale
X_new_scaled = scaler.transform(X_new)

# Predict
predictions = model.predict(X_new_scaled)

result = pd.DataFrame({
    "Dropout_Risk": predictions
}).replace({
    0: "High Risk (Dropout)",
    1: "Low Risk (Continue)"
})

print(result.head())
