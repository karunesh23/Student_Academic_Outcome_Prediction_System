from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_and_preprocess_data
import joblib

# Load data
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
    "AcademicSuccess.csv"
)

# XGBoost model
model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",   # Important to avoid warning
    use_label_encoder=False
)

# Hyperparameter space
param_dist = {
    "n_estimators": [300, 500, 700],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3, 0.5]
}

# Random Search CV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25,
    cv=7,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

# Train
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)

print("Best Parameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save everything
joblib.dump(best_model, "dropout_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "feature_columns.pkl")

print("\n✅ XGBoost training completed")
