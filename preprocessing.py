import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Features & target
    X = df.drop(columns=["Target"])
    y = df["Target"].map({
        "Dropout": 0,
        "Enrolled": 1,
        "Graduate": 1
    })

    print("Target distribution:")
    print(y.value_counts())

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = list(X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #  Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
