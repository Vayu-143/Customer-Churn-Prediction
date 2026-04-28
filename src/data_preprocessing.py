import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(path):
    df = pd.read_csv(path)

    print("Data Loaded Successfully")

    # Drop ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Remove missing
    df.dropna(inplace=True)

    print("\nBefore Encoding:")
    print(df.dtypes)

    encoders = {}

    # 🔥 FORCE ENCODE EVERYTHING NON-NUMERIC
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Encoding: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Save encoders
    joblib.dump(encoders, 'models/encoders.pkl')

    print("\nAfter Encoding:")
    print(df.dtypes)

    return df