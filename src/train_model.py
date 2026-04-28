# src/train_model.py

import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


def train_model(df):

    # -------- SPLIT FEATURES & TARGET --------
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -------- TRAIN TEST SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- HYPERPARAMETER TUNING --------
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    # -------- BEST MODEL --------
    model = grid.best_estimator_

    print("\n🔥 Best Parameters:", grid.best_params_)

    # -------- PREDICTIONS --------
    y_pred = model.predict(X_test)

    # -------- ACCURACY --------
    accuracy = accuracy_score(y_test, y_pred)

    # -------- SAVE MODEL --------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("✅ Model saved to models/model.pkl")

    # -------- RETURN VALUES --------
    return model, accuracy, X.columns, X_test, y_test, y_pred