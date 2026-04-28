# main.py

from src.data_preprocessing import load_and_preprocess
from src.train_model import train_model
from src.visualization import plot_feature_importance

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import os


def main():

    # -------- LOAD DATA --------
    df = load_and_preprocess("data/churn.csv")

    # -------- TRAIN MODEL --------
    model, accuracy, features, X_test, y_test, y_pred = train_model(df)

    print(f"\n✅ Model Accuracy: {accuracy:.2f}")

    # -------- CLASSIFICATION REPORT --------
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # -------- CREATE IMAGES FOLDER --------
    os.makedirs("images", exist_ok=True)

    # -------- CONFUSION MATRIX --------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("images/confusion_matrix.png")
    plt.close()

    print("✅ Confusion matrix saved")

    # -------- ROC CURVE --------
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("images/roc_curve.png")
    plt.close()

    print("✅ ROC curve saved")

    # -------- FEATURE IMPORTANCE --------
    plot_feature_importance(model, features)


if __name__ == "__main__":
    main()