import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")

# ------------------ LOAD MODEL ------------------
model = joblib.load("models/model.pkl")

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align: center;'>📊 Customer Churn Prediction</h1>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About"])

st.sidebar.subheader("Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Accuracy: ~0.79")

if page == "About":
    st.write("""
    ### 📌 About Project
    This app predicts customer churn using Machine Learning.

    Built using:
    - Scikit-learn
    - Pandas & NumPy
    - Streamlit

    Helps businesses identify customers likely to leave.
    """)
    st.stop()

# ------------------ INPUT ------------------

def encode_yes_no(val):
    return 1 if val == "Yes" else 0

st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (months)", 0, 72)

phone = st.selectbox("Phone Service", ["No", "Yes"])
multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

security = st.selectbox("Online Security", ["No", "Yes"])
backup = st.selectbox("Online Backup", ["No", "Yes"])
device = st.selectbox("Device Protection", ["No", "Yes"])
support = st.selectbox("Tech Support", ["No", "Yes"])

tv = st.selectbox("Streaming TV", ["No", "Yes"])
movies = st.selectbox("Streaming Movies", ["No", "Yes"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card"
])

monthly = st.number_input("Monthly Charges", value=70.0, step=1.0)
total = st.number_input("Total Charges", value=1000.0, step=10.0)

# ------------------ PREDICTION ------------------

if st.button("Predict Churn"):

    input_data = {
        "gender": 1 if gender == "Female" else 0,
        "SeniorCitizen": encode_yes_no(senior),
        "Partner": encode_yes_no(partner),
        "Dependents": encode_yes_no(dependents),
        "tenure": tenure,
        "PhoneService": encode_yes_no(phone),
        "MultipleLines": encode_yes_no(multiple),
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        "OnlineSecurity": encode_yes_no(security),
        "OnlineBackup": encode_yes_no(backup),
        "DeviceProtection": encode_yes_no(device),
        "TechSupport": encode_yes_no(support),
        "StreamingTV": encode_yes_no(tv),
        "StreamingMovies": encode_yes_no(movies),
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaperlessBilling": encode_yes_no(paperless),
        "PaymentMethod": {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer": 2,
            "Credit card": 3
        }[payment],
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    df = pd.DataFrame([input_data])

    # FIX column order
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # ------------------ RESULT ------------------
    st.subheader("📊 Prediction Result")

    if prob > 0.8:
        st.error(f"🔴 HIGH RISK ({prob:.2%})")
    elif prob > 0.5:
        st.warning(f"🟠 MEDIUM RISK ({prob:.2%})")
    else:
        st.success(f"🟢 LOW RISK ({prob:.2%})")

    # ------------------ PROGRESS BAR ------------------
    st.subheader("📈 Churn Probability")
    st.progress(float(prob))

    # ------------------ FEATURE IMPORTANCE ------------------
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    importance = importance.sort_values(ascending=False)

    st.subheader("🔍 Top Factors Affecting This Prediction")

    for feature in importance.head(5).index:
        st.write(f"👉 {feature}: {df[feature].values[0]}")

    # ------------------ GRAPH ------------------
    st.subheader("📊 Model Feature Importance")

    fig, ax = plt.subplots()
    importance.head(10).plot(kind='barh', ax=ax)
    plt.gca().invert_yaxis()

    st.pyplot(fig)

    # ------------------ DOWNLOAD ------------------
    result_df = df.copy()
    result_df["Churn Probability"] = prob

    st.download_button(
        label="📥 Download Prediction",
        data=result_df.to_csv(index=False),
        file_name="prediction.csv"
    )

    # ------------------ SAVE HISTORY ------------------
    history_file = "predictions_log.csv"

    log_df = result_df.copy()

    try:
        existing = pd.read_csv(history_file)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    except:
        pass

    log_df.to_csv(history_file, index=False)

    # ------------------ BUSINESS RECOMMENDATION ------------------
    st.subheader("💡 Recommended Action")

    if prob > 0.8:
        st.write("👉 Offer discount / retention plan immediately")
    elif prob > 0.5:
        st.write("👉 Engage with customer (email / call)")
    else:
        st.write("👉 Customer is stable — no action needed")

# ------------------ HISTORY DISPLAY ------------------
st.subheader("📁 Prediction History")

if os.path.exists("predictions_log.csv"):
    history = pd.read_csv("predictions_log.csv")
    st.dataframe(history.tail(10))
else:
    st.info("No history yet")