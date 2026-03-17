import streamlit as st
import requests
import time
import pandas as pd

st.set_page_config(page_title="Customer Retention Intelligence", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

st.title("💳 Customer Retention Intelligence System")

# ---------------- SESSION STATE ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("📥 Customer Input")

    credit_score = st.number_input("Credit Score", 300, 900, 650, help="Customer credit score")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 1000000.0, 50000.0)
    num_products = st.number_input("Products", 1, 4, 2)
    has_credit = st.selectbox("Credit Card", [0, 1])
    is_active = st.selectbox("Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 1000000.0, 50000.0)

    st.markdown("---")
    st.success("🟢 API Connected")

# ---------------- CACHE API ---------------- #
@st.cache_data
def call_api(data):
    return requests.post(API_URL, json=data).json()

# ---------------- MAIN ACTION ---------------- #
if st.button("🚀 Predict Churn"):

    # -------- VALIDATION -------- #
    if age <= 0 or balance < 0 or salary < 0:
        st.error("⚠️ Invalid input values")
        st.stop()

    data = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_credit),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": float(salary)
    }

    # -------- API CALL -------- #
    try:
        with st.spinner("🔍 Analyzing customer behavior..."):
            start = time.time()
            result = call_api(data)
            end = time.time()

        if "error" in result:
            st.error("⚠️ API Error")
            st.stop()

        prediction = int(result["prediction"])
        probability = float(result["churn_probability"])
        prob_percent = probability * 100

        # -------- DASHBOARD METRICS -------- #
        col1, col2, col3 = st.columns(3)

        col1.metric("📊 Probability", f"{prob_percent:.2f}%")
        col2.metric("⚡ Response Time", f"{end-start:.2f}s")
        col3.metric("📌 Decision", "Churn" if prediction else "Stay")

        st.progress(probability)

        # -------- DECISION -------- #
        if prediction == 1:
            st.error(f"⚠️ Customer likely to churn ({prob_percent:.2f}%)")
        else:
            st.success(f"✅ Customer likely to stay ({prob_percent:.2f}%)")

        # -------- RISK LEVEL -------- #
        def risk_label(prob):
            if prob > 0.75:
                return "🔴 High Risk"
            elif prob > 0.5:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"

        st.markdown(f"### {risk_label(probability)}")

        # -------- BUSINESS ACTION -------- #
        st.subheader("💼 Recommended Action")

        if prediction == 1:
            st.error("🚨 Immediate Action: Call + retention offer")
        elif probability > 0.5:
            st.warning("📩 Send engagement email")
        else:
            st.success("👍 No action needed")

        # -------- INPUT SUMMARY -------- #
        st.subheader("📋 Customer Summary")
        st.json(data)

        # -------- SIMPLE EXPLANATION -------- #
        st.subheader("🔍 Key Influencing Factors")

        if balance > 100000:
            st.write("• High balance → possible churn risk")

        if is_active == 0:
            st.write("• Inactive member → higher churn probability")

        if age > 50:
            st.write("• Older customers → higher churn tendency")

        if num_products <= 1:
            st.write("• Low product usage → low engagement")

        # -------- SHAP (IF AVAILABLE) -------- #
        if "reasons" in result:
            st.subheader("🧠 Model Explanation")
            for r in result["reasons"]:
                st.write(f"• {r}")

        # -------- HISTORY TRACKING -------- #
        st.session_state.history.append(probability)

        st.subheader("📈 Prediction History")

        df = pd.DataFrame({
            "Prediction #": range(1, len(st.session_state.history) + 1),
            "Probability": st.session_state.history
        })

        st.line_chart(df.set_index("Prediction #"))

        # -------- RESET BUTTON -------- #
        if st.button("🧹 Reset History"):
            st.session_state.history = []
            st.success("History Cleared!")

    except Exception as e:
        st.error("🚨 Failed to connect to API")
        st.text(str(e))