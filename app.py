import os
import streamlit as st
import pandas as pd
import subprocess
import sys


st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 AI-Powered Fraud Detection System")
st.markdown("---")

if not os.path.exists("artifacts/model_trainer/model.pkl"):
    import subprocess
    subprocess.run([sys.executable, "main.py"])

from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

# ===============================
# BASIC TRANSACTION SECTION
# ===============================

st.subheader("📌 Basic Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    hour_of_day = st.slider("Hour of Day", 0, 23, 12)
    is_weekend = st.selectbox("Is Weekend?", [0, 1])

with col2:
    txn_count_last_24h = st.slider("Transactions in Last 24h", 0, 50, 1)
    avg_amount_last_24h = st.number_input("Avg Amount (Last 24h)", min_value=0.0, value=500.0)
    is_international = st.selectbox("International Transaction?", [0, 1])

st.markdown("---")

# ===============================
# ADVANCED FRAUD SIGNALS
# ===============================

with st.expander("🔎 Advanced Fraud Signals (Behavioral & Risk Indicators)"):

    col3, col4 = st.columns(2)

    with col3:
        device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.8)
        ip_address_risk_score = st.slider("IP Address Risk Score", 0.0, 1.0, 0.1)
        otp_success_rate_customer = st.slider("OTP Success Rate", 0.0, 1.0, 0.95)

    with col4:
        merchant_historical_fraud_rate = st.slider("Merchant Fraud Rate", 0.0, 1.0, 0.05)
        past_fraud_count_customer = st.slider("Past Fraud Count", 0, 10, 0)
        location_change_flag = st.selectbox("Location Changed?", [0, 1])
        device_change_flag = st.selectbox("Device Changed?", [0, 1])

st.markdown("---")

# ===============================
# PREDICTION BUTTON
# ===============================

if st.button("🚀 Analyze Transaction"):

    input_df = pd.DataFrame([{
        "amount": amount,
        "txn_count_last_24h": txn_count_last_24h,
        "avg_amount_last_24h": avg_amount_last_24h,
        "device_trust_score": device_trust_score,
        "ip_address_risk_score": ip_address_risk_score,
        "merchant_historical_fraud_rate": merchant_historical_fraud_rate,
        "otp_success_rate_customer": otp_success_rate_customer,
        "past_fraud_count_customer": past_fraud_count_customer,
        "location_change_flag": location_change_flag,
        "device_change_flag": device_change_flag,
        "is_international": is_international,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend
    }])

    pred, prob = pipeline.predict(input_df)
    fraud_probability = prob[0] * 100
    
    if (ip_address_risk_score > 0.8) and (device_trust_score < 0.3):
        pred[0] = 1
        fraud_probability = max(fraud_probability, 95)

    st.markdown("## 📊 Analysis Result")

    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric("Fraud Probability", f"{fraud_probability:.2f}%")

    with col_result2:
        if fraud_probability < 30:
            st.success("🟢 Low Risk")
        elif fraud_probability < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

    st.markdown("---")

    if pred[0] == 1:
        st.error("🚨 This Transaction is Likely FRAUDULENT")
    else:
        st.success("✅ This Transaction Appears LEGITIMATE")