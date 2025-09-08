import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the trained model
model = xgb.Booster()
model.load_model("churn_model.json")

# Streamlit UI
st.title("📊 Churn Prediction App")
st.markdown("Provide the key customer details below:")

# Input fields (Top 15 features only)
international_plan = st.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
number_customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
area_code = st.selectbox("Area Code", [408, 415, 510])
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)
number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0)
total_day_charge = st.number_input("Total Day Charge", min_value=0.0)
total_intl_calls = st.number_input("Total International Calls", min_value=0)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0)
total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0)
total_intl_charge = st.number_input("Total International Charge", min_value=0.0)
total_night_charge = st.number_input("Total Night Charge", min_value=0.0)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0)

# Prediction logic
if st.button("Predict Churn"):
    # Build input row
    input_data = pd.DataFrame([{
        "account_length": 0,
        "area_code": area_code,
        "international_plan": 1 if international_plan == "Yes" else 0,
        "total_intl_minutes": total_intl_minutes,
        "total_intl_calls": total_intl_calls,
        "voice_mail_plan": 1 if voice_mail_plan == "Yes" else 0,
        "number_vmail_messages": number_vmail_messages,
        "total_day_minutes": total_day_minutes,
        "total_day_calls": 0,
        "total_day_charge": total_day_charge,
        "total_eve_minutes": total_eve_minutes,
        "total_eve_calls": total_eve_calls,
        "total_eve_charge": total_eve_charge,
        "total_night_minutes": total_night_minutes,
        "total_night_calls": 0,
        "total_night_charge": total_night_charge,
        "total_intl_minutes_2": 0,
        "total_intl_calls_2": 0,
        "total_intl_charge": total_intl_charge,
        "number_customer_service_calls": number_customer_service_calls
    }])

    # Match training feature order
    expected_features = [
        'account_length', 'area_code', 'international_plan', 'total_intl_minutes',
        'total_intl_calls', 'voice_mail_plan', 'number_vmail_messages',
        'total_day_minutes', 'total_day_calls', 'total_day_charge',
        'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge',
        'total_intl_minutes_2', 'total_intl_calls_2', 'total_intl_charge',
        'number_customer_service_calls'
    ]
    input_data = input_data[expected_features]

    # Convert to DMatrix
    dmatrix = xgb.DMatrix(input_data)

    # Get prediction probabilities
    prob = model.predict(dmatrix)[0]   # booster.predict gives probabilities for binary:logistic
    prediction = 1 if prob >= 0.5 else 0

    # Display results
    label = "Churn" if prediction == 1 else "No Churn"
    st.success(f"Prediction: {label}")
    st.info(f"Churn Probability: {prob:.2%}")