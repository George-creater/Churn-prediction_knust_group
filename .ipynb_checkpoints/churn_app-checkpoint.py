import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the XGBoost model from JSON
model = xgb.Booster()
model.load_model("churn_model.json")

# App title
st.title("📊 Churn Prediction App")
st.markdown("Provide the key customer details below to predict churn probability:")

# --------------------------
# Top 10 Input Fields
# --------------------------
st.sidebar.header("Customer Details")

international_plan = st.sidebar.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ["No", "Yes"])
number_customer_service_calls = st.sidebar.number_input("Customer Service Calls", min_value=0)
area_code = st.sidebar.selectbox("Area Code", [408, 415, 510])
total_day_minutes = st.sidebar.number_input("Total Day Minutes", min_value=0.0)
number_vmail_messages = st.sidebar.number_input("Number of Voicemail Messages", min_value=0)
total_day_charge = st.sidebar.number_input("Total Day Charge", min_value=0.0)
total_intl_calls = st.sidebar.number_input("Total International Calls", min_value=0)
total_eve_minutes = st.sidebar.number_input("Total Evening Minutes", min_value=0.0)
total_intl_minutes = st.sidebar.number_input("Total International Minutes", min_value=0.0)

# --------------------------
# Build Full Feature DataFrame
# --------------------------
if st.button("Predict Churn"):
    # Convert categorical inputs
    intl_plan = 1 if international_plan == "Yes" else 0
    vmail_plan = 1 if voice_mail_plan == "Yes" else 0

    # Full 20-feature DataFrame (fill missing features with 0)
    input_data = pd.DataFrame([[
        0,                           # account_length
        area_code,
        intl_plan,
        total_intl_minutes,
        total_intl_calls,
        vmail_plan,
        number_vmail_messages,
        total_day_minutes,
        0,                           # total_day_calls
        total_day_charge,
        total_eve_minutes,
        0,                           # total_eve_calls
        0,                           # total_eve_charge
        0,                           # total_night_minutes
        0,                           # total_night_calls
        0,                           # total_night_charge
        0,                           # total_intl_minutes_2
        0,                           # total_intl_calls_2
        0,                           # total_intl_charge
        number_customer_service_calls
    ]], columns=[
        'account_length', 'area_code', 'international_plan', 'total_intl_minutes',
        'total_intl_calls', 'voice_mail_plan', 'number_vmail_messages',
        'total_day_minutes', 'total_day_calls', 'total_day_charge',
        'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge',
        'total_intl_minutes_2', 'total_intl_calls_2', 'total_intl_charge',
        'number_customer_service_calls'
    ])

    # Convert to DMatrix
    dmatrix = xgb.DMatrix(input_data)

   # Predict churn probability
    prob = model.predict(dmatrix)[0]  # may be numpy.float32
    prediction_label = "Churn" if prob >= 0.5 else "No Churn"
    
    # Convert to Python float for Streamlit
    prob_float = float(prob)
    
    # Display
    st.subheader(f"Prediction: {prediction_label}")
    st.progress(min(max(prob_float, 0.0), 1.0))
    st.write(f"Churn Probability: {prob_float:.2%}")