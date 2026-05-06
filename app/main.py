import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inference import ChurnPredictor

st.set_page_config(page_title="ChurnSense", page_icon="📈", layout="wide")

st.title("📈 ChurnSense: Predict Customer Churn")
st.markdown("Enter the customer's behavioral and demographic data to evaluate their risk of churning.")

@st.cache_resource
def load_predictor():
    try:
        return ChurnPredictor()
    except Exception as e:
        return None

predictor = load_predictor()

if not predictor:
    st.error("Model artifacts not found. Please train the model first by running `src/data_pipeline.py` and `src/model.py`.")
    st.stop()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Customer Profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
with col2:
    st.header("📱 Service Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Toggle extra services if internet is active
    if internet_service != "No":
        options = ["No", "Yes"]
    else:
        options = ["No internet service"]
        
    online_security = st.selectbox("Online Security", options)
    online_backup = st.selectbox("Online Backup", options)

with col3:
    st.header(" ")
    device_protection = st.selectbox("Device Protection", options)
    tech_support = st.selectbox("Tech Support", options)
    streaming_tv = st.selectbox("Streaming TV", options)
    streaming_movies = st.selectbox("Streaming Movies", options)
    
st.divider()

col4, col5 = st.columns(2)

with col4:
    st.header("💳 Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
with col5:
    st.header("💰 Charges")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8670.0, value=500.0)

user_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

st.divider()

if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing profile..."):
        result = predictor.predict(user_data)
        
    st.markdown("### Prediction Result")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if result['churn_prediction']:
            st.error(f"⚠️ High Risk of Churn!")
        else:
            st.success(f"✅ Low Risk of Churn.")
            
    with res_col2:
        st.metric(label="Churn Probability", value=f"{result['churn_probability']:.1%}")
