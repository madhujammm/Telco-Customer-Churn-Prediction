import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and expected feature columns
model = joblib.load('random_forest_model.pkl')
columns = joblib.load('columns.pkl')

st.title("📊 Telco Customer Churn Prediction App")

# Input UI
st.markdown("Enter customer details:")

gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ['Yes', 'No'])
Dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service?", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines?", ['No phone service', 'No', 'Yes'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security?", ['Yes', 'No', 'No internet service'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
PaperlessBilling = st.selectbox("Paperless Billing?", ['Yes', 'No'])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

# Build input DataFrame with all features
input_dict = {
    'gender': [1 if gender == 'Male' else 0],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [1 if Partner == 'Yes' else 0],
    'Dependents': [1 if Dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if PhoneService == 'Yes' else 0],
    'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'Contract': [Contract],
    'PaymentMethod': [PaymentMethod]
}

input_df = pd.DataFrame(input_dict)

# One-hot encode
input_df_encoded = pd.get_dummies(input_df)

# Add missing columns from training
for col in columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Ensure correct order
input_df_encoded = input_df_encoded[columns]

# Scale features if needed (optional — only if you used a scaler and saved it)
# scaler = joblib.load('scaler.pkl')
# input_df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
#     input_df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df_encoded)[0]
    prob = model.predict_proba(input_df_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {prob:.2f})")
