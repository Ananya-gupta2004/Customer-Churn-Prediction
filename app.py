import streamlit as st 
import pandas as pd
import joblib
 
#load saved files
model=joblib.load("churn_model.pkl")
scaler=joblib.load("scaler.pkl")
features_names=joblib.load("feature_names.pkl")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn probability.")

#user inputs
tenure=st.number_input(
    "tenure Months",
    min_value=0,
    max_value=100,
    value=12
)
monthly_charges=st.number_input(
    "Monthly charges",
    min_value=0.0,
    value=70.0
    
)
total_charges=st.number_input(
    "Total charges",
    min_value=0.0,
    value=1000.0
    
)
dependents=st.selectbox(
    "Dependents",
    ["No","Yes"]
)
contract=st.selectbox(
    "Contract Type",
    [
        "Month-to month",
        "One Year",
        "Two Year"
    ]
)
internet_service=st.selectbox(
    "Internet Service",
    [
        "DSL",
        "fiber optic",
        "No"
    ]
)
paperless=st.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
)
payment_method=st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)
# Predict button
if st.button("Predict Churn"):

    # Create empty dataframe
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=features_names
    )

    # Numerical features
    input_data["Tenure Months"] = tenure
    input_data["Monthly Charges"] = monthly_charges
    input_data["Total Charges"] = total_charges

    # Dependents
    if "Dependents_Yes" in input_data.columns:
        input_data["Dependents_Yes"] = (
            1 if dependents == "Yes" else 0
        )

    # Contract
    if contract == "One Year":
        if "Contract_One year" in input_data.columns:
            input_data["Contract_One year"] = 1

    elif contract == "Two Year":
        if "Contract_Two year" in input_data.columns:
            input_data["Contract_Two year"] = 1

    # Internet Service
    if internet_service == "fiber optic":
        if "Internet Service_Fiber optic" in input_data.columns:
            input_data["Internet Service_Fiber optic"] = 1

    elif internet_service == "No":
        if "Internet Service_No" in input_data.columns:
            input_data["Internet Service_No"] = 1

    # Paperless Billing
    if paperless == "Yes":
        if "Paperless Billing_Yes" in input_data.columns:
            input_data["Paperless Billing_Yes"] = 1

    # Payment Method
    if payment_method == "Electronic check":
        if "Payment Method_Electronic check" in input_data.columns:
            input_data["Payment Method_Electronic check"] = 1

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader(
        f"Churn Probability: {probability:.2%}"
    )

    if prediction == 1:
        st.error(
            "⚠️ Customer is likely to churn."
        )
    else:
        st.success(
            "✅ Customer is likely to stay."
        )