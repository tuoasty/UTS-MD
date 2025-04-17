import streamlit as st
import pandas as pd
import pickle
from model import LoanApprovalModel

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("💰 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval.")

st.header("Applicant Information")

# Options for categorical inputs
gender_options = ["Male", "Female"]
education_options = ["Master", "HighSchool", "Bachelor", "Associate", "Doctorate"]
home_ownership_options = ["Rent", "Own", "Mortgage", "Other"]
loan_intent_options = ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Debt Consolidation"]
default_history_options = ["No", "Yes"]

with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", gender_options)
        age = st.slider("Age", 18, 80, 18)
        education = st.selectbox("Education Level", education_options)
        income = st.number_input("Annual Income ($)", min_value=0)

    with col2:
        employment_exp = st.number_input("Employment Experience (years)", min_value=0, value=0)
        home_ownership = st.selectbox("Home Ownership", home_ownership_options)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000)
        loan_intent = st.selectbox("Loan Purpose", loan_intent_options)

    with col3:
        loan_interest = st.slider("Loan Interest Rate (%)", 0.0, 30.0, 0)
        credit_length = st.slider("Credit History Length", 0, 50, 0)
        credit_score = st.slider("Credit Score", 300, 850, 300)
        previous_defaults = st.selectbox("Previous Loan Defaults", default_history_options)

    submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        loan_percent_income = (loan_amount / income) if income else 0

        input_df = pd.DataFrame([{
            "person_age": age,
            "person_gender": gender,
            "person_education": education,
            "person_income": income,
            "person_emp_exp": employment_exp,
            "person_home_ownership": home_ownership,
            "loan_amnt": loan_amount,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_interest,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": credit_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_defaults
        }])

        processed = model.preprocess_input(input_df)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0]

        st.success("Prediction: **Approved** ✅" if prediction == 1 else "Prediction: **Rejected** ❌")
        st.info(f"Approval Probability: {prob:.2%}")
