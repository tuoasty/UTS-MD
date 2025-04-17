import pandas as pd
from model import LoanApprovalModel 

model = LoanApprovalModel.load("xgboost_model.pkl")

dummy_input = pd.DataFrame([{
    'person_age': 22,
    'person_gender': 'Female',
    'person_education': 'Master',
    'person_income': 71948,
    'person_emp_exp': 0,
    'person_home_ownership': 'Rent',
    'loan_amnt': 350000,
    'loan_intent': 'Personal',
    'loan_int_rate': 16,
    'loan_percent_income': 0.49,
    'cb_person_cred_hist_length': 3,
    'credit_score': 561,
    'previous_loan_defaults_on_file': 'No'
}])

processed_input = model.preprocess_input(dummy_input)
prediction = model.predict(processed_input)[0]
probability = model.predict_proba(processed_input)[0]

print(prediction)
print(f"Probability of approval: {probability:.2%}")