from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle

class LoanApprovalModel:
    def __init__(self, data_path='Dataset_A_loan.csv', model_path='rf_model.pkl'):
        self.label_encoders = {}
        self.scaler = None
        self.model = None
        self.numeric_features = ['person_age', 'person_income', 'person_emp_exp',
                                 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                                 'cb_person_cred_hist_length', 'credit_score']
        self.categorical_cols = []
        self._train_and_save(data_path, model_path)

    def _train_and_save(self, data_path, model_path):
        df = pd.read_csv(data_path)

        df = self.fix_and_encode(df)

        x = df.drop('loan_status', axis=1)
        y = df['loan_status']

        for col in self.categorical_cols:
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col])
            self.label_encoders[col] = le

        x_train, _, y_train, _ = train_test_split(x, y, train_size=0.7, random_state=0)

        x_train = self.preprocess(x_train, is_train=True)

        self.model = RandomForestClassifier(n_estimators=100, random_state=5)
        self.model.fit(x_train, y_train)

        self._evaluate_model(x, y)

        self.save(model_path)

    def fix_and_encode(self, df):
        df.dropna()
        self.categorical_cols = df.select_dtypes(include=['object']).columns

        for col in self.categorical_cols:
            unique_vals = df[col].unique()
            print(f"{col}: {unique_vals}")
            gender_mapping = {
                'female': 'Female',
                'fe male': 'Female',
                'male': 'Male',
                'Male': 'Male'
            }

            education_mapping = {
                'master': 'Master',
                'high school': 'HighSchool',
                'bachelor': 'Bachelor',
                'associate': 'Associate',
                'doctorate': 'Doctorate'
            }

            home_ownership_mapping = {
                'RENT': 'Rent',
                'OWN': 'Own',
                'MORTGAGE': 'Mortgage',
                'OTHER': 'Other'
            }

            loan_intent_mapping = {
                'PERSONAL': 'Personal',
                'EDUCATION': 'Education',
                'MEDICAL': 'Medical',
                'VENTURE': 'Venture',
                'HOMEIMPROVEMENT': 'Home Improvement',
                'DEBTCONSOLIDATION': 'Debt Consolidation'
            }

            loan_default_mapping = {
                'No': 'No',
                'Yes': 'Yes'
            }

            df['person_gender'] = df['person_gender'].str.lower().map(gender_mapping)

            df['person_education'] = df['person_education'].str.lower().map(education_mapping)

            df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)

            df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)

            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(loan_default_mapping)

            return df

    def preprocess(self, df, is_train=True):
        df = df.copy()

        if is_train:
            self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col in self.numeric_features:
            df[col] = df[col].fillna(df[col].median())

        if is_train:
            self.scaler = StandardScaler()
            df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
        else:
            df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        return df


    def preprocess_input(self, input_df):
        df = input_df.copy()

        for col in self.label_encoders:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col])

        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        return df

    def train(self, x_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=5)
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _evaluate_model(self, x, y):
        _, x_test, _, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

        for col in self.categorical_cols:
            le = self.label_encoders[col]
            x_test[col] = le.transform(x_test[col])

        x_test = self.preprocess(x_test, is_train=False)

        y_pred = self.model.predict(x_test)
        y_proba = self.model.predict_proba(x_test)[:, 1]

        print("\nModel Evaluation:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        

if __name__ == "__main__":
    model = LoanApprovalModel()
