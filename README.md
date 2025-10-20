# ML-Loan-Prediction# Loan Approval Prediction

## Project Overview
This project builds a machine learning model to predict loan approval status based on applicant financial and demographic data. The model uses a Decision Tree Classifier and handles class imbalance with SMOTE.

---

## Dataset
- The dataset used contains 500 loan applications with features such as number of dependents, education, employment status, income, loan amount, credit score, and asset values.
- The dataset file is named `loan_approval_dataset_500.csv`.
- The `loan_id` column is dropped as it does not contribute to prediction.

---

## Features
- Number of Dependents
- Education (Graduate/Not Graduate)
- Self Employed (Yes/No)
- Annual Income
- Loan Amount
- Loan Term (in years)
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

---

## Model Pipeline

1. **Data Preprocessing**
    - Strip whitespace from column names and categorical values.
    - Encode categorical variables using Label Encoding.
    - Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

2. **Train-Test Split**
    - Split the dataset into training (80%) and testing (20%) sets.

3. **Feature Scaling**
    - Scale numerical features using StandardScaler for improved model performance.

4. **Model Training**
    - Train a Decision Tree Classifier with specified hyperparameters.

5. **Evaluation**
    - Evaluate the model using accuracy score.
    - Display confusion matrix and classification report.
    - Visualize feature importances.

6. **User Input Prediction**
    - Interactive function to predict loan approval based on user input.

---

## How to Run

1. Ensure you have Python 3.x installed.

2. Install required packages:

```bash
pip install pandas numpy scikit-learn imblearn matplotlib seaborn
