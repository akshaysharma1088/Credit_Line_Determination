# Credit_Line_Determination
Project Overview
This project predicts the optimal credit line for credit card customers based on a variety of demographic, behavioral, and financial features. The goal is to balance customer satisfaction with risk management, enabling financial institutions to make more data-driven, explainable credit line decisions.

Objectives
Build predictive models to estimate the appropriate credit limit for each customer.

Use explainable AI techniques (e.g., SHAP) to interpret model behavior.

Support business teams in automating parts of the credit decision process.

Tools & Technologies
Language: Python

Libraries: pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, matplotlib, seaborn

Environment: Jupyter Notebook

Data Description
Features: income, credit utilization, delinquency history, inquiries, tenure, DTI ratio, spending behavior.

Target: Recommended credit line amount.

Data was preprocessed to handle missing values, outliers, and class imbalance.

Methodology
Exploratory Data Analysis – Identified key patterns, correlations, and anomalies.

Feature Engineering – Created utilization bands, delinquency streaks, and capped outliers.

Modeling – Tested Logistic Regression, Random Forest, and XGBoost.

Imbalance Handling – Applied SMOTE to address rare high-limit approvals.

Explainability – Used SHAP values to highlight key decision drivers.

Evaluation – Measured performance via ROC-AUC, PR-AUC, F1 score, and calibration curves.

Results
XGBoost model achieved the highest performance, improving AUC by 8–12 points over baseline.

SHAP analysis revealed utilization, delinquency history, and income stability as the top predictors.

Demonstrated potential to increase safe approvals while reducing manual review workload.

How to Run
Clone the repo
git clone https://github.com/akshaysharma1088/Credit_Line_Prediction.git cd Credit_Line_Prediction

Install dependencies
pip install -r requirements.txt

Run the notebook
jupyter notebook Credit_Line_Prediction.ipynb
