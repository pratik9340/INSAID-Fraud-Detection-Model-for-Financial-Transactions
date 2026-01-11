# Fraud Detection for Financial Transactions 

# Project Overview

This project implements an end-to-end machine learning pipeline for proactive fraud detection in financial transactions.
The solution focuses on data quality, feature engineering, class imbalance handling, and model selection to accurately identify fraudulent activities at scale.

Domain: Banking & Financial Risk

Dataset Size: Millions of transactions

Target: Binary classification (Fraud vs Non-Fraud)

Outcome: Achieved 99.96% accuracy using Random Forest with optimized features

# Business Problem

Financial institutions face significant losses due to fraudulent transactions, often hidden within highly imbalanced datasets.

Challenges addressed:

Severe class imbalance

High multicollinearity between financial features

Need for interpretable and scalable ML models

Early detection of high-risk transactions

# Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

ML Models: Logistic Regression, Decision Tree, Random Forest, XGBoost

Techniques: Feature Engineering, VIF, GridSearchCV, Stratified Sampling

# End-to-End Pipeline
1️). Data Ingestion

Loaded transaction data from .csv into Pandas DataFrame

Identified 8 numerical and 3 categorical features

Verified no missing values

2️). Exploratory Data Analysis (EDA)

Log-transformed highly skewed features (amount, oldbalanceOrg, oldbalanceDest)

Identified fraud-prone transaction types:

TRANSFER

CASH_OUT

Confirmed no fraud cases in PAYMENT, DEBIT, CASH_IN

3️). Feature Selection & Encoding

Dropped non-informative identifiers (nameOrig, nameDest)

Applied One-Hot Encoding on transaction type

Performed Pearson correlation analysis to assess feature relationships

4️). Handling Multicollinearity

Computed Variance Inflation Factor (VIF)

Removed highly correlated features:

oldbalanceOrg

oldbalanceDest

Ensured remaining features had VIF < 5

5️). Feature Engineering

Created new business-driven features:

balance_change = oldbalanceOrg - newbalanceOrig

transaction_speed = amount / step

isFlaggedFraud (rule-based indicator)

These features significantly improved fraud signal detection.

6️). Class Imbalance Handling

Used Stratified Train-Test Split

Ensured minority fraud class was properly represented

# Model Development & Evaluation
Models Tested
Model	Accuracy
Logistic Regression	99.87%
Decision Tree	99.89%
XGBoost	99.87%
Random Forest	99.96% ✅

Evaluation Metrics:

Confusion Matrix

Precision, Recall, F1-Score

Accuracy

# Random Forest outperformed all models and provided robust fraud detection with minimal false negatives.

# Key Fraud Indicators

Large transaction amounts

Sudden balance depletion

High transaction velocity

TRANSFER and CASH_OUT operations

These indicators align with real-world fraud patterns, validating model assumptions.

# Fraud Prevention Recommendations

Real-time flagging of high-risk transactions

Velocity-based transaction monitoring

Threshold-based alerts on balance changes

Continuous model retraining with recent data

# Measuring Success Post-Deployment

Reduction in false negatives

Increase in fraud capture rate

Monitoring fraud loss reduction over time

Model performance drift analysis

# Key Takeaways (Recruiter Highlight)

Built production-ready ML pipeline

Strong focus on data quality & feature engineering

Experience handling real-world class imbalance

Model interpretability aligned with business risk teams

