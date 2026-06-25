# Customer Churn Prediction System

## Overview

Customer churn is a major challenge for subscription-based businesses such as telecom companies. Retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer is likely to leave the service based on customer demographics, account information, and service usage patterns.

The project uses Machine Learning techniques to analyze customer behavior, identify churn drivers, and provide churn predictions through an interactive Streamlit web application.

---

## Objectives

* Analyze customer data to understand churn patterns.
* Build and compare multiple machine learning models.
* Identify the most influential factors affecting customer churn.
* Provide explainable predictions using SHAP (SHapley Additive exPlanations).
* Deploy the model through a user-friendly Streamlit dashboard.

---

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer information such as:

* Customer demographics
* Account information
* Contract details
* Billing information
* Internet services
* Churn status

Target Variable:

* **Churn Value**

  * 1 → Customer Churned
  * 0 → Customer Retained

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## Project Workflow

### 1. Data Preprocessing

* Removed irrelevant columns:

  * CustomerID
  * State
  * Country
  * City
  * Lat Long
  * Churn Reason

* Converted Total Charges to numeric format.

* Handled missing values using median imputation.

* Applied one-hot encoding to categorical features.

---

### 2. Exploratory Data Analysis (EDA)

Analyzed relationships between customer attributes and churn using:

* Count plots
* Histograms
* Box plots
* Distribution analysis

Key findings:

* Customers with shorter tenure were more likely to churn.
* Month-to-month contracts showed higher churn rates.
* Fiber-optic users exhibited higher churn tendencies.
* Long-term contracts significantly reduced churn.

---

### 3. Feature Engineering

* One-hot encoding of categorical variables.
* Feature scaling using StandardScaler.
* Train-test split with stratification to maintain class balance.

---

### 4. Model Development

The following models were trained and evaluated:

#### Logistic Regression

Accuracy: **80.27%**

#### Random Forest

Accuracy: **79.06%**

#### XGBoost

Accuracy: **79.13%**

After comparison, Logistic Regression achieved the best overall performance and was selected as the final model.

---

## Model Evaluation

Final Logistic Regression Results:

* Accuracy: 80.27%
* Precision (Churn): 64%
* Recall (Churn): 57%
* F1-Score (Churn): 61%

Confusion Matrix analysis was performed to evaluate classification performance.

---

## Explainable AI (SHAP)

SHAP was used to interpret model predictions and identify key churn drivers.

### Most Important Features

* Tenure Months
* Monthly Charges
* Internet Service (Fiber Optic)
* Dependents
* Contract Type
* Total Charges

### Key Insights

Customers are more likely to churn if they:

* Have Fiber Optic Internet Service
* Use Electronic Check payment methods
* Use Streaming Services

Customers are less likely to churn if they:

* Have longer tenure
* Have one-year or two-year contracts
* Use Online Security services
* Use Tech Support services

---

## Deployment

A Streamlit web application was developed to allow users to:

* Enter customer information
* Predict churn probability
* View churn risk classification

The application loads:

* Trained Logistic Regression Model
* StandardScaler
* Feature Metadata

and performs real-time churn prediction.

---

## Project Structure

customer_churn_prediction/

├── app.py

├── train_model.py

├── churn_model.pkl

├── scaler.pkl

├── feature_names.pkl

├── requirements.txt

├── README.md

└── dataset/

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Class imbalance handling using SMOTE
* Deployment on Streamlit Cloud
* Real-time database integration
* Automated retraining pipeline
* Advanced ensemble methods

---

## Results

The developed churn prediction system successfully identifies customers at risk of leaving and provides interpretable insights into the factors influencing churn behavior. The combination of predictive modeling and explainable AI enables businesses to take proactive retention measures and improve customer satisfaction.

---

## Author

Ananya

Electronics and Communication Engineering (ECE)

Machine Learning & AI Enthusiast
