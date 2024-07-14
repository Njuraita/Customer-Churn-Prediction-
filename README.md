# Customer Churn Prediction Project

## Overview

In this project, we aim to predict customer churn using various machine learning models. By identifying the factors influencing churn, we strive to build a reliable model that provides actionable insights for businesses to enhance their customer retention strategies.

## Models Used

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting Machine (GBM)
- Neural Network (MLPClassifier)

## Key Metrics and Success Criteria

1. **Accuracy**: The model should have an accuracy score of at least 85%.
2. **Precision and Recall**: Achieve at least 80% to ensure reliable churn prediction and identification.
3. **F1 Score**: Minimum of 0.75 to balance precision and recall, especially for imbalanced classes.
4. **AUC-ROC Score**: At least 0.85 to effectively distinguish between churn and non-churn customers.
5. **Confusion Matrix**: Lower the number of False Negatives (FN) to ensure most churn cases are identified.


## Features

- `CustomerID`: A unique customer identification
- `Gender`: Whether the customer is male or female
- `SeniorCitizen`: Whether a customer is a senior citizen or not
- `Partner`: Whether the customer has a partner or not (Yes, No)
- `Dependents`: Whether the customer has dependents or not (Yes, No)
- `Tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Whether the customer has a phone service or not (Yes, No)
- `MultipleLines`: Whether the customer has multiple lines or not
- `InternetService`: Customer's internet service provider (DSL, Fiber Optic, No)
- `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No Internet)
- `OnlineBackup`: Whether the customer has online backup or not (Yes, No, No Internet)
- `DeviceProtection`: Whether the customer has

