# CreditRisk_Prediction

This project predicts whether a loan application will be approved or not using a Logistic Regression model trained on the [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).


## Files
- `loan_prediction.py`: Python script with full preprocessing, model training, and evaluation.
- `train.csv`: Dataset used for training (placed in the `task2` folder).

## Objective

To build a classification model that can predict the likelihood of a loan getting approved based on applicant information like income, education, employment status, and more.


## Skills Used

- Data Cleaning: Handling missing values using `mode` and `median`.
- Feature Encoding: Using `Label Encoding` and `One-Hot Encoding`.
- Exploratory Data Analysis: Histogram, boxplot, countplot.
- Feature Scaling: Using `StandardScaler`.
- Modeling: Logistic Regression using `sklearn`.
- Model Evaluation: Accuracy Score and Confusion Matrix.

## Summary

- Missing values in columns like `LoanAmount`, `Gender`, and `Self_Employed` were filled appropriately.
- Categorical variables were mapped to numerical values or one-hot encoded.
- Features were scaled using `StandardScaler` to improve model performance.
- Logistic Regression was trained to classify loans as `Approved` or `Not Approved`.
- Evaluation was done using:
  - Accuracy Score: Measures overall correctness.
  - Confusion Matrix: Gives detailed insight into true vs. predicted classes.



## Visualizations

- Distribution of `LoanAmount` and `ApplicantIncome`.
- Box plot to detect outliers in `LoanAmount`.
- Countplot showing `Education` vs `Loan_Status`.

---

## How to Run

```bash
# In your terminal
python loan.py
