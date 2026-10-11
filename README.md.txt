# SWYNEX Employee Attrition Prediction

A machine learning project developed as part of the SWYNEX Technologies Machine Learning Internship.

## Project Overview

This project uses employee data to build and evaluate classification models that predict whether an employee may leave an organization. The workflow covers data validation, preprocessing, model training, evaluation, and hyperparameter tuning.

## Objectives

- Explore and validate the employee attrition dataset.
- Prepare features and encode the target variable.
- Build reproducible preprocessing and classification pipelines.
- Compare Logistic Regression and Random Forest.
- Tune Logistic Regression using stratified cross-validation.
- Evaluate model performance using classification metrics and ROC-AUC.

## Dataset

- Records: 1,470
- Original columns: 35
- Target variable: `Attrition`
- Target encoding: `No = 0`, `Yes = 1`
- Training set: 80%
- Test set: 20%
- Split: stratified, with `random_state=42`

The project uses an Employee Attrition dataset. Refer to the dataset's original source and license before redistributing the data.

## Models

1. Logistic Regression with balanced class weights
2. Random Forest with balanced class weights
3. Tuned Logistic Regression selected using 5-fold stratified cross-validation

The tuning search selected `C = 0.01` and `class_weight = balanced`, optimizing cross-validation recall.

## Final Test Results

| Metric | Tuned Logistic Regression |
|---|---:|
| Accuracy | 0.7687 |
| Precision | 0.3820 |
| Recall | 0.7234 |
| F1 Score | 0.5000 |
| ROC-AUC | 0.7985 |

The final model correctly identified 34 of the 47 employees who left in the held-out test set. It also flagged 55 employees who stayed as potential leavers. This demonstrates the trade-off between detecting more attrition cases and generating false positives.

## Repository Structure

```text
SWYNEX-Machine-Learning-Model/
├── SWYNEX_Employee_Attrition_Model.ipynb
├── README.md
├── requirements.txt
└── artifacts/
    ├── employee_attrition_model.joblib
    ├── model_comparison.csv
    └── final_metrics.json
```

## How to Run

1. Install Python 3.10 or later.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `SWYNEX_Employee_Attrition_Model.ipynb` in Jupyter Notebook or Google Colab.
4. Follow the notebook cells in order.
5. Upload the dataset when prompted, if it is not already available in the environment.

## Load the Saved Model

```python
import joblib

model = joblib.load("artifacts/employee_attrition_model.joblib")
```

Use the same feature columns and compatible preprocessing environment as the training notebook when making predictions.

## Limitations and Responsible Use

- This model is an educational project, not a validated HR decision system.
- Predictions are uncertain and should not be used alone to make employment decisions.
- Employee data may contain historical or organizational biases.
- False positives and false negatives can affect people differently.
- The model requires further fairness, calibration, generalization, and business-impact testing before any real-world use.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Joblib · Google Colab

## Internship

Developed as part of the Machine Learning Internship at SWYNEX Technologies.