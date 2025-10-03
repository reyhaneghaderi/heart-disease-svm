# Heart Disease Prediction with Calibrated SVM

This project builds a machine learning pipeline for predicting the risk of heart disease using patient data (age, blood pressure, cholesterol, chest pain type, etc.).
The model is a Support Vector Machine (SVM) with full preprocessing, hyperparameter tuning, and probability calibration for reliable clinical interpretation.
# Project Overview
- Objective: Predict presence of heart disease (binary classification).
- Techniques:
    - Preprocessing with ColumnTransformer:
    - Numerical features: KNN Imputer → StandardScaler
    - Categorical features: SimpleImputer → OneHotEncoder
    - Model: Support Vector Classifier (SVC)
    - Hyperparameter tuning: GridSearchCV with 5-fold Stratified CV
    - Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1
    - Calibration (Platt scaling – sigmoid) for trustworthy probability outputs
# Key Results

  - Best parameters: kernel='rbf', C≈1.1, gamma≈0.01

-  - Cross-validation ROC-AUC (mean): 0.9216

   - Test performance (pre-calibration):

   - ROC-AUC ≈ 0.922

   - PR-AUC ≈ 0.923

 - Test performance (post-calibration):

    - ROC-AUC ≈ 0.925

    - PR-AUC ≈ 0.931

  - Brier score: 0.102

  - ECE (Expected Calibration Error): 0.047

- Interpretation: Calibration improved probability reliability—after sigmoid scaling, predicted probabilities better match true frequencies, crucial for medical decision support.
