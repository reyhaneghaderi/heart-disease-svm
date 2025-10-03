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
 - To predict whether a patient is likely to have heart disease (target = HeartDisease, binary classification) and provide not only accurate predictions but also trustworthy probability estimates that can support doctors in decision-making.
      
 
 # Pipeline Design

  - Preprocessing (ColumnTransformer):
  - Numerical: KNNImputer → StandardScaler
  - Categorical: SimpleImputer → OneHotEncoder
   - Classifier: SVC (linear / RBF kernel)
  - Model Selection: GridSearchCV with Stratified 5-fold CV
  - Calibration: Platt scaling (CalibratedClassifierCV(method="sigmoid"))
       
# Key Results

  - Best parameters: kernel='rbf', C≈1.1, gamma≈0.01
  - Cross-validation ROC-AUC (mean): 0.9216
   - Test performance (pre-calibration):
   - ROC-AUC ≈ 0.922
   - PR-AUC ≈ 0.923
 - Test performance (post-calibration):
    - ROC-AUC ≈ 0.925
    - PR-AUC ≈ 0.931
  - Brier score: 0.102
  - ECE (Expected Calibration Error): 0.047

 # Threshold Optimization

  - Instead of the default 0.5, the optimal threshold was chosen at 0.438 to balance precision and recall:
  - Precision: 0.877
  - Recall: 0.912
  - F1-score: 0.894

- This threshold ensures high sensitivity (few missed heart disease cases) while maintaining strong precision.   
- Interpretation: Calibration improved probability reliability—after sigmoid scaling, predicted probabilities better match true frequencies, crucial for medical decision support.

  # Explain Results 

   - Overall accuracy: The model achieves about 92.5% discriminative power (ROC-AUC = 0.925). This means out of 100 patients, around 92 are correctly classified   as having or not having heart disease.
    - Precision (87.7%): When the model predicts “disease,” it is correct almost 9 out of 10 times.
   -  Recall (91.2%): The model successfully detects 9 out of 10 actual patients with heart disease, missing only about 9%.
    - Calibration: The model outputs reliable probabilities. For example, if it says a patient has a 70% chance of disease, in reality ~70% of such patients are  truly positive. This is critical for medical trust.

  - Use in practice: The model can act as a decision support tool to help doctors identify high-risk patients earlier. It is not a replacement for clinical judgment, but a complement to tests and medical expertise.

  # Visual Insights

- Hyperparameter search plots: Show stability across C values for linear kernel and overfitting risk for large gamma in RBF kernel.
- Calibration curve: After Platt scaling, probabilities closely follow the ideal diagonal.

# Why This Project Stands Out

 - Combines strong predictive accuracy with probability calibration → essential in healthcare.
 - Demonstrates full end-to-end pipeline: imputation, scaling, encoding, model selection, calibration, evaluation.
 - Achieved state-of-the-art level ROC-AUC > 0.92 on UCI Heart dataset.
 - Highlights ability to balance applied ML engineering (pipelines, GridSearch, calibration) with scientific rigor (validation, reliability analysis).

 # Skills Demonstrated

 - Machine Learning Engineering: Scikit-learn Pipelines, cross-validation, model selection.
 - Data Preprocessing: Missing data imputation, scaling, categorical encoding.
  - Evaluation: ROC/PR curves, confusion matrix, calibration metrics.
  - Explainability: Reliability diagrams, threshold tuning for clinical trade-offs.

# What I Learned

 - How to design a complete ML pipeline with preprocessing, imputation, scaling, and encoding using scikit-learn.
 - Practical experience in tuning SVM hyperparameters (C, gamma) and understanding their effect on overfitting/underfitting.
 - The importance of probability calibration (Platt scaling) to make outputs clinically trustworthy, not just accurate.
 - How to evaluate models beyond accuracy, using ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score, ECE, and interpret them for non-technical audiences (e.g., doctors).
 - Skills in threshold optimization to balance precision and recall depending on the use case (e.g., minimizing missed heart disease cases).
  - Gained deeper insight into communicating ML results — both in technical terms (for ML/PhD audience) and in simple, practical language (for medical practitioners).
