# Support Vector Machine Classification on Heart Disease Dataset

This project demonstrates how to build, tune, and evaluate an SVM classifier for predicting the presence of heart disease using a publicly available dataset.

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Feature Encoding](#feature-encoding)  
- [Model Training & Hyperparameter Tuning](#model-training--hyperparameter-tuning)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [Requirements](#requirements)  
- [Usage](#usage)  

---

## Overview

We load a heart‐disease dataset, perform cleaning and imputation, encode categorical features, scale the data, and train an SVM with a linear kernel. We then tune hyperparameters via grid search and evaluate performance using standard classification metrics and ROC analysis.

## Dataset

- **Source**: `heart.csv`  
- **Shape**: ~1,000 records, 14 columns:  
  - **Features**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, etc.  
  - **Target**: `HeartDisease` (0 = no disease, 1 = disease)

## Preprocessing

1. **Load Data**  
   ```python
   df = pd.read_csv("heart.csv")
   data = df.copy()
   ```

2. **Identify Missing Values**  
   - Replace zeroes in numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`) with NaN, since zeros are not physiologically valid.  
   - Count missing entries.  

3. **Imputation**  
   - Use **KNNImputer** (k=3) to fill missing values based on nearest neighbors.  
   ```python
   imputer = KNNImputer(n_neighbors=3)
   data_imputed = imputer.fit_transform(data)
   ```

## Feature Encoding

- **Categorical Label Encoding**  
  - For binary categories (e.g. `Sex`, `ExerciseAngina`), apply `LabelEncoder`.  

- **One-Hot Encoding**  
  - For multi-class categoricals (`ChestPainType`, `RestingECG`, `ST_Slope`), apply `OneHotEncoder` and append the resulting dummy columns.  
  ```python
  ohe = OneHotEncoder()
  ohe_feats = ohe.fit_transform(data[["ChestPainType","RestingECG","ST_Slope"]]).toarray()
  ```

- **Concatenate**  
  - Merge encoded features back with the numeric array, and separate out the target vector.

## Train/Test Split & Scaling

- Split data into **training** (70%) and **testing** (30%) sets with `train_test_split(shuffle=True)`  
- Scale features to zero mean/unit variance with `StandardScaler`.  

## Model Training & Hyperparameter Tuning

1. **Baseline SVM**  
   ```python
   svc = SVC(kernel='linear', probability=True)
   ```

2. **Grid Search**  
   - Tune `C` over `[0.1, 1, 10, 100]` with 5-fold CV using `GridSearchCV`.  
   - Select best `C` based on cross-validated accuracy.  

3. **Final Model**  
   - Retrain SVM on full training set with the optimal hyperparameter.

## Evaluation

- **Predictions** on the hold-out test set.  
- **Metrics**:  
  - Confusion matrix  
  - Classification report: precision, recall, f1-score  
  - ROC curve and AUC  

## Results

- **Best Parameter**: `C = 1`  
- **Test Accuracy**: 0.86  
- **AUC**: 0.90  
- **Interpreted via confusion matrix**:  
  - True Positives: 144  
  - False Positives: 25  
  - True Negatives: 95  
  - False Negatives: 12  

```text
              precision    recall  f1-score   support

           0       0.88      0.79      0.83       120
           1       0.85      0.92      0.88       156

    accuracy                           0.86       276
   macro avg       0.87      0.85      0.86       276
weighted avg       0.86      0.86      0.86       276
```

## Requirements

```txt
numpy
pandas
scikit-learn
matplotlib
```

