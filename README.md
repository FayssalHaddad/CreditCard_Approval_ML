# Credit Card Approval Prediction

## Overview
This project aims to predict credit card approval decisions using machine learning models. It focuses on preprocessing data, training models like Random Forest and Neural Networks, and evaluating their performance.

## Data
- **Source**: Credit Card Approval dataset.
- **Features**: Various financial and personal attributes of applicants.
- **Target**: Approval (Yes/No) of credit card applications.

## Preprocessing
- **Categorical Encoding**: Transforming categorical features into numeric using Label Encoding.
- **Feature Scaling**: Standardizing features for model compatibility.
- **Data Cleaning**: Handling missing values and anomalies.

## Modeling
1. **Random Forest**:
   - Trained with 5-fold Stratified Cross-Validation.
   - Evaluation using precision, recall, and F1-score.

2. **Neural Network**:
   - Architecture: Sequential model with Dense and Dropout layers.
   - Optimized using Adam optimizer and Early Stopping.

## Metrics Evaluation
- **Comparison of Models**: Assessing both models on precision, recall, F1-score, and accuracy.
- **Feature Importance Analysis**: Identifying key predictors in the Random Forest model.

## Key Libraries
- pandas
- numpy
- scikit-learn
- TensorFlow

## Execution
Run the Python script in an environment supporting the aforementioned libraries. The script includes data loading, preprocessing, model training, evaluation, and feature importance analysis.

## Conclusion
The project demonstrates the effective use of machine learning techniques in predicting credit card approvals, highlighting the importance of feature preprocessing and model selection for optimal performance. You can find relevant results of the model performances in "ML_Results.pdf" file.
