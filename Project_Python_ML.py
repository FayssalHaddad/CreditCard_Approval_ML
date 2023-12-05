# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:00:43 2023

@author: Fayssal
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# Load the data
file_path = ''
df = pd.read_csv(file_path)

# Preprocess the data
df['CNT_CHILDREN'] = df['CNT_CHILDREN'].apply(lambda x: 3 if x == '2+ children' else 0 if x == 'No children' else int(x.split()[0]))
le = LabelEncoder()

# Encoding categorical features
categorical_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'JOB', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

label_encoders = {}

for column in categorical_columns:
    le = preprocessing.LabelEncoder()
    le.fit(df[column])
    df[column] = le.transform(df[column])
    label_encoders[column] = le


# Split the data into features (X) and target (y)
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Randomized 5-fold cross-validation for Random Forest
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_precision_scores = []
rf_recall_scores = []
rf_f1_scores = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]
    
    rf.fit(X_train_kf, y_train_kf)
    y_pred_rf = rf.predict(X_val_kf)
    
    rf_precision_scores.append(precision_score(y_val_kf, y_pred_rf))
    rf_recall_scores.append(recall_score(y_val_kf, y_pred_rf))
    rf_f1_scores.append(f1_score(y_val_kf, y_pred_rf))

# Calculate average metrics for Random Forest
rf_avg_precision = np.mean(rf_precision_scores)
rf_avg_recall = np.mean(rf_recall_scores)
rf_avg_f1 = np.mean(rf_f1_scores)

# Neural Network
nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Randomized 5-fold cross-validation for Neural Network
nn_precision_scores = []
nn_recall_scores = []
nn_f1_scores = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]
    
    nn.fit(X_train_kf, y_train_kf, validation_data=(X_val_kf, y_val_kf), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    y_pred_nn = (nn.predict(X_val_kf) > 0.5).astype("int32")
    
    nn_precision_scores.append(precision_score(y_val_kf, y_pred_nn))
    nn_recall_scores.append(recall_score(y_val_kf, y_pred_nn))
    nn_f1_scores.append(f1_score(y_val_kf, y_pred_nn))

# Calculate average metrics for Neural Network
nn_avg_precision = np.mean(nn_precision_scores)
nn_avg_recall = np.mean(nn_recall_scores)
nn_avg_f1 = np.mean(nn_f1_scores)

# Print average metrics
print("Random Forest:")
print("Average Precision: {:.4f}".format(rf_avg_precision))
print("Average Recall: {:.4f}".format(rf_avg_recall))
print("Average F1 Score: {:.4f}".format(rf_avg_f1))
print("\nNeural Network:")
print("Average Precision: {:.4f}".format(nn_avg_precision))
print("Average Recall: {:.4f}".format(nn_avg_recall))
print("Average F1 Score: {:.4f}".format(nn_avg_f1))

# Print average metrics for comparison
print("Comparing Random Forest and Neural Network:\n")
print("Average Precision:")
print("Random Forest: {:.4f}".format(rf_avg_precision))
print("Neural Network: {:.4f}".format(nn_avg_precision))
print("\nAverage Recall:")
print("Random Forest: {:.4f}".format(rf_avg_recall))
print("Neural Network: {:.4f}".format(nn_avg_recall))
print("\nAverage F1 Score:")
print("Random Forest: {:.4f}".format(rf_avg_f1))
print("Neural Network: {:.4f}".format(nn_avg_f1))

# Generate confusion matrices for Random Forest and Neural Network

# Train the models on the full training set
rf.fit(X_train, y_train)
nn.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)

# Make predictions on the test set
y_pred_rf_test = rf.predict(X_test)
y_pred_nn_test = (nn.predict(X_test) > 0.5).astype("int32")

# Calculate confusion matrices
rf_cm = confusion_matrix(y_test, y_pred_rf_test)
nn_cm = confusion_matrix(y_test, y_pred_nn_test)

# Print confusion matrices
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)
print("\nNeural Network Confusion Matrix:")
print(nn_cm)

# Calculate standard deviation of metrics for Random Forest
rf_std_precision = np.std(rf_precision_scores)
rf_std_recall = np.std(rf_recall_scores)
rf_std_f1 = np.std(rf_f1_scores)

# Calculate standard deviation of metrics for Neural Network
nn_std_precision = np.std(nn_precision_scores)
nn_std_recall = np.std(nn_recall_scores)
nn_std_f1 = np.std(nn_f1_scores)

# Print standard deviation of metrics
print("Random Forest:")
print("Standard Deviation of Precision: {:.4f}".format(rf_std_precision))
print("Standard Deviation of Recall: {:.4f}".format(rf_std_recall))
print("Standard Deviation of F1 Score: {:.4f}".format(rf_std_f1))
print("\nNeural Network:")
print("Standard Deviation of Precision: {:.4f}".format(nn_std_precision))
print("Standard Deviation of Recall: {:.4f}".format(nn_std_recall))
print("Standard Deviation of F1 Score: {:.4f}".format(nn_std_f1))

# Random Forest feature importances
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances for Random Forest
plt.figure(figsize=(10, 5))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Method to evaluate input variable importance for Neural Network
def drop_column_importance(model, X_val, y_val, base_score):
    column_importances = []
    for col_idx in range(X_val.shape[1]):
        X_val_copy = X_val.copy()
        X_val_copy[:, col_idx] = 0
        dropped_col_score = f1_score(y_val, (model.predict(X_val_copy) > 0.5).astype("int32"))
        column_importances.append(base_score - dropped_col_score)
    return column_importances

# Neural Network feature importances
nn_base_score = f1_score(y_test, (nn.predict(X_test) > 0.5).astype("int32"))
nn_importances = drop_column_importance(nn, X_test, y_test, nn_base_score)

# Plot feature importances for Neural Network
plt.figure(figsize=(10, 5))
plt.title("Feature Importances (Neural Network)")
plt.bar(range(X.shape[1]), nn_importances, align="center")
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Add necessary imports
from sklearn.metrics import accuracy_score

# Initialize lists for training and validation accuracy for Random Forest
rf_train_acc_scores = []
rf_val_acc_scores = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]
    
    rf.fit(X_train_kf, y_train_kf)
    
    y_pred_rf_train = rf.predict(X_train_kf)
    y_pred_rf_val = rf.predict(X_val_kf)
    
    rf_train_acc_scores.append(accuracy_score(y_train_kf, y_pred_rf_train))
    rf_val_acc_scores.append(accuracy_score(y_val_kf, y_pred_rf_val))

# Calculate average training and validation accuracy for Random Forest
rf_avg_train_acc = np.mean(rf_train_acc_scores)
rf_avg_val_acc = np.mean(rf_val_acc_scores)

# Initialize lists for training and validation loss for Neural Network
nn_train_loss_scores = []
nn_val_loss_scores = []

for train_index, val_index in kf.split(X_train, y_train):
    X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
    y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]
    
    history = nn.fit(X_train_kf, y_train_kf, validation_data=(X_val_kf, y_val_kf), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    nn_train_loss_scores.append(history.history['loss'][-1])
    nn_val_loss_scores.append(history.history['val_loss'][-1])

# Calculate average training and validation loss for Neural Network
nn_avg_train_loss = np.mean(nn_train_loss_scores)
nn_avg_val_loss = np.mean(nn_val_loss_scores)

# Print average training and validation metrics
print("Random Forest:")
print("Average Training Accuracy: {:.4f}".format(rf_avg_train_acc))
print("Average Validation Accuracy: {:.4f}".format(rf_avg_val_acc))
print("\nNeural Network:")
print("Average Training Loss: {:.4f}".format(nn_avg_train_loss))
print("Average Validation Loss: {:.4f}".format(nn_avg_val_loss))


# Split the data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

men_indices_train = np.array(X_train_raw.index[X_train_raw['CODE_GENDER'] == 1])
women_indices_train = np.array(X_train_raw.index[X_train_raw['CODE_GENDER'] == 0])
men_indices_test = np.array(X_test_raw.index[X_test_raw['CODE_GENDER'] == 1])
women_indices_test = np.array(X_test_raw.index[X_test_raw['CODE_GENDER'] == 0])

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


def evaluate_gender_bias(model, X_train, X_test, y_train, y_test, men_indices_train, women_indices_train, men_indices_test, women_indices_test):
    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Get indices relative to X_test
    men_indices_test_rel = np.array([i for i in range(len(X_test)) if X_test_raw.index[i] in men_indices_test])
    women_indices_test_rel = np.array([i for i in range(len(X_test)) if X_test_raw.index[i] in women_indices_test])

    # Make predictions for men and women separately in the test set
    y_pred_men = model.predict(X_test[men_indices_test_rel, :])
    y_pred_women = model.predict(X_test[women_indices_test_rel, :])

    # Calculate the evaluation metric (accuracy) for men and women
    accuracy_men = accuracy_score(y_test.iloc[men_indices_test_rel], y_pred_men)
    accuracy_women = accuracy_score(y_test.iloc[women_indices_test_rel], y_pred_women)

    # Calculate the gender bias (difference in accuracy)
    gender_bias = abs(accuracy_men - accuracy_women)

    print("Accuracy for men: {:.2f}".format(accuracy_men))
    print("Accuracy for women: {:.2f}".format(accuracy_women))
    print("Gender bias: {:.2f}".format(gender_bias))

    return gender_bias


print("Random Forest Gender Bias Evaluation:")
evaluate_gender_bias(rf, X_train, X_test, y_train, y_test, men_indices_train, women_indices_train, men_indices_test, women_indices_test)

def evaluate_gender_bias_nn(model, X_train, X_test, y_train, y_test, men_indices_train, women_indices_train, men_indices_test, women_indices_test):
    # Split the training data into training and validation subsets
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Train the neural network model
    model.fit(X_train_sub, y_train_sub, validation_data=(X_val_sub, y_val_sub), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)

    # Get indices relative to X_test
    men_indices_test_rel = np.array([i for i in range(len(X_test)) if X_test_raw.index[i] in men_indices_test])
    women_indices_test_rel = np.array([i for i in range(len(X_test)) if X_test_raw.index[i] in women_indices_test])

    # Make predictions for men and women separately in the test set
    y_pred_men = (model.predict(X_test[men_indices_test_rel, :]) > 0.5).astype("int32")
    y_pred_women = (model.predict(X_test[women_indices_test_rel, :]) > 0.5).astype("int32")

    # Calculate the evaluation metric (accuracy) for men and women
    accuracy_men = accuracy_score(y_test.iloc[men_indices_test_rel], y_pred_men)
    accuracy_women = accuracy_score(y_test.iloc[women_indices_test_rel], y_pred_women)

    # Calculate the gender bias (difference in accuracy)
    gender_bias = abs(accuracy_men - accuracy_women)

    print("Accuracy for men: {:.2f}".format(accuracy_men))
    print("Accuracy for women: {:.2f}".format(accuracy_women))
    print("Gender bias: {:.2f}".format(gender_bias))

    return gender_bias


print("Neural Network Gender Bias Evaluation:")
evaluate_gender_bias_nn(nn, X_train, X_test, y_train, y_test, men_indices_train, women_indices_train, men_indices_test, women_indices_test)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("men_indices_train:", men_indices_train)
print("women_indices_train:", women_indices_train)
print("men_indices_test:", men_indices_test)
print("women_indices_test:", women_indices_test)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("Men indices train:", men_indices_train)
print("Women indices train:", women_indices_train)
print("Men indices test:", men_indices_test)
print("Women indices test:", women_indices_test)

print("Number of men in the train set:", len(men_indices_train))
print("Number of women in the train set:", len(women_indices_train))
print("Number of men in the test set:", len(men_indices_test))
print("Number of women in the test set:", len(women_indices_test))