#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

#loading & preparing data
df = pd.read_csv('/Users/vinuthnapagidi/Downloads/Alzheimers_Data.csv')

#keep only relevant numeric + category columns
data = df[['Class', 'Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']].dropna()

#encoding target: 1 = Mental Health, 0 = Other
data['Target'] = np.where(data['Class'] == 'Mental Health', 1, 0)

X = data[['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']].to_numpy()
y = data['Target'].to_numpy()

#scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

#baseline model
base_model = LogisticRegression(max_iter=500, class_weight='balanced')
start_base = time.time()
base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)
base_acc = accuracy_score(y_test, base_pred)
base_bal_acc = balanced_accuracy_score(y_test, base_pred)
end_base = time.time()

#optimized model with GridSearchCV
param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear'], 'C': np.logspace(-3, 3, 7)},
    {'penalty': ['l2'], 'solver': ['lbfgs'], 'C': np.logspace(-3, 3, 7)}
]

grid_model = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy'
)

start_opt = time.time()
grid_model.fit(X_train, y_train)
end_opt = time.time()

#evaluating optimized model
opt_pred = grid_model.predict(X_test)
opt_acc = accuracy_score(y_test, opt_pred)
opt_bal_acc = balanced_accuracy_score(y_test, opt_pred)
roc = roc_auc_score(y_test, opt_pred)
conf_mat = confusion_matrix(y_test, opt_pred)

#comparing performance
print("\nBaseline Accuracy:", round(base_acc, 3))
print("Baseline Balanced Accuracy:", round(base_bal_acc, 3))
print("Optimized Accuracy:", round(opt_acc, 3))
print("Optimized Balanced Accuracy:", round(opt_bal_acc, 3))
print("ROC-AUC Score:", round(roc, 3))
print("Best Parameters:", grid_model.best_params_)
print("Baseline Time:", round(end_base - start_base, 3), "s")
print("Optimized Time:", round(end_opt - start_opt, 3), "s")
print("\nClassification Report:\n", classification_report(y_test, opt_pred, zero_division=0))

#confusion matrix visualization
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='PuRd',
            xticklabels=['Other', 'Mental Health'],
            yticklabels=['Other', 'Mental Health'])
plt.title('Optimized Logistic Regression – Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
