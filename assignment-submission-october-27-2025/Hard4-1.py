#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#loading dataset
df = pd.read_csv(
    "/Users/vinuthnapagidi/Downloads/BehavioralRF–Vision&Eye2025.csv",
    usecols=["YearStart", "Age", "Sex", "RiskFactor", "Data_Value"],  # using only necessary columns
    dtype={
        "YearStart": "int16",
        "Age": "category",
        "Sex": "category",
        "RiskFactor": "category",
        "Data_Value": "float32"
    })
#filtering for the latest year and dropping missing values
df = df[df["YearStart"] == 2022].dropna(subset=["Data_Value"])

#create binary target variable (above/below national average)
avg_value = df["Data_Value"].mean()
df["Above_Avg"] = (df["Data_Value"].values > avg_value).astype(np.int8)

#encode categorical features efficiently (using .cat.codes instead of LabelEncoder loop)
for col in ["Age", "Sex", "RiskFactor"]:
    df[col] = df[col].cat.codes.astype("int8")

#splitting data into features and target
X = df[["Age", "Sex", "RiskFactor"]]
y = df["Above_Avg"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

#hyperparameter tuning with GridSearchCV
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"],
    "penalty": ["l1", "l2"],
    "max_iter": [200, 500]}

log_reg = LogisticRegression()  
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,      # keep this for parallel grid search
    verbose=1)

#fit the tuned model
grid_search.fit(X_train, y_train)

#evaluating best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nHyperparameter Optimization Results")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", round(grid_search.best_score_, 3))

print("\nTest Set Evaluation")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
