#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

#loading dataset
df = pd.read_csv(
    "/Users/vinuthnapagidi/Downloads/BehavioralRF–Vision&Eye2025.csv",
    usecols=["YearStart", "Age", "Sex", "RiskFactor", "Data_Value"],
    dtype={
        "YearStart": "int16",
        "Age": "category",
        "Sex": "category",
        "RiskFactor": "category",
        "Data_Value": "float32"
    }
)

#cleaning and filtering
df = df.dropna(subset=["Data_Value"])
df_2022 = df[df["YearStart"] == 2022]

#create binary target variable
avg_value = df_2022["Data_Value"].mean()
df_2022 = df_2022.copy()  # <-- add this line before modifying anything
df_2022.loc[:, "Above_Avg"] = (df_2022["Data_Value"].values > avg_value).astype(np.int8)

#encode categorical variables
for col in ["Age", "Sex", "RiskFactor"]:
    df_2022.loc[:, col] = df_2022[col].cat.codes.astype("int8")

#train-test split
X = df_2022[["Age", "Sex", "RiskFactor"]]
y = df_2022["Above_Avg"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#hyperparameter tuning for XGBoost
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
xgb = XGBClassifier(eval_metric="logloss", random_state=42)

grid_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_xgb.fit(X_train, y_train)

#evaluate optimized model
best_xgb = grid_xgb.best_estimator_
y_pred = best_xgb.predict(X_test)

print("\nXGBoost Optimization Results")
print("Best Parameters:", grid_xgb.best_params_)
print("Best Cross-Validation Accuracy:", round(grid_xgb.best_score_, 3))
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#visualization – Age & RiskFactor trends
age_summary = df_2022.groupby("Age")["Data_Value"].mean().sort_values(ascending=False)
risk_summary = df_2022.groupby("RiskFactor")["Data_Value"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
age_summary.plot(kind="bar", color="teal")
plt.title("Average Vision Difficulty by Age Group (2022)")
plt.xlabel("Age Group (Encoded)")
plt.ylabel("Average % Vision Difficulty")
plt.tight_layout()
plt.show()
plt.show(block=False)
plt.pause(2)
plt.close()


plt.figure(figsize=(8, 4))
risk_summary.plot(kind="bar", color="orange")
plt.title("Average Vision Difficulty by Risk Factor (2022)")
plt.xlabel("Risk Factor (Encoded)")
plt.ylabel("Average % Vision Difficulty")
plt.tight_layout()
plt.show()
plt.show(block=False)
plt.pause(2)
plt.close()


#trend over time (2013–2022)
trend = df.groupby("YearStart")["Data_Value"].mean()
trend.plot(marker="o", color="darkgreen", figsize=(8, 4))
plt.title("Trend of Average Vision Difficulty in the U.S. (2013–2022)")
plt.xlabel("Year")
plt.ylabel("Average % Vision Difficulty")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
plt.show(block=False)
plt.pause(2)
plt.close()

#feature importance visualization
import matplotlib.pyplot as plt

importance = best_xgb.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.barh(features, importance, color='seagreen')
plt.title("Feature Importance in XGBoost Model")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show(block=False)
plt.pause(2)
plt.close()
