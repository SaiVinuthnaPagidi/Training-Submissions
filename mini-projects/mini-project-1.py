import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

#loading dataset

df = pd.read_csv("/Users/vinuthnapagidi/Downloads/U.S._Chronic_Disease_Indicators_20251103.csv", low_memory=False)

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nDataset Summary Statistics:")
print(df.describe(include='all').head())
print("\nFirst 5 Rows:")
print(df.head())

#selecting relevant columns and preprocessing
df = df[['YearStart', 'LocationDesc', 'Topic', 'Question', 'DataValue']]
df = df.dropna(subset=['DataValue'])
df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')

#filtering indicators (check both Topic and Question)
def keyword_filter(df, keyword):
    mask = df['Topic'].str.contains(keyword, case=False, na=False) | df['Question'].str.contains(keyword, case=False, na=False)
    return df[mask]

diabetes = keyword_filter(df, "diab")
obesity = keyword_filter(df, "obes")
inactivity = keyword_filter(df, "inactiv")

print("\nRows found:")
print(f"  Diabetes: {len(diabetes)}")
print(f"  Obesity: {len(obesity)}")
print(f"  Inactivity: {len(inactivity)}")

#aggregate by state
def avg_by_state(data, name):
    return data.groupby('LocationDesc', as_index=False)['DataValue'].mean().rename(columns={'DataValue': name})

diabetes_state = avg_by_state(diabetes, 'Diabetes')
obesity_state = avg_by_state(obesity, 'Obesity')
inactivity_state = avg_by_state(inactivity, 'Inactivity')

#merge datasets
merged = diabetes_state.merge(obesity_state, on='LocationDesc', how='inner')
merged = merged.merge(inactivity_state, on='LocationDesc', how='inner')

print(f"\nMerged Dataset Shape: {merged.shape}")
print("\nMerged Dataset Info:")
print(merged.info())
print("\nMerged Dataset Head:")
print(merged.head())
print("\nMerged Dataset Summary Statistics:")
print(merged.describe())

#define features (X) and target (y)
X = merged[['Obesity', 'Inactivity']]
y = merged['Diabetes']

#train and test split
if len(merged) < 5:
    raise ValueError("Not enough data to train the model.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear Regression (Baseline)
print("\n Running Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"\nLinear Regression Performance:")
print(f"R² Score: {r2_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")

#random Forest (Default)
print("\nRunning Random Forest Regressor")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\nRandom Forest (Default) Performance:")
print(f"R² Score: {r2_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")

#random forest optimization with GridSearchCV
print("\nOptimizing Random Forest using GridSearchCV")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest Parameters: {best_params}")

y_pred_best = best_rf.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\nOptimized Random Forest Performance:")
print(f"R² Score: {r2_best:.2f}")
print(f"RMSE: {rmse_best:.2f}")

#model comparison summary
print("\nModel Comparison Summary:")
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest (Default)', 'Random Forest (Optimized)'],
    'R² Score': [r2_lr, r2_rf, r2_best],
    'RMSE': [rmse_lr, rmse_rf, rmse_best]
})
print(comparison)

#feature importance (Optimized Random Forest)
importances = best_rf.feature_importances_
plt.barh(X.columns, importances, color='mediumseagreen')
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Optimized Random Forest)')
plt.show()

#plot Actual vs Predicted (Optimized Model)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, color='teal')
plt.xlabel('Actual Diabetes Prevalence')
plt.ylabel('Predicted Diabetes Prevalence')
plt.title('Actual vs Predicted (Optimized Random Forest)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
