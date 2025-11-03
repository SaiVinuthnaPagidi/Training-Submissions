import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#loading dataset
df = pd.read_csv("/Users/vinuthnapagidi/Downloads/U.S._Chronic_Disease_Indicators_20251103.csv", low_memory=False)
df = df[['LocationDesc', 'Topic', 'Question', 'DataValue']]
df = df.dropna(subset=['DataValue'])
df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')

#filtering indicators (check both Topic & Question)
def keyword_filter(df, keyword):
    mask = df['Topic'].str.contains(keyword, case=False, na=False) | df['Question'].str.contains(keyword, case=False, na=False)
    return df[mask]

heart = keyword_filter(df, "heart")
obesity = keyword_filter(df, "obes")
smoking = keyword_filter(df, "smok")
inactivity = keyword_filter(df, "inactiv")
bp = keyword_filter(df, "blood pressure")
chol = keyword_filter(df, "cholesterol")

print("Rows found:")
print(f"  Heart disease: {len(heart)}")
print(f"  Obesity: {len(obesity)}")
print(f"  Smoking: {len(smoking)}")
print(f"  Inactivity: {len(inactivity)}")
print(f"  High BP: {len(bp)}")
print(f"  Cholesterol: {len(chol)}")

#aggregating by state
def avg_by_state(data, name):
    return data.groupby('LocationDesc', as_index=False)['DataValue'].mean().rename(columns={'DataValue': name})

heart_state = avg_by_state(heart, 'HeartDisease')
obesity_state = avg_by_state(obesity, 'Obesity')
smoking_state = avg_by_state(smoking, 'Smoking')
inactivity_state = avg_by_state(inactivity, 'Inactivity')
bp_state = avg_by_state(bp, 'HighBP')
chol_state = avg_by_state(chol, 'Cholesterol')

#merging all predictors
merged = heart_state.merge(obesity_state, on='LocationDesc', how='inner')
merged = merged.merge(smoking_state, on='LocationDesc', how='inner')
merged = merged.merge(inactivity_state, on='LocationDesc', how='inner')
merged = merged.merge(bp_state, on='LocationDesc', how='inner')
merged = merged.merge(chol_state, on='LocationDesc', how='inner')

print(f"\nMerged Dataset Shape: {merged.shape}")
print(merged.head())

#create classification target (High vs Low risk)
threshold = merged['HeartDisease'].median()
merged['HeartRisk'] = np.where(merged['HeartDisease'] >= threshold, 1, 0)

#define features (X) and target (y)
features = ['Obesity', 'Smoking', 'Inactivity', 'HighBP', 'Cholesterol']
X = merged[features]
y = merged['HeartRisk']

#split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Random Forest Classifier (Default)
print("\nRunning Random Forest Classifier (Default)")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nDefault Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

#hyperparameter optimization
print("\nOptimizing Random Forest with GridSearchCV")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"\nBest Parameters: {best_params}")

#evaluate optimized model
y_pred_best = best_rf.predict(X_test)

acc_best = accuracy_score(y_test, y_pred_best)
print(f"\nOptimized Model Accuracy: {acc_best:.2f}")
print(classification_report(y_test, y_pred_best))

#confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
disp.plot(cmap='Greens')
plt.title('Confusion Matrix (Optimized Random Forest)')
plt.show()

#feature importance
importances = best_rf.feature_importances_
plt.barh(features, importances, color='teal')
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Optimized Random Forest)')
plt.show()
