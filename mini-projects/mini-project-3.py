import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#loading and exploring data
df = pd.read_csv("/Users/vinuthnapagidi/Downloads/Respiratory_Conditions_Treated_in_the_Emergency_Department_20251103.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nInfo:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

#data preprocessing
df['percent_visits'] = pd.to_numeric(df['percent_visits'], errors='coerce')
df = df.dropna(subset=['percent_visits', 'condition', 'age_group'])

#create binary target based on median percent_visits
threshold = df['percent_visits'].median()
df['HighVisit'] = np.where(df['percent_visits'] > threshold, 1, 0)

print(f"\nThreshold for HighVisit classification: {threshold:.2f}")
print(df['HighVisit'].value_counts())

#encode categorical variables
le_condition = LabelEncoder()
le_age = LabelEncoder()

df['condition_encoded'] = le_condition.fit_transform(df['condition'])
df['age_encoded'] = le_age.fit_transform(df['age_group'])

#define features and target
X = df[['condition_encoded', 'age_encoded']]
y = df['HighVisit']

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - High vs. Low Visit Classification (Logistic Regression)')
plt.show()

#interpret coefficients
coef_df = pd.DataFrame({
    'Feature': ['Condition', 'Age Group'],
    'Coefficient': model.coef_[0]
})
print("\nModel Coefficients:")
print(coef_df)

#visualize coefficient strength
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='teal')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients - Respiratory Visit Classification')
plt.show()

