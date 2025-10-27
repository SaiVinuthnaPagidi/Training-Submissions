import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#load and filter the dataset
df = pd.read_csv("/Users/vinuthnapagidi/Downloads/BehavioralRF–Vision&Eye2025.csv")
df_2022 = df[df["YearStart"] == 2022][["LocationDesc", "Age", "Sex", "RiskFactor", "Data_Value"]].dropna()

#create a binary target variable (above or below average)
avg_value = df_2022["Data_Value"].mean()
df_2022["Above_Avg"] = (df_2022["Data_Value"] > avg_value).astype(int)

#encoding categorical columns
label_cols = ["Age", "Sex", "RiskFactor"]
encoder = LabelEncoder()
for col in label_cols:
    df_2022[col] = encoder.fit_transform(df_2022[col])

#split data into features and target
X = df_2022[["Age", "Sex", "RiskFactor"]]
y = df_2022["Above_Avg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#making predictions and evaluating
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Evaluation")
print("Accuracy:", round(accuracy, 3))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
