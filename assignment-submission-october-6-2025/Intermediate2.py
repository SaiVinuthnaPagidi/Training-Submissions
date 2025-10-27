from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer() # importing the inbuilt breast cancer data (X = features, y = labels)
X, y = data.data, data.target

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
#Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Training the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Prediction and evaluation of the model
y_pred = model.predict(X_test) 
acc = accuracy_score(y_test, y_pred) 
#print accuracy and classification report
print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))