# fine_tuned_logreg_manual.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True) ## importing the inbuilt breast cancer data
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# Building pipeline with preprocessing + model
# Manually fine-tuning key parameters
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="lbfgs",   
        C=1.5,            # reduced regularization for better fit
        max_iter=2000,    # ensures convergence
        penalty="l2",     
        random_state=42
    ))
])
model.fit(X_tr, y_tr) #Training the model
y_pred = model.predict(X_te) #Predicting and evaluation of the model
#Display the results
acc = accuracy_score(y_te, y_pred)
print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_te, y_pred))
