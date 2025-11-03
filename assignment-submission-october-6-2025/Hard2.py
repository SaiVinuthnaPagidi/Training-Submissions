import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

df = pd.read_csv("/Users/vinuthnapagidi/Downloads/Sleep_health_and_lifestyle_dataset.csv") #loading the dataset

print(df.shape)
print(df.head())
print(df.info())

#cleaning the data
# Drop Person ID
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

print("\nMissing values:\n", df.isna().sum()) # Checking for missing values

df = df.dropna() # Filling or drop missing values

#Encoding categorical columns
cat_cols = df.select_dtypes(include="object").columns
print("\nCategorical columns:", list(cat_cols))

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#Features and Target
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

#Splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model using random forest
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluation of the model and classification metrics
y_pred = model.predict(X_test)

print("\n=== Sleep Disorder Prediction Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
print("\n📊 Saved confusion_matrix.png")

#Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top10, y=top10.index)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
print("📈 Saved feature_importance.png")

