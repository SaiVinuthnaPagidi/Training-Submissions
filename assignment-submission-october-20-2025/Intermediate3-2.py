import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load the Alzheimer's dataset
df = pd.read_csv("/Users/vinuthnapagidi/Downloads/Alzheimers_Data.csv")
# focus only on the Mental Health category
mental_health = df[df['Class'] == 'Mental Health']
# select relevant numeric columns and remove missing values
data = mental_health[['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']].dropna()

# define features (X) and target variable (y)
X = data[['Low_Confidence_Limit', 'High_Confidence_Limit']].to_numpy()
y = data['Data_Value'].to_numpy()

# scale the data for uniform comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# initialize and fit a simple linear regression model
model = LinearRegression()
model.fit(X_scaled, y)
# predict using the model
predictions = model.predict(X_scaled)

# evaluate model performance using NumPy functions
mse = np.mean((y - predictions) ** 2)
corr = np.corrcoef(y, predictions)[0, 1]

print("Model Coefficients:", model.coef_)
print("Intercept:", round(model.intercept_, 2))
print("Mean Squared Error:", round(mse, 2))
print("Correlation (Actual vs Predicted):", round(corr, 2))

# visualize actual vs predicted values
plt.figure(figsize=(7,5))
plt.scatter(y, predictions, alpha=0.6)
plt.xlabel("Actual Data Value (%)")
plt.ylabel("Predicted Data Value (%)")
plt.title("Actual vs Predicted Mental Health Data Values")
plt.grid(True)
plt.show()
