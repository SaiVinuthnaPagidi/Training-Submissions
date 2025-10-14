import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

results = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/results.csv")
races = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/races.csv")

# merging results with race info to get the year column
merged = results.merge(races, on="raceId")

print(results.head())
print(results.tail())
print(results.info())
print(results.describe())
print(merged.head())

# selecting the relevant columns
data = merged[['grid', 'laps', 'year', 'points']].dropna()

# define features (X) and target (y)
X = data[['grid', 'laps', 'year']]
y = data['points']

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predicting and evaluating
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))