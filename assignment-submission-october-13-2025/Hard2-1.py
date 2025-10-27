import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# loading datasets with optimized data types and selective columns
results = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/results.csv",
                      usecols=['grid', 'laps', 'points', 'raceId'], #loading only essential columns
                      dtype={'grid': 'int16', 'laps': 'int16', 'points': 'float32', 'raceId': 'int16'})
races = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/races.csv",
                    usecols=['raceId', 'year'], #loading only essential columns
                    dtype={'raceId': 'int16', 'year': 'int16'})

# merge efficiently using optimized keys
merged = pd.merge(results, races, on='raceId', how='inner')

# convert to NumPy arrays for faster numerical operations
X = merged[['grid', 'laps', 'year']].to_numpy(dtype=np.float32)
y = merged['points'].to_numpy(dtype=np.float32)

# splitting data and training Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression(n_jobs=-1)  # parallel computation
model.fit(X_train, y_train)

# evaluating the performance
y_pred = model.predict(X_test)
print("Optimized R² Score:", r2_score(y_test, y_pred))
print("Memory usage (MB):", merged.memory_usage(deep=True).sum() / 1e6)
