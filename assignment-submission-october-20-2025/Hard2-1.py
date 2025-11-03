import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import time

# load the alzheimer's dataset
df = pd.read_csv('/Users/vinuthnapagidi/Downloads/Alzheimers_Data.csv')

# focus on dental health category and select numeric columns
mental_health = df[df['Class'] == 'Mental Health']
data = mental_health[['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']].dropna()

# define features (X) and target variable (y)
X = data[['Low_Confidence_Limit', 'High_Confidence_Limit']].to_numpy()
y = data['Data_Value'].to_numpy()

# optimization strategy
# combine scaling and modeling into a single pipeline
# use cross-validation for faster, parallelized performance

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# measure execution time
start_time = time.time()

# cross-validated model training (using parallel jobs for speed)
scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

end_time = time.time()

# evaluate performance
r2_mean = np.mean(scores)
execution_time = end_time - start_time

print("Cross-validated R² score:", round(r2_mean, 3))
print("Execution time:", round(execution_time, 4), "seconds")
