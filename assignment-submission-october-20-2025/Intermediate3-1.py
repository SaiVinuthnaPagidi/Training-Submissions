import pandas as pd
import numpy as np

# load the uploaded Alzheimer's dataset
data = pd.read_csv("/Users/vinuthnapagidi/Downloads/Alzheimers_Data.csv")
print(data.head())
print(data.info)
print(data.describe)

# focus only on the Mental Health category
mental_health = data[data['Class'] == 'Mental Health']

# convert the Data_Value column to a NumPy array
values = mental_health['Data_Value'].dropna().to_numpy()

# basic statistics
average = np.mean(values)
peak = np.max(values)
lowest = np.min(values)
spread = np.ptp(values)  # range = max - min

# identify the "happiest" and "most stressed" locations
best_state = mental_health.iloc[np.argmax(values)]['LocationDesc']
worst_state = mental_health.iloc[np.argmin(values)]['LocationDesc']

# standardizing the data to compare fairly
z_scores = (values - average) / np.std(values)
high_outliers = np.sum(z_scores > 1)
low_outliers = np.sum(z_scores < -1)

print("Average mental health score:", round(average, 2))
print("Range (spread):", round(spread, 2))
print("Highest value:", round(peak, 2), "in", best_state)
print("Lowest value:", round(lowest, 2), "in", worst_state)
print("Regions significantly above average:", high_outliers)
print("Regions significantly below average:", low_outliers)

