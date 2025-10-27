# 🏎️ Predicting Austin GP 2025 Winner Using Singapore 2025 Data

import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# setup and load Singapore GP data
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# load the Singapore GP 2025 race
session = fastf1.get_session(2025, 'Singapore', 'R')
session.load()

# 🏁 print the actual top 3 finishers
print("\n🏁 Actual Top 3 Finishers – Singapore GP 2025:")
results = session.results[['Position', 'Abbreviation', 'FullName']].head(3)
print(results.to_string(index=False))

# extract clean lap and weather data
laps = session.laps.pick_quicklaps().merge(session.weather_data, left_index=True, right_index=True, how='left')
df = laps[['Driver', 'LapTime', 'TyreLife', 'Compound', 'AirTemp', 'TrackTemp', 'Humidity']].dropna()
df['LapTime'] = df['LapTime'].dt.total_seconds()

# aggregate average driver stats
driver_stats = df.groupby('Driver').agg({
    'LapTime': 'mean',
    'TyreLife': 'mean',
    'AirTemp': 'mean',
    'TrackTemp': 'mean',
    'Humidity': 'mean'
}).reset_index()

# identify the Real Winner
winner_code = session.results.iloc[0]['Abbreviation']
print(f"\n🏆 Confirmed Winner (Singapore 2025): {winner_code}")

# Label winner before encoding
driver_stats['Winner'] = (driver_stats['Driver'].str.upper() == winner_code.upper()).astype(int)

# Encode driver column for ML
driver_stats_encoded = pd.get_dummies(driver_stats, columns=['Driver'], drop_first=True)

# base Model (pre-augmentation)
X = driver_stats_encoded.drop('Winner', axis=1)
y = driver_stats_encoded['Winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_pre = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=5)
model_pre.fit(X_train, y_train)

print("\n📊 Classification Report (Pre-Augmentation):")
print(classification_report(y_test, model_pre.predict(X_test), zero_division=0))
# predicting win probabilities (Pre)
probs_pre = model_pre.predict_proba(X)[:, 1]
pred_df_pre = pd.DataFrame({
    'Driver': driver_stats['Driver'],
    'Win_Probability': probs_pre
}).sort_values(by='Win_Probability', ascending=False)

print("\n🏎️ Top 3 Predicted Drivers (Pre-Augmentation):")
print(pred_df_pre.head(3).to_string(index=False))

# performing data augmentation for balance
X_aug, y_aug = X.copy(), y.copy()

if y_aug.sum() == 1:
    winner_row = X.loc[y == 1].copy()
    for i in range(3):
        new_row = winner_row.copy()
        for col in new_row.columns:
            if np.issubdtype(new_row[col].dtype, np.number):
                new_row[col] *= np.random.uniform(0.97, 1.03)
        X_aug = pd.concat([X_aug, new_row], ignore_index=True)
        y_aug = pd.concat([y_aug, pd.Series([1])], ignore_index=True)
    print("\n✅ Added synthetic winner-like samples for balance.")
 
#post-Augmentation Model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_aug, y_aug, test_size=0.3, random_state=42)

model_post = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=5)
model_post.fit(X_train2, y_train2)

print("\n📊 Classification Report (Post-Augmentation):")
print(classification_report(y_test2, model_post.predict(X_test2), zero_division=0))

#predicting win probabilities (post)
probs_post = model_post.predict_proba(X_aug)[:, 1]
pred_df_post = pd.DataFrame({
    'Driver': driver_stats['Driver'].tolist() + [f"{winner_code}_aug{i+1}" for i in range(len(probs_post)-len(driver_stats))],
    'Win_Probability': probs_post
}).sort_values(by='Win_Probability', ascending=False)

# filtering the real drivers only for display
pred_df_post_real = pred_df_post[~pred_df_post['Driver'].str.contains('_aug')]

print("\n🏎️ Top 3 Predicted Drivers (Post-Augmentation) – Austin GP 2025:")
print(pred_df_post_real.head(3).to_string(index=False))

# visualization: pre vs post augmentation Comparison
merged = pred_df_pre.merge(pred_df_post_real, on="Driver", suffixes=("_Pre", "_Post"))
merged = merged.sort_values("Win_Probability_Post", ascending=False)

plt.figure(figsize=(9,5))
plt.barh(merged["Driver"], merged["Win_Probability_Pre"], color="lightcoral", label="Pre-Augmentation")
plt.barh(merged["Driver"], merged["Win_Probability_Post"], color="skyblue", alpha=0.7, label="Post-Augmentation")
plt.gca().invert_yaxis()
plt.xlabel("Predicted Win Probability")
plt.title("Predicted Austin GP 2025 Winner Probabilities\nPre vs Post Augmentation")
plt.legend()
plt.tight_layout()
plt.show()