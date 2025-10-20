import pandas as pd #importing pandas
#uploading datasets
races = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/races.csv")
results = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/results.csv")
drivers = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/drivers.csv")
constructors = pd.read_csv("/Users/vinuthnapagidi/Downloads/archive-2/constructors.csv")

# merging results with driver and constructor info
merged = results.merge(drivers, on="driverId").merge(constructors, on="constructorId").merge(races, on="raceId")

# filter for 2024 season
season_2024 = merged[merged["year"] == 2024]

# counting wins per driver
driver_wins = season_2024[season_2024["positionOrder"] == 1]["surname"].value_counts().to_dict()

# calculating the average points per constructor
team_points = season_2024.groupby("name_x")["points"].mean().sort_values(ascending=False).head(5).to_dict()

print("Driver Wins (2024):", driver_wins)
print("\nTop 5 Constructors by Average Points (2024):", team_points) #printing the top 5 teams 