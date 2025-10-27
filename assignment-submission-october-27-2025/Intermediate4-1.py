import pandas as pd

# Load the uploaded dataset
file_path = "/Users/vinuthnapagidi/Downloads/BehavioralRF–Vision&Eye2025.csv"
df = pd.read_csv(file_path)

#displaying basic information about the dataset
df.info() 

#displaying the first few rows
print("\nFIRST 5 ROWS")
print(df.head())

#displaying statistical summary for numeric columns
print("\nSUMMARY STATISTICS")
print(df.describe())

#checking for missing values in key column
print("\nMissing values in 'Data_Value':", df['Data_Value'].isnull().sum())

#filtering for the most recent year (2022)
df_2022 = df[df["YearStart"] == 2022]

#group by state and calculate the mean vision difficulty (%)
state_avg = (
    df_2022.groupby("LocationDesc")["Data_Value"]
    .mean()
    .sort_values(ascending=False))

#displaying results
print("\nTOP 5 STATES WITH HIGHEST AVERAGE VISION DIFFICULTY")
print(state_avg.head(5))

print("\nBOTTOM 5 STATES WITH LOWEST AVERAGE VISION DIFFICULTY")
print(state_avg.tail(5))

# displaying summary statistics for 2022 data values
print("\nSUMMARY STATISTICS FOR 2022 DATA VALUES")
print(df_2022["Data_Value"].describe())