import pandas as pd #importing packages
df = pd.read_csv("/Users/vinuthnapagidi/Downloads/covid_indonesia_data.csv") #loading the dataset
print (df.head(5)) #displaying the first five rows of the dataset
print (df.tail(5)) #displaying the last five rows of the dataset
print("\nDataset shape:", df.shape) #shape of the dataset
columns_to_check = ["New Cases", "New Deaths", "New Recovered"] #selecting columns to analyse 
from statistics import mean 
# Calculate totals and averages using built-in functions
total_cases = df["New Cases"].sum()
average_cases = df["New Cases"].mean()
total_deaths = df["New Deaths"].sum()
average_deaths = df["New Deaths"].mean()

# Display results
print("\n=== COVID-19 Summary ===")
print(f"Total New Cases: {total_cases:,}")
print(f"Average Daily New Cases: {average_cases:,.2f}")
print(f"Total New Deaths: {total_deaths:,}")
print(f"Average Daily New Deaths: {average_deaths:,.2f}")

# Loop through each column
for col in columns_to_check:
    # Convert column to a list (ignoring missing values)
    values = df[col].dropna().tolist()

    # Calculate total using a loop
    total = 0
    for v in values:
        total += v

 # Calculate mean using the built-in mean() function
    avg = mean(values)

    print(f"{col}: Total = {total:,}, Average = {avg:,.2f}")