import pandas as pd

#creating a small dataset using a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 28],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'Salary': [50000, 60000, 70000, 65000, 52000]}

#convert the dictionary into a DataFrame
df = pd.DataFrame(data)

#displaying the first few rows
print(df.head())

#calculating the average salary by department
avg_salary = df.groupby('Department')['Salary'].mean()
print("\nAverage Salary by Department:")
print(avg_salary)

#filtering employees older than 30
older_employees = df[df['Age'] > 30]
print("\nEmployees older than 30:")
print(older_employees)
