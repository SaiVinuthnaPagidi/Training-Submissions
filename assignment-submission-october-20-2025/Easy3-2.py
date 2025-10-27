import numpy as np #importing numpy package

#heart rates of five patients after exercise
heart_rates = np.array([78, 85, 90, 95, 88])

#basic statistics
average_rate = np.mean(heart_rates)
max_rate = np.max(heart_rates)
min_rate = np.min(heart_rates)
normalized = heart_rates / np.max(heart_rates)

#printing the results
print("Heart Rates:", heart_rates)
print("Average:", average_rate)
print("Max:", max_rate)
print("Min:", min_rate)
print("Normalized Values:", normalized)
