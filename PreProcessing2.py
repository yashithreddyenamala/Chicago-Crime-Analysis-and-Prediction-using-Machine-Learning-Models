import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# No need for 'from torch import torch.nn.BCELoss'
# Correct import statement for BCELoss
from torch.nn import BCELoss

import matplotlib.pyplot as plt

dataset = './Datasets/Crimes.csv'


if os.path.exists(dataset):
    crimes = pd.read_csv(dataset)
    crimes = crimes.dropna() 
    
else:
    print("Dataset not found")


crimes = crimes[~crimes['Year'].isin([2001, 2002])]

# Assuming your original DataFrame is named 'df'
crimes_smaller = crimes.iloc[::10, :].copy()

crimes_smaller = crimes_smaller.drop(['Location', 'Latitude', 'Longitude', 'IUCR', 'FBI Code', 'Beat'], axis=1)
crimes = crimes.drop(['Location', 'Latitude', 'Longitude', 'IUCR', 'FBI Code', 'Beat'], axis=1)

grouped_data_smaller_year = crimes_smaller.groupby('Year')
grouped_data_year = crimes.groupby('Year')

# Calculate the number of entries for each year
year_counts_smaller = grouped_data_smaller_year.size()
year_counts = grouped_data_year.size()

# Normalizing directly on the Pandas Series (preserving the index)
year_counts_normalized = year_counts / year_counts.sum()
year_counts_smaller_normalized = year_counts_smaller / year_counts_smaller.sum()

years_array = np.array(year_counts.index)

plt.figure(figsize=(20, 10))

# Plot the full dataset normalized counts
plt.plot(year_counts_normalized.index, year_counts_normalized.values, marker='o', linestyle='-', color='b', label='Full Dataset')

# Plot the smaller dataset normalized counts
plt.plot(year_counts_smaller_normalized.index, year_counts_smaller_normalized.values, marker='o', linestyle='-', color='r', label='Smaller Dataset')

# Labeling the plot
plt.xticks(years_array)
plt.xlabel('Year')
plt.ylabel('Normalized Number of Entries')
plt.title('Normalized Number of Entries Per Year (Full vs. Smaller Dataset)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

grouped_data_smaller_district = crimes_smaller.groupby('District')
grouped_data_district = crimes.groupby('District')

# Calculate the number of entries for each district
district_counts_smaller = grouped_data_smaller_district.size()
district_counts = grouped_data_district.size()

district_counts_normalized = district_counts / district_counts.sum()
district_counts_smaller_normalized = district_counts_smaller / district_counts_smaller.sum()

district_array = np.array(district_counts.index)
plt.figure(figsize=(10, 6))

plt.plot(district_counts_normalized.index, district_counts_normalized.values, marker='o', linestyle='-', color='b', label='Full Dataset')
plt.plot(district_counts_smaller_normalized.index, district_counts_smaller_normalized.values, marker='o', linestyle='-', color='r', label='Smaller Dataset')

plt.xticks(district_array)
plt.xlabel('District')
plt.ylabel('Normalized Number of Entries')
plt.title('Normalized Number of Entries Per District (Full vs. Smaller Dataset)')
plt.legend()
plt.grid(True)

plt.show()

grouped_data_smaller_primary_type = crimes_smaller.groupby('Primary Type')
grouped_data_primary_type = crimes.groupby('Primary Type')

primary_type_counts_smaller = grouped_data_smaller_primary_type.size()
primary_type_counts = grouped_data_primary_type.size()

primary_type_counts_normalized = primary_type_counts / primary_type_counts.sum()
primary_type_counts_smaller_normalized = primary_type_counts_smaller / primary_type_counts_smaller.sum()

primary_type_array = np.array(primary_type_counts.index)

plt.figure(figsize=(20, 10))

plt.plot(primary_type_counts_normalized.index, primary_type_counts_normalized.values, marker='o', linestyle='-', color='b', label='Full Dataset')
plt.plot(primary_type_counts_smaller_normalized.index, primary_type_counts_smaller_normalized.values, marker='o', linestyle='-', color='r', label='Smaller Dataset')

plt.xticks(primary_type_array, rotation=90)
plt.xlabel('Primary Type')
plt.ylabel('Normalized Number of Entries')
plt.title('Normalized Number of Entries Per Primary Type (Full vs. Smaller Dataset)')
plt.legend()
plt.grid(True)

plt.show()

arrest_counts = crimes['Arrest'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(7, 7))
plt.pie(arrest_counts, labels=['No Arrest', 'Arrest'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)

# Title for the pie chart
plt.title('Arrest vs No Arrest')

arrest_counts_smaller = crimes_smaller['Arrest'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(arrest_counts_smaller, labels=['No Arrest', 'Arrest'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)

plt.title('Arrest vs No Arrest (Smaller Dataset)')

plt.show()

print(crimes_smaller.dtypes)

total_entries_per_year = crimes.groupby('Year').size()

# Group by Year for entries where Arrest is True
arrest_true_per_year = crimes[crimes['Arrest']].groupby('Year').size()

# Calculate percentage of Arrest cases
arrest_percentage_per_year = (arrest_true_per_year / total_entries_per_year) * 100

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(arrest_percentage_per_year.index, arrest_percentage_per_year.values, marker='o', linestyle='-', color='blue')

# Adding labels and title
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Arrests (%)', fontsize=14)
plt.title('Percentage of Arrest Cases Per Year', fontsize=16)
plt.grid(True)

# Show plot
plt.show()

total_entries_per_district = crimes.groupby('District').size()

# Group by District for entries where Arrest is True
arrest_true_per_district = crimes[crimes['Arrest']].groupby('District').size()

# Calculate percentage of Arrest cases for each district
arrest_percentage_per_district = (arrest_true_per_district / total_entries_per_district) * 100

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(arrest_percentage_per_district.index, arrest_percentage_per_district.values, marker='o', linestyle='-', color='green')

# Adding labels and title
plt.xlabel('District', fontsize=14)
plt.ylabel('Percentage of Arrests (%)', fontsize=14)
plt.title('Percentage of Arrest Cases Per District', fontsize=16)
plt.grid(True)

# Show plot
plt.show()

crimes_smaller.to_csv('second_dataset.csv', index=False)
