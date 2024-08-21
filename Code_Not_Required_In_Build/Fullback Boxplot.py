import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loads the data from the specified Excel sheet
file_path = 'Six Nations.xlsx'  # Adjust if necessary
data = pd.read_excel(file_path, sheet_name='All player stats')

#Filters players based on their 'Position Detailed'
positions = ['Fullback, Wing', 'Fullback, Centre, Fly-half', 'Fullback, Centre', 'Fullback, Fly-half', 'Fullback']
filtered_data = data[data['Position Detailed'].isin(positions)]

#Calculates the required metrics, normalising by 'Six Nations Matches' where necessary
filtered_data['Try Saver Norm'] = filtered_data['Try Saver'] / filtered_data['Six Nations Matches']
filtered_data['Defensive Catch Norm'] = filtered_data['Defensive Catch'] / filtered_data['Six Nations Matches']
filtered_data['Mark Norm'] = filtered_data['Mark'] / filtered_data['Six Nations Matches']

#Handling outliers using the IQR method
metrics = ['Try Saver Norm', 'Defensive Catch Norm', 'Mark Norm', 'Territorial Kick Meters']
Q1 = filtered_data[metrics].quantile(0.25)
Q3 = filtered_data[metrics].quantile(0.75)
IQR = Q3 - Q1
filtered_no_outliers = filtered_data[~((filtered_data[metrics] < (Q1 - 1.5 * IQR)) | (filtered_data[metrics] > (Q3 + 1.5 * IQR))).any(axis=1)]

#Creates box plots for the metrics
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(data=filtered_no_outliers[metrics[:-1]], ax=ax1)  # Excludes 'Territorial Kick Meters' for now
ax1.set_title('Fullback\'s Metrics per Game')
ax1.set_ylabel('Values per Game')
ax1.set_xticklabels(['Try Saving Actions', 'Successful Defensive Catches', 'Marks Called', 'Territorial Kick Meters'])

#Adding the secondary Y-axis for the 'Territorial Kick Meters'
ax2 = ax1.twinx()
sns.boxplot(data=filtered_no_outliers[['Territorial Kick Meters']], ax=ax2, color='red')
ax2.set_ylabel('Distance (meters)')

plt.show()
