import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loads the CSV file
file_path = 'GPT Heights and Weights.csv'
data = pd.read_csv(file_path)

blindside_flanker_avg_weight = data[data['Position'] == 'Blindside Flanker']['Weight (kg)'].mean()
openside_flanker_avg_weight = data[data['Position'] == 'Openside Flanker']['Weight (kg)'].mean()

print(f"Average weight of Blindside Flankers: {blindside_flanker_avg_weight} kg")
print(f"Average weight of Openside Flankers: {openside_flanker_avg_weight} kg")
flanker_avg_weight_difference=blindside_flanker_avg_weight-openside_flanker_avg_weight
print(f"Difference: {flanker_avg_weight_difference} kg")
#Combines the positions
data['Position'] = data['Position'].replace({
    'Loosehead Prop': 'Prop', 
    'Tighthead Prop': 'Prop', 
    'Blindside Flanker': 'Flanker', 
    'Openside Flanker': 'Flanker'
})

#Ensures only numeric columns are used for averaging
numeric_data = data[['Height (cm)', 'Weight (kg)', 'Position']]

#Calculates the averages for each combined position
position_averages_combined = numeric_data.groupby('Position').mean().reset_index()

#Sets up the plot for individual players
plt.figure(figsize=(12, 8))

#Creates the scatter plot for individual players
scatter_plot = sns.scatterplot(data=data, x='Height (cm)', y='Weight (kg)', hue='Position', palette='hls', s=100)

#Sets plot title and labels for individual players
scatter_plot.set_title('Rugby Players Height and Weight by Position (Combined)')
scatter_plot.set_xlabel('Height (cm)')
scatter_plot.set_ylabel('Weight (kg)')

#Displays the plot for individual players
plt.legend(title='Position')
plt.show()

#Sets up the plot for position averages
plt.figure(figsize=(12, 8))

#Creates the scatter plot for position averages
average_plot = sns.scatterplot(data=position_averages_combined, x='Height (cm)', y='Weight (kg)', hue='Position', palette='hls', 
                               s=200, markers='D', edgecolor='black', linewidth=1.5)

#Sets plot title and labels for position averages
average_plot.set_title('Average Height and Weight of Rugby Players by Position')
average_plot.set_xlabel('Height (cm)')
average_plot.set_ylabel('Weight (kg)')

#Displays the plot for position averages
plt.legend(title='Position')
plt.show()

#Combines positions into forwards and backs
forward_positions = ['Prop', 'Hooker', 'Lock', 'Flanker', 'Number 8']
back_positions = ['Scrum-half', 'Fly-half', 'Centre', 'Wing', 'Fullback']

data['Category'] = data['Position'].apply(lambda x: 'Forward' if x in forward_positions else 'Back')

#Ensures only numeric columns are used for averaging
numeric_data_category = data[['Height (cm)', 'Weight (kg)', 'Category']]

#Calculates the averages for forwards and backs
category_averages = numeric_data_category.groupby('Category').mean().reset_index()

#Sets up the bar plot for height
plt.figure(figsize=(12, 8))
height_plot = sns.barplot(data=category_averages, x='Category', y='Height (cm)', palette='hls')
height_plot.set_title('Average Height of Rugby Players by Category')
height_plot.set_xlabel('Position')
height_plot.set_ylabel('Height (cm)')
height_plot.set_ylim(0, category_averages['Height (cm)'].max() + 20)  # Extend the y-axis limit

#Displays the plot for height
plt.show()

#Sets up the bar plot for weight
plt.figure(figsize=(12, 8))
weight_plot = sns.barplot(data=category_averages, x='Category', y='Weight (kg)', palette='hls')
weight_plot.set_title('Average Weight of Rugby Players by Category')
weight_plot.set_xlabel('Position')
weight_plot.set_ylabel('Weight (kg)')
weight_plot.set_ylim(0, category_averages['Weight (kg)'].max() + 20)  # Extend the y-axis limit

#Displays the plot for weight
plt.show()


