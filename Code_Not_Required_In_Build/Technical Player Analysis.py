import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#Loads the Excel file and the specific sheet
file_path = 'Six Nations.xlsx'
df_stats = pd.read_excel(file_path, sheet_name='All player stats')

#Rename 'Back-row' to 'Flanker, No. 8' in the 'Position Detailed' column, Eases later Issue of 'Back-row' not being included in 'Flanker', and 'No. 8' Calculations
df_stats['Position Detailed'] = df_stats['Position Detailed'].str.replace('Back-row', 'Flanker, No. 8', regex=False)

#Define the positions and their corresponding metrics
position_metrics = {
    'Prop': ['Scrum Score'],
    'Prop, Lock': ['Scrum Score', 'LineOut Take', 'LineOut Steal'],
    'Hooker': ['Lineout Score'],
    'Lock': ['LineOut Take', 'LineOut Steal'],
    'Lock, Flanker': ['LineOut Take', 'LineOut Steal', 'Tackle Turnover'],
    'Flanker': ['Tackle Turnover'],
    'Flanker, No. 8': ['Tackle Turnover', 'Meters Per Carry'],
    'No. 8': ['Meters Per Carry'],
    'Scrum-half': ['Pass Complete', 'Territorial Kick Meters'],
    'Fly-half': ['Influence', 'Goal Success'],
    'Fullback, Fly-half': ['Influence', 'Goal Success', 'Try Saver', 'Defensive Catch', 'Mark', 'Territorial Kick Meters'],
    'Fullback, Centre, Fly-half': ['Break', 'Influence', 'Goal Success', 'Try Saver', 'Defensive Catch', 'Mark', 'Territorial Kick Meters'],
    'Centre': ['Break'],
    'Fullback, Centre': ['Break', 'Try Saver', 'Defensive Catch', 'Mark', 'Territorial Kick Meters'],
    'Wing, Centre': ['Influence', 'Attacking', 'Try Saver', 'Break'],
    'Wing': ['Influence', 'Attacking', 'Try Saver'],
    'Fullback, Wing': ['Influence', 'Attacking', 'Try Saver', 'Defensive Catch', 'Mark', 'Territorial Kick Meters'],
    'Fullback': ['Try Saver', 'Defensive Catch', 'Mark', 'Territorial Kick Meters']
}

#Metrics that need to be divided by Number of Six Nations Matches Played
metrics_to_adjust = {
    'Influence': True,
    'Attacking': True,
    'Territorial Kick Meters': False, #Already per Game
    'Try Saver': True,
    'Scrum Score': True,
    'Lineout Score': True,
    'Tackle Turnover': True,
    'LineOut Take': True,
    'LineOut Steal': True,
    'Pass Complete': False, #Already per Game
    'Mark': True,
    'Defensive Catch': True,
    'Goal Success': True,
    'Break': True,
    'Meters Per Carry': False  #Already per carry
}

#New names for metrics, for Visualisation
metric_rename = {
    'Influence': 'Number of Influential Moments',
    'Attacking': 'Attacking Actions',
    'Territorial Kick Meters': 'Territorial Kick Meters',
    'Try Saver': 'Try Saving Actions',
    'Scrum Score': 'Scrums Won',
    'Lineout Score': 'Lineouts Won',
    'Tackle Turnover': 'Tackle Turnovers',
    'LineOut Take': 'Lineout Takes',
    'LineOut Steal': 'Lineout Steals',
    'Pass Complete': 'Completed Passes',
    'Mark': 'Marks Called',
    'Defensive Catch': 'Successful Defensive Catches',
    'Goal Success': 'Successful Conversions and Penalties',
    'Break': 'Line Breaks',
    'Meters Per Carry': 'Meters Gained Per Carry'
}

#Initialises dictionaries to store the results
average_stats = {}
std_dev_stats = {}

#Function to adjust metrics by dividing by 'Six Nations Matches'
def adjust_metrics(data):
    for metric, needs_adjustment in metrics_to_adjust.items():
        if metric in data.columns and needs_adjustment:
            data[metric] = data[metric] / data['Six Nations Matches']
    return data

#Function to remove outliers using the IQR method
def remove_outliers(data):
    for column in data.columns:
        if column != 'Six Nations Matches':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]
    return data

#Function to gather relevant data for a given position
def gather_relevant_data(position, metrics):
    relevant_data = pd.DataFrame()
    for pos_key, pos_metrics in position_metrics.items():
        if position in pos_key.split(', '):
            position_data = df_stats[df_stats['Position Detailed'].str.contains(pos_key, na=False)]
            if not position_data.empty:
                relevant_data = pd.concat([relevant_data, position_data[['Six Nations Matches'] + metrics]], ignore_index=True)
    return remove_outliers(adjust_metrics(relevant_data))

#Calculates average and standard deviation for each position
for position, metrics in position_metrics.items():
    relevant_data = gather_relevant_data(position, metrics)
    
    if not relevant_data.empty:
        # Calculate mean and standard deviation, excluding NaN values
        average_stats[position] = relevant_data[metrics].mean(skipna=True)
        std_dev_stats[position] = relevant_data[metrics].std(skipna=True)
    else:
        # Handle positions with no data
        average_stats[position] = pd.Series([np.nan] * len(metrics), index=metrics)
        std_dev_stats[position] = pd.Series([np.nan] * len(metrics), index=metrics)

#Converts the results into DataFrames for easier plotting
average_df = pd.DataFrame(average_stats).T
std_dev_df = pd.DataFrame(std_dev_stats).T

#Function to create radar charts with different scales for each metric
def create_radar_chart(data, title):
    #Rename the columns for readability
    data = data.rename(columns=metric_rename)
    
    labels = data.columns
    num_vars = len(labels)
    
    #Computes angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    values = data.iloc[0].tolist()
    values += values[:1]
    
    #Determines the maximum value for each metric
    max_value = max([math.ceil(data[label].max(skipna=True) / 5.0) * 5 for label in labels])
    min_value = 0  # Always start from 0
    y_ticks = np.linspace(min_value, max_value, num=6)
    y_tick_labels = [f'{int(x)}' for x in y_ticks]
    ax.set_ylim(min_value, max_value)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    plt.title(title, size=20, color='red', y=1.1)
    plt.show()

#Function to create box plots
def create_box_plot(data, title):
    #Rename the columns for readability
    data = data.rename(columns=metric_rename)
    
    #Filters out rows with NaN values before plotting
    filtered_data = data.dropna()
    if not filtered_data.empty:
        max_value = math.ceil(filtered_data.max().max(skipna=True) / 5.0) * 5
        min_value = math.floor(filtered_data.min().min(skipna=True) / 5.0) * 5
        ax = filtered_data.plot(kind='box', figsize=(10, 6))
        ax.set_ylim(min_value, max_value)
        ax.set_yticks(np.linspace(min_value, max_value, num=6))
        ax.set_yticklabels([f'{int(x)}' for x in np.linspace(min_value, max_value, num=6)])
        plt.title(title, size=20)
        plt.ylabel('Value per Game')
        plt.show()

#Creates appropriate charts for each position
for position, metrics in position_metrics.items():
    relevant_data = gather_relevant_data(position, metrics)
    if not relevant_data.empty:
        create_box_plot(relevant_data[metrics], f"{position}'s Metrics per Game")
        if len(metrics) >= 3:
            if position == 'Fullback':
                metrics_for_radar = [metric for metric in metrics if metric != 'Territorial Kick Meters']
                create_radar_chart(relevant_data[metrics_for_radar].mean().to_frame().T, f"{position}'s Stats per Game")
                create_box_plot(relevant_data[['Territorial Kick Meters'], ['Try Saver'], ['Defensive Catch'], ['Mark']], f"{position}'s Metrics per Game")
            else:
                create_radar_chart(relevant_data[metrics].mean().to_frame().T, f"{position}'s Stats per Game")

