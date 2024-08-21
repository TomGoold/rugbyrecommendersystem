import pandas as pd

#Loads the excel file
file_path = '/Users/tomgo/Documents/Keele/Semester 3/Raw Data/Six Nations.xlsx'
sheet_name = 'All player stats'

#Reads the sheet into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

#Function to remove outliers using IQR method
def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]

#Function to calculate descriptive statistics for a given position and score column
def calculate_statistics(df, positions, score_column, divide_by_matches=True):
    filtered_df = df[df['Position Detailed'].isin(positions)]
    scores = filtered_df[score_column]
    
    if divide_by_matches and score_column not in ['Meters Per Carry', 'Territorial Kick Meters']:
        scores = scores / filtered_df['Six Nations Matches']
    
    empty_values_count = scores.isna().sum()
    scores = scores.dropna()
    
    #Removes outliers
    scores = remove_outliers(scores)
    
    stats = scores.describe()
    return {
        "metric": score_column,
        "positions": ', '.join(positions),
        "empty_values_count": empty_values_count,
        "count": stats['count'],
        "mean": stats['mean'],
        "std": stats['std'],
        "min": stats['min'],
        "25%": stats['25%'],
        "50%": stats['50%'],
        "75%": stats['75%'],
        "max": stats['max']
    }

#Defines the positions and score columns for each metric
metrics = {
    "Scrum Score for Props": (['Prop', 'Prop, Lock'], 'Scrum Score'),
    "Lineout Score for Hookers": (['Hooker'], 'Lineout Score'),
    "Lineout Take for Locks": (['Lock', 'Lock, Flanker', 'Lock, Back-row'], 'LineOut Take'),
    "Lineout Steal for Locks": (['Lock', 'Lock, Flanker', 'Lock, Back-row'], 'LineOut Steal'),
    "Tackle Turnover for Flankers": (['Lock, Flanker', 'Flanker, No. 8', 'Flanker', 'Lock, Back-row', 'Back-row'], 'Tackle Turnover'),
    "Meters Per Carry for No. 8": (['Back-row', 'Flanker, No. 8', 'No. 8', 'Lock, Back-row'], 'Meters Per Carry'),
    "Territorial Kick Meters for Scrum Half": (['Scrum-half'], 'Territorial Kick Meters'),
    "Pass Complete for Scrum Half": (['Scrum-half'], 'Pass Complete'),
    "Influence for Fly-half": (['Fly-half', 'Fullback, Fly-half', 'Full-back, Centre, Fly-half'], 'Influence'),
    "Goal Success for Fly-half": (['Fly-half', 'Fullback, Fly-half', 'Full-back, Centre, Fly-half'], 'Goal Success'),
    "Break for Centres": (['Centre', 'Fullback, Centre', 'Wing, Centre', 'Fullback, Centre, Fly-half'], 'Break'),
    "Snaffle for Centres": (['Centre', 'Fullback, Centre', 'Wing, Centre', 'Fullback, Centre, Fly-half'], 'Snaffle'),
    "Influence for Wingers": (['Wing', 'Fullback, Wing'], 'Influence'),
    "Attacking for Wingers": (['Wing', 'Fullback, Wing'], 'Attacking'),
    "Try Saver for Wingers": (['Wing', 'Fullback, Wing'], 'Try Saver'),
    "Try Saver for Fullbacks": (['Fullback', 'Fullback, Wing', 'Fullback, Fly-half', 'Fullback, Centre, Fly-half', 'Fullback, Centre'], 'Try Saver'),
    "Defensive Catch for Fullbacks": (['Fullback', 'Fullback, Wing', 'Fullback, Fly-half', 'Fullback, Centre, Fly-half', 'Fullback, Centre'], 'Defensive Catch'),
    "Mark for Fullbacks": (['Fullback', 'Fullback, Wing', 'Fullback, Fly-half', 'Fullback, Centre, Fly-half', 'Fullback, Centre'], 'Mark'),
    "Territorial Kick Meters for Fullbacks": (['Fullback', 'Fullback, Wing', 'Fullback, Fly-half', 'Fullback, Centre, Fly-half', 'Fullback, Centre'], 'Territorial Kick Meters')
}

#List to store all results
results = []

#Loops through each metric and calculate the statistics
for metric, (positions, score_column) in metrics.items():
    result = calculate_statistics(df, positions, score_column, divide_by_matches=(score_column not in ['Meters Per Carry', 'Territorial Kick Meters']))
    result['metric_name'] = metric
    results.append(result)

#Converts results to a DataFrame
results_df = pd.DataFrame(results)

#Saves results to a new CSV file
output_file_path = '/Users/tomgo/Documents/Keele/Semester 3/Raw Data/technical_analysis.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")
