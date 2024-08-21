import pandas as pd

#Loads the CSV file
file_path = '/Users/tomgo/Documents/Keele/Semester 3/Recommender System/GPT Heights and Weights.csv'
data = pd.read_csv(file_path)

#Groups by 'Position' and calculate descriptive statistics for 'Height (cm)' and 'Weight (kg)'
descriptive_stats = data.groupby('Position').agg({
    'Height (cm)': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'Weight (kg)': ['mean', 'median', 'std', 'min', 'max', 'count']
}).reset_index()

#Renames columns for clarity
descriptive_stats.columns = [
    'Position',
    'Height Mean', 'Height Median', 'Height Std', 'Height Min', 'Height Max', 'Height Count',
    'Weight Mean', 'Weight Median', 'Weight Std', 'Weight Min', 'Weight Max', 'Weight Count'
]

#Displays the descriptive statistics
print(descriptive_stats)
#Saves the descriptive statistics to a new CSV file
descriptive_stats.to_csv('Descriptive Statistics of Heights and Weights.csv', index=False)
