import sqlite3
import pandas as pd

#Loads in the CSV files
file_path = 'Descriptive Statistics of Heights and Weights.csv'
data = pd.read_csv(file_path)

technical_file_path = 'technical_analysis.csv'

#Reads the technical analysis file with the specified encoding
try:
    technical_data = pd.read_csv(technical_file_path, encoding='utf-8')
except UnicodeDecodeError:
    technical_data = pd.read_csv(technical_file_path, encoding='latin1')

#Checks if 'Position' column values are strings and handles missing values
technical_data['Position'] = technical_data['Position'].astype(str)

#Converts all drills, videos, and URLs to strings and handles missing values
drill_columns = ['Drill 1', 'Drill 2', 'Drill 3']
video_columns = ['Video 1', 'Video 2', 'Video 3']
url_columns = ['URL 1', 'URL 2', 'URL 3']

for column in drill_columns + video_columns + url_columns:
    technical_data[column] = technical_data[column].fillna('').astype(str)

#Expands the technical_data to have individual positions
expanded_technical_data = []

for _, row in technical_data.iterrows():
    if pd.isna(row['Position']) or row['Position'].strip() == '':
        continue
    positions_list = row['Position'].split(', ')
    for pos in positions_list:
        expanded_technical_data.append({
            'Position': pos,
            'Technical Attributes': row['metric_name'],
            'Mean': row['mean'],
            'Drill 1': row['Drill 1'],
            'Video 1': row['Video 1'],
            'URL 1': row['URL 1'],
            'Drill 2': row['Drill 2'],
            'Video 2': row['Video 2'],
            'URL 2': row['URL 2'],
            'Drill 3': row['Drill 3'],
            'Video 3': row['Video 3'],
            'URL 3': row['URL 3']
        })

expanded_technical_df = pd.DataFrame(expanded_technical_data)

#Removes rows with NaN values in Technical Attributes or Mean
expanded_technical_df = expanded_technical_df.dropna(subset=['Technical Attributes', 'Mean'])

#Combines the physical and technical attributes
combined_data = data.merge(expanded_technical_df, on='Position', how='left')

#Connects to the database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

#Creates tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Position TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS position_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    min_height REAL,
    max_height REAL,
    avg_height REAL,
    min_weight REAL,
    max_weight REAL,
    avg_weight REAL,
    FOREIGN KEY(position_id) REFERENCES positions(id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS position_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Position TEXT,
    height REAL,
    weight REAL,
    technical_attributes TEXT,
    mean REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS physical_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    squat REAL,
    bench_press REAL,
    exercises TEXT,
    FOREIGN KEY(position_id) REFERENCES positions(id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS technical_drills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    drill1 TEXT,
    video1 TEXT,
    url1 TEXT,
    drill2 TEXT,
    video2 TEXT,
    url2 TEXT,
    drill3 TEXT,
    video3 TEXT,
    url3 TEXT,
    FOREIGN KEY(position_id) REFERENCES positions(id)
)
''')

#Inserts data into positions table
positions = combined_data[['Position']].drop_duplicates().reset_index(drop=True)
positions['id'] = positions.index + 1  #Creates a unique ID for each position
positions.to_sql('positions', conn, if_exists='replace', index=False)

#Inserts data into position_requirements table
position_requirements = combined_data[['Position', 'Height Min', 'Height Max', 'Height Mean', 'Weight Min', 'Weight Max', 'Weight Mean']]
position_requirements = position_requirements.merge(positions, on='Position')[['id', 'Position', 'Height Min', 'Height Max', 'Height Mean', 'Weight Min', 'Weight Max', 'Weight Mean']]
position_requirements.columns = ['position_id', 'Position', 'min_height', 'max_height', 'avg_height', 'min_weight', 'max_weight', 'avg_weight']
position_requirements.to_sql('position_requirements', conn, if_exists='replace', index=False)

#Inserts data into position_details table
position_details = combined_data[['Position', 'Height Mean', 'Weight Mean', 'Technical Attributes', 'Mean']].drop_duplicates().reset_index(drop=True)
position_details.columns = ['Position', 'height', 'weight', 'technical_attributes', 'mean']
position_details.to_sql('position_details', conn, if_exists='replace', index=False)

#Calculates and inserts physical training data
physical_training_data = []

for _, row in position_requirements.iterrows():
    avg_weight = row['avg_weight']
    squat_target = avg_weight * 2.0
    bench_press_target = avg_weight * 1.5
    physical_training_data.append({
        'position_id': row['position_id'],
        'squat': squat_target,
        'bench_press': bench_press_target,
        'exercises': 'Squats, Deadlifts, Bench Press, Power Cleans, Sprints, Push Press'  
    })

physical_training_df = pd.DataFrame(physical_training_data)
physical_training_df.to_sql('physical_training', conn, if_exists='replace', index=False)

#Inserts data into technical_drills table
technical_drills_data = []

for _, row in expanded_technical_df.iterrows():
    position_id = positions[positions['Position'] == row['Position']]['id'].values[0]  
    technical_drills_data.append({
        'position_id': position_id,
        'drill1': row['Drill 1'],
        'video1': row['Video 1'],
        'url1': row['URL 1'],
        'drill2': row['Drill 2'],
        'video2': row['Video 2'],
        'url2': row['URL 2'],
        'drill3': row['Drill 3'],
        'video3': row['Video 3'],
        'url3': row['URL 3'],
    })

technical_drills_df = pd.DataFrame(technical_drills_data)
technical_drills_df.to_sql('technical_drills', conn, if_exists='replace', index=False)

#Closes the connection
conn.close()

print("Database setup complete.")
