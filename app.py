from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
import sqlite3

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path='best_neural_network_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = load('scaler.joblib')
label_encoder = load('label_encoder.joblib')

# Define rugby positions
positions = label_encoder.classes_

# Connect to the database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Recommendation logic using the neural network
@app.route('/recommend', methods=['POST'])
def recommend():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    sprint = float(request.form['sprint'])

    # Validate the input
    if height < 166 or height > 210 or weight < 80 or weight > 170 or sprint < 1.5 or sprint > 2.5:
        return "Input height must be between 166 cm and 210 cm, weight must be between 80 kg and 170 kg, and sprint time between 1.5 and 2.5 seconds."

    # Prepare the input data for the neural network
    input_data = pd.DataFrame([[height, weight, sprint]], columns=['Height (cm)', 'Weight (kg)', '10m Sprint Time (s)'])
    input_data_scaled = scaler.transform(input_data)
    
    # Get predictions from the model
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get the top 3 recommended positions
    top_indices = predictions.argsort()[-3:][::-1]
    recommended_positions = [{'name': positions[i], 'probability': predictions[i]} for i in top_indices]

    return render_template('recommend.html', positions=recommended_positions)

# Route to display position details
@app.route('/position_details', methods=['POST'])
def position_details():
    selected_position = request.form['selected_position']

    # Query the database to get details about the selected position
    conn = get_db_connection()
    position_details = conn.execute('SELECT * FROM position_details WHERE Position = ?', (selected_position,)).fetchone()
    conn.close()

    if position_details is None:
        return "Position details not found."

    return render_template('position_details.html', position_details=position_details)

# Route to display physical training details
@app.route('/physical_training/<position>', methods=['GET'])
def physical_training(position):
    conn = get_db_connection()
    
    # Query to get position ID and details
    position_row = conn.execute('SELECT * FROM positions WHERE Position = ?', (position,)).fetchone()
    
    if not position_row:
        return "Position not found."
    
    # Fetch training details using the position ID
    position_id = position_row['id']
    training_details = conn.execute('SELECT * FROM physical_training WHERE position_id = ?', (position_id,)).fetchone()
    conn.close()

    if training_details is None:
        return "Physical training details not found."

    # Pass the position name to the template
    return render_template('physical_training.html', training_details=training_details, position_name=position)

# Route to display technical drills
@app.route('/technical_drills/<position>', methods=['GET'])
def technical_drills(position):
    conn = get_db_connection()
    position_id = conn.execute('SELECT id FROM positions WHERE Position = ?', (position,)).fetchone()['id']
    drill_details = conn.execute('SELECT * FROM technical_drills WHERE position_id = ?', (position_id,)).fetchall()
    conn.close()

    if not drill_details:
        return "Technical drills not found."

    # Prepare drill information for the template
    drill_info = []
    for drill in drill_details:
        # Iterate over drill/video/url combinations
        for i in range(1, 4):  # Assuming three sets of drills, videos, and URLs
            drill_name = drill[f'drill{i}']
            video_link = drill[f'video{i}']
            url_link = drill[f'url{i}']
            
            # Transform video link to embed format
            if video_link and video_link.lower() != 'none':
                # Extract video ID
                if "watch?v=" in video_link:
                    video_id = video_link.split('watch?v=')[-1].split('&')[0]  # Get the video ID before any '&'
                elif "youtu.be/" in video_link:
                    video_id = video_link.split('youtu.be/')[-1].split('?')[0]  # Handle shortened URLs

                # Construct embed URL
                video_link = f"https://www.youtube.com/embed/{video_id}"

            # Only add if the drill name is valid (not 'None' or empty)
            if drill_name and drill_name.lower() != 'none':
                drill_info.append({
                    'drill': drill_name,
                    'video': video_link if video_link.lower() != 'none' else None,
                    'url': url_link if url_link.lower() != 'none' else None
                })

    return render_template('technical_drills.html', drill_info=drill_info, position_name=position)



if __name__ == '__main__':
    app.run(debug=True)
