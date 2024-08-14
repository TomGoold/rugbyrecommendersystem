import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from joblib import dump
import os
import itertools

#File Paths
train_file_path = '/Users/tomgo/Documents/Keele/Semester 3/Recommender System/GPT Heights and Weights.csv'
model_save_path = '/Users/tomgo/Documents/Keele/Semester 3/Recommender System/best_neural_network_model.keras'
scaler_save_path = '/Users/tomgo/Documents/Keele/Semester 3/Recommender System/scaler.joblib'
label_encoder_save_path = '/Users/tomgo/Documents/Keele/Semester 3/Recommender System/label_encoder.joblib'
checkpoint_save_path = '/Users/tomgo/Documents/Keele/Semester 3/Trained Models/best_model.keras'

#Creates directories if they don't already exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

#Loads data in
data = pd.read_csv(train_file_path)

#Encodes target labels for model training
label_encoder = LabelEncoder()
data['Position_encoded'] = label_encoder.fit_transform(data['Position'])

#Calculats average height and weight for each position
average_positions = data.groupby('Position')[['Height (cm)', 'Weight (kg)']].mean()

#Calculates Euclidean distances between each pair of positions
distances = pd.DataFrame(index=average_positions.index, columns=average_positions.index, dtype=float)
for pos1, pos2 in itertools.combinations(average_positions.index, 2):
    dist = np.linalg.norm(average_positions.loc[pos1] - average_positions.loc[pos2])
    distances.loc[pos1, pos2] = dist
    distances.loc[pos2, pos1] = dist

#Determines the nearest neighbors for each position
nearest_neighbors = {}
for position in distances.index:
    nearest = distances.loc[position].nsmallest(3).index.to_list()
    if position in nearest:
        nearest.remove(position) 
    else:
        nearest = nearest[:2]  
    nearest_neighbors[label_encoder.transform([position])[0]] = set(label_encoder.transform(nearest))

#Decides columns for training
X = data[['Height (cm)', 'Weight (kg)', '10m Sprint Time (s)']]
y = data['Position_encoded']

#Standardises the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Splits the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Defines the neural network model 
model = Sequential([
    Dense(256, input_dim=3, activation='relu'),  
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

#Compiles the models
optimiser = Adam(learning_rate=0.001)  #Learning rate could be adjusted
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Defines early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_save_path, monitor='val_accuracy', save_best_only=True, mode='max')

#Trains the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

#Plots training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Evaluates the model on validation data
y_val_pred_prob = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)

#Calculates accuracy, F1-score, and confusion matrix
accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

#Analyses confusion matrix
def analyse_confusion_matrix(conf_matrix, class_names):
    print("\nDetailed Confusion Matrix Analysis:")
    for i, true_class in enumerate(class_names):
        for j, predicted_class in enumerate(class_names):
            if i != j and conf_matrix[i, j] > 0:
                print(f"Actual: {true_class} - Predicted: {predicted_class} - Count: {conf_matrix[i, j]}")

analyse_confusion_matrix(conf_matrix, label_encoder.classes_)

#Calculates Top-3 Accuracy
def calculate_top_3_accuracy(y_true, y_pred_prob, neighbors):
    top_3_correct = 0
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        top_3_indices = np.argsort(pred_prob)[-3:] 
        top_3_classes = set(top_3_indices)
        
        #Adds nearest neighbors to the predicted classes
        expanded_top_classes = set(top_3_classes)
        for idx in top_3_indices:
            expanded_top_classes.update(neighbors[idx])
        
        #Checks if the true label is in the top 3 predicted classes
        if true_label in expanded_top_classes:
            top_3_correct += 1

    top_3_accuracy = top_3_correct / len(y_true)
    return top_3_accuracy

top_3_accuracy = calculate_top_3_accuracy(y_val, y_val_pred_prob, nearest_neighbors)
print(f'Top-3 Accuracy: {top_3_accuracy:.4f}')

#Saves the model and other necessary components
model.save(model_save_path)
dump(scaler, scaler_save_path)
dump(label_encoder, label_encoder_save_path)
