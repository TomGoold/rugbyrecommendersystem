import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

#Loads and preprocesses data
file_path = 'GPT Heights and Weights.csv'
data = pd.read_csv(file_path)

X = data[['Height (cm)', 'Weight (kg)']]
y = data['Position']

#Encodes target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Standardises the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#Trains the model with cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=kf, scoring='accuracy')

#Fits the model on the entire dataset
rf_model.fit(X_scaled, y_encoded)

#Saves the model and other necessary components
dump(rf_model, 'random_forest_model.joblib')
dump(scaler, 'scaler.joblib')
dump(label_encoder, 'label_encoder.joblib')

#Prints cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation accuracy: {cv_scores.mean()}")
