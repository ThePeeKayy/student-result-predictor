import torch
import pandas as pd
import pickle
from pytorchmodel import model
# Load the trained scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

new_data = pd.read_csv('StudentsPerformance.csv')
new_features = new_data.iloc[1]
print (new_features)
# Reorder and add missing columns to match the original data
new_features_encoded = pd.concat([new_features[['writing score', 'reading score']], pd.get_dummies(new_features[['parental level of education', 
                                                                                             'gender', 'race/ethnicity', 'lunch', 'test preparation course']])], axis=1)
new_features_encoded.columns = new_features_encoded.columns.astype(str)

new_num_features = new_features_encoded.shape[1]
print("New number of features:", new_num_features)

# Scale the features
new_features_scaled = scaler.transform(new_features_encoded)  # Assuming `scaler` is your trained StandardScaler object
new_features_tensor = torch.tensor(new_features_scaled, dtype=torch.float32)

# Use the model to predict the math score
model.eval()
with torch.no_grad():
    predicted_math_score = model(new_features_tensor)

# The predicted math score will be a tensor, you can convert it to a float if needed
predicted_math_score = predicted_math_score.item()

print("Predicted Math Score:", predicted_math_score)
