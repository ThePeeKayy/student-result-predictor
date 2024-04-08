import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib

# Load the dataset
data_encoded = pd.read_csv('StudentsPerformance.csv')

# Prepare the data
X = pd.concat([data_encoded[['writing score', 'reading score']], pd.get_dummies(data_encoded['parental level of education'])], axis=1)
y = data_encoded['math score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Ridge regression model
model = Ridge(alpha=1)  # You can adjust the alpha parameter for regularization
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'student_result_predictor.joblib')