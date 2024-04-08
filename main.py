import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('StudentsPerformance.csv')
df = pd.DataFrame(data)
columns_to_drop = ['lunch', 'race/ethnicity']
df = df.drop(columns=columns_to_drop, axis=1)

# Prepare the data
X = df.drop('math score', axis=1)  # Assuming 'result_column_name' is the column to predict
y = df['math score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Ridge regression model
model = Ridge(alpha=1.0)  # You can adjust the alpha parameter for regularization
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
