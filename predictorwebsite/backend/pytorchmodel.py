import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pickle


# Load the dataset
data_encoded = pd.read_csv('StudentsPerformance.csv')

# Prepare the data
X = pd.concat([data_encoded[['writing score', 'reading score']], pd.get_dummies(data_encoded[['parental level of education', 
                                                                                             'gender', 'race/ethnicity', 'lunch', 'test preparation course']])], axis=1)
y = data_encoded['math score']

math_score_mean = data_encoded['math score'].mean()
math_score_std = data_encoded['math score'].std()

print(f'Mean of math score: {math_score_mean}')
print(f'Standard deviation of math score: {math_score_std}')

num_features = X.shape[1]
print("Number of features:", num_features)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 512)  # Update input size to match the number of features
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create an instance of the neural network
model = Net()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00125)

# Define data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(250):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    mse = criterion(outputs, y_test_tensor.unsqueeze(1))
    print(f'Mean Squared Error: {mse}')

# Save the model
# After fitting the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

torch.save(model.state_dict(), 'student_result_predictor.pth')


# Assuming `scaler` and `model` are already loaded or trained

# Prepare the new single input
new_input = pd.DataFrame({
    'writing score': [44],  # Example writing score
    'reading score': [57],  # Example reading score
    'parental level of education': ['associate\'s degree'],  # Example parental level of education
    'gender': ['male'],  # Example gender
    'race/ethnicity': ['group A'],  # Example race/ethnicity
    'lunch': ['free/reduced'],  # Example lunch
    'test preparation course': ['none']  # Example test preparation course
})
# Encode the new input
new_input_encoded = pd.get_dummies(new_input)

# Reorder and add missing columns to match the original data
original_columns = ['writing score', 'reading score', 'parental level of education_associate\'s degree', 'parental level of education_bachelor\'s degree', 'parental level of education_high school', 'parental level of education_master\'s degree', 'parental level of education_some college', 'parental level of education_some high school', 'gender_female', 'gender_male', 'race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C', 'race/ethnicity_group D', 'race/ethnicity_group E', 'lunch_free/reduced', 'lunch_standard', 'test preparation course_completed', 'test preparation course_none']
new_input_encoded = new_input_encoded.reindex(columns=original_columns, fill_value=0)

# Scale the features
new_input_scaled = scaler.transform(new_input_encoded)

# Convert to PyTorch tensor
new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)

# Use the model to predict the math score
model.load_state_dict(torch.load('student_result_predictor.pth'))
model.eval()

with torch.no_grad():
    predicted_math_score = model(new_input_tensor)

# The predicted math score will be a tensor, you can convert it to a float if needed
predicted_math_score = predicted_math_score.item()

print("Predicted Math Score:", predicted_math_score)
