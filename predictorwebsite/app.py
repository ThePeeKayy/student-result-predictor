from flask import Flask, request, jsonify
import pandas as pd
import torch
from torch import nn
import pickle
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 512)
        self.fc2 = nn.Linear(512, 256)
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


@app.route('/bot', methods=['POST'])
def predict_math_score():
    # Get data from the request
    data = request.json
    reading_score = data.get('readingResult')
    writing_score = data.get('writingResult')
    test_option = data.get('testOption')
    race_option = data.get('raceOption')
    lunch_option = data.get('lunchOption')
    gender_option = data.get('genderOption')
    education_option = data.get('educationOption')
    # Prepare the new single input
    new_input = pd.DataFrame({
        'writing score': [int(writing_score)],
        'reading score': [int(reading_score)],
        'parental level of education': [education_option],
        'gender': [gender_option],
        'race/ethnicity': [race_option],
        'lunch': [lunch_option],
        'test preparation course': [test_option]
    })
    
    # Encode and scale the input
    original_columns = ['writing score', 'reading score', 'parental level of education_associate\'s degree', 'parental level of education_bachelor\'s degree', 'parental level of education_high school', 'parental level of education_master\'s degree', 'parental level of education_some college', 'parental level of education_some high school', 'gender_female', 'gender_male', 'race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C', 'race/ethnicity_group D', 'race/ethnicity_group E', 'lunch_free/reduced', 'lunch_standard', 'test preparation course_completed', 'test preparation course_none']
    new_input_encoded = pd.get_dummies(new_input)
    new_input_encoded = new_input_encoded.reindex(columns=original_columns, fill_value=0)
    new_input_scaled = scaler.transform(new_input_encoded)
    new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)

    model.load_state_dict(torch.load('student_result_predictor.pth'))
    model.eval()

    # Use the model to predict the math score
    with torch.no_grad():
        predicted_math_score = model(new_input_tensor)

    # The predicted math score will be a tensor, you can convert it to a float if needed
    predicted_math_score = predicted_math_score.item()

    return jsonify({'mathResult': predicted_math_score})

if __name__ == '__main__':
    app.run()