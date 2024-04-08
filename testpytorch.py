from pytorchmodel import Net
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_data = [[70, 80, 'bachelor\'s degree'], [65, 75, 'master\'s degree']]
filtered_data = pd.concat([new_data[['writing score', 'reading score']], pd.get_dummies(new_data['parental level of education'])])
model = Net()
model.load_state_dict(torch.load('student_result_predictor.pth'))
model.eval()

# Assuming 'new_data' is a DataFrame containing the new data
X_new_scaled = scaler.transform(filtered_data)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_new_tensor)

print(outputs)