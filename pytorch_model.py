from main import Net
import torch

model = Net()
model.load_state_dict(torch.load('student_result_predictor.pth'))
model.eval()


