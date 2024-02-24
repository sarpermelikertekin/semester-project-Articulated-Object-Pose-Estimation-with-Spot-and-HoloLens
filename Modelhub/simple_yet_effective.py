import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PoseEstimationNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PoseEstimationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_model(model_path, input_size, output_size):
    model = PoseEstimationNet(input_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def prepare_input(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints_2d = [kp['position'] for kp in data['keypoints']]
    keypoints_2d = np.array([(kp['x'], kp['y']) for kp in keypoints_2d]).flatten()
    return torch.tensor([keypoints_2d], dtype=torch.float32)

def predict_3d_keypoints(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.numpy()

def run_prediction():
    model_path = "C:/Users/sakar/Semester Project/simple-effective.pth"
    json_path = "C:/Users/sakar/Semester Project/Spot Datasets/13 - 2 Spots/test/mapping_2d/1002.json"

    model = load_model(model_path, input_size=26, output_size=39)
    input_tensor = prepare_input(json_path)
    predicted_3d_keypoints = predict_3d_keypoints(model, input_tensor)
    
    print(predicted_3d_keypoints)
    print(type(predicted_3d_keypoints))
