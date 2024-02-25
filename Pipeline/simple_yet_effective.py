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

def prepare_input(data_string):
    # Parse the string to extract keypoints, skipping the first five elements (class name, center_x, center_y, width, height)
    data_list = data_string[0].split(',')[5:]

    keypoints_2d = np.array([float(val) for val in data_list]).reshape(-1, 3)[:, :2]  # Take only x, y, ignore visibility
    keypoints_2d = keypoints_2d.flatten()
    return torch.tensor([keypoints_2d], dtype=torch.float32)

def predict_3d_keypoints(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.numpy()

def run_prediction(data_string, model_path):
    model = load_model(model_path, input_size=26, output_size=39) 
    input_tensor = prepare_input(data_string)
    predicted_3d_keypoints = predict_3d_keypoints(model, input_tensor)
    
    # Convert the numpy array of 3D keypoints to a comma-separated string
    keypoints_str = ",".join(map(str, predicted_3d_keypoints.flatten()))
    
    return keypoints_str