import torch
import torch.nn as nn
import torch.optim as optim

class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size=42, num_classes=50):  # Assuming 21 keypoints * 2 (x, y)
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, keypoints):
        """ Convert keypoints to tensor and get predicted gesture """
        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.forward(keypoints_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label  # Return class index

