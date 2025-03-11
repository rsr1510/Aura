import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionModel(nn.Module):
    def __init__(self, input_size=2304, num_classes=4):  # Assuming 48x48 grayscale input
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)  # 4 emotions: happy, sad, angry, neutral

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, face_tensor):
        """ Takes an image tensor and returns the predicted emotion label. """
        with torch.no_grad():
            output = self.forward(face_tensor.view(-1))  
            print("Raw Scores:", output)
            predicted_label = torch.argmax(output).item()


        
        emotions = ["happy", "sad", "angry", "neutral"]
        return emotions[predicted_label]  # Return emotion string
