import cv2
import torch
import torchvision.transforms as transforms
from models.emotion import EmotionModel

model = EmotionModel()
model.load_state_dict(torch.load("models/emotion_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
])


import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def extract_face(frame):
    """ Detects the face in the frame and returns the cropped face. """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                  int(bboxC.width * w), int(bboxC.height * h))

            # Crop face region
            face = frame[y:y+h_box, x:x+w_box]
            return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for emotion model

    return None  # Return None if no face is detected


from PIL import Image  # Import PIL library

def detect_emotion(frame):
    """ Detects facial emotion and returns an emotion label. """
    face = extract_face(frame)  # Extract the face (returns a NumPy array)

    if face is None or (face.shape[0] == 0 or face.shape[1] == 0):
        return "No Face Detected"


    face_pil = Image.fromarray(face)  # Convert NumPy array to PIL Image
    face_tensor = transform(face_pil).unsqueeze(0)  # Apply transformations

    emotion = model.predict(face_tensor)
    return emotion

