import cv2
import mediapipe as mp
import numpy as np
import torch
from utils.handtracker import detect_hands, is_finger_spelling
from utils.posetracker import detect_posture
from utils.face import detect_emotion
from models.gesture import GestureRecognitionModel
from models.fingerspelling import FingerSpellingModel
from utils.text_to_speech import generate_speech

# Load pre-trained models
gesture_model = GestureRecognitionModel()
gesture_model.load_state_dict(torch.load("models/gesture_model.pth"))
gesture_model.eval()

fingerspelling_model = FingerSpellingModel()
fingerspelling_model.load_state_dict(torch.load("models/fingerspelling_model.pth"))
fingerspelling_model.eval()

# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    hand_keypoints = detect_hands(frame)
    posture = detect_posture(frame)
    emotion = detect_emotion(frame)

    # Decide mode (Finger Spelling or Word Gesture)
    if len(hand_keypoints) > 0:
        if is_finger_spelling(hand_keypoints):
            letter = fingerspelling_model.predict(hand_keypoints)
            text_output += letter  # Accumulate letters into words
        else:
            word = gesture_model.predict(hand_keypoints)
            text_output = word  # Full-word recognition
    
    # Generate speech with emotional tone
    if text_output:
        generate_speech(text_output, emotion, posture)

    # Display processed video feed
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
