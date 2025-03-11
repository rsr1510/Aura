# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import tensorflow as tf


# class SignLanguageDetector:
#     def __init__(self, model_path=None):
#         # Initialize MediaPipe hands module for hand landmark detection
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(static_image_mode=False, 
#                                          max_num_hands=2, 
#                                          min_detection_confidence=0.5,
#                                          min_tracking_confidence=0.5)
#         self.mp_drawing = mp.solutions.drawing_utils
        
#         # Define class labels (ASL alphabet)
#         self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
#                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
#                       'Y', 'Z', 'space', 'del', 'nothing']
        
#         # Load model if path is provided
#         if model_path and os.path.exists(model_path):
#             self.model = tf.keras.models.load_model(model_path)
#             print(f"Model loaded from {model_path}")
#         else:
#             self.model = None
#             print("No model loaded. Use train_model() to train a new model.")
        
#         # Initialize text-to-speech components
    
        
#         # Text buffer for word formation
#         self.text_buffer = ""
#         self.last_prediction = None
#         self.prediction_stability_count = 0
#         self.min_stability_count = 10  # Frames with same prediction to confirm


# if __name__ == "__main__":
#     detector = SignLanguageDetector("C:\\Aura\\backend\\asl_model_best.h5")
#     sample_image = cv2.imread("C:\\Aura\\backend\\asl_dataset\\A\\A3.jpg")
#     processed_image = cv2.resize(sample_image, (128, 128)) / 255.0
#     processed_image = np.expand_dims(processed_image, axis=0)
#     prediction = detector.model.predict(processed_image)[0]
#     print("Prediction Probabilities:", prediction)
#     print("Predicted Class:", detector.labels[np.argmax(prediction)])


import sys
print(sys.executable)
