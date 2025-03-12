import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import tempfile
import pygame
from threading import Thread
import time

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load gesture recognition model
gesture_model = tf.keras.models.load_model("C:/Users/ACER/gesture/action (1).h5", compile=False)
# Actions for gesture recognition
gesture_actions = np.array(['hello', 'thanks', 'iloveyou'])

# Load facial emotion model
emotion_model = tf.keras.models.load_model("facialemotionmodel (1).h5")
# Emotion labels - note the order matters for the model's output mapping
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Define face detection classifier for emotion detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Visualization colors for gesture probabilities
gesture_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Emotion visualization colors
emotion_colors = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 128),  # Teal
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Yellow
    'sad': (128, 128, 128),    # Gray
    'surprise': (255, 128, 0)  # Orange
}

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Global variables for TTS
last_spoken_gesture = ""
last_detected_emotion = "neutral"  # Default emotion
last_spoken_time = 0
speaking_cooldown = 3  # seconds between speaking the same phrase
emotion_confidence_threshold = 0.6  # Minimum confidence for emotion detection

# Function to speak text using gTTS with emotion-based parameters
def speak_text_with_emotion(text, emotion="neutral"):
    print(f"Speaking '{text}' with {emotion} emotion")
    
    # Speech parameters for each emotion type
    speech_params = {
        "angry": {"slow": False, "lang": "en"},     # Faster, intense
        "disgust": {"slow": True, "lang": "en"},    # Slower, disgusted tone
        "fear": {"slow": False, "lang": "en"},      # Quick, anxious
        "happy": {"slow": False, "lang": "en"},     # Upbeat, cheerful
        "neutral": {"slow": False, "lang": "en"},   # Standard speaking
        "sad": {"slow": True, "lang": "en"},        # Slower, melancholic
        "surprise": {"slow": False, "lang": "en"}   # Quicker, excited
    }
    
    # Get parameters for the current emotion (default to neutral if not found)
    params = speech_params.get(emotion, speech_params["neutral"])
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        temp_filename = f.name
    
    # Generate speech with emotion-specific parameters
    tts = gTTS(text=text, lang=params["lang"], slow=params["slow"])
    tts.save(temp_filename)
    
    # Play the speech
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()
    
    # Delete file after it's played
    def cleanup():
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        try:
            os.remove(temp_filename)
        except:
            pass
    
    # Start cleanup in a separate thread
    Thread(target=cleanup).start()

# Modify the detected gesture based on emotion
def modify_gesture_text(gesture, emotion):
    # Emotion-specific gesture phrases
    emotion_gesture_mapping = {
        "angry": {
            "hello": "Hello! I'm feeling angry right now.",
            "thanks": "Thanks! Though I'm quite upset.",
            "iloveyou": "I love you, even though I'm angry."
        },
        "disgust": {
            "hello": "Hello... I'm feeling disgusted.",
            "thanks": "Thanks, I guess. I'm not pleased.",
            "iloveyou": "I love you, despite feeling disgusted."
        },
        "fear": {
            "hello": "Hello? I'm feeling scared.",
            "thanks": "Thanks! I'm a bit frightened.",
            "iloveyou": "I love you! I'm feeling afraid though."
        },
        "happy": {
            "hello": "Hello! I'm so happy to see you!",
            "thanks": "Thanks! I'm really happy about this!",
            "iloveyou": "I love you! I'm so happy!"
        },
        "neutral": {
            "hello": "Hello there.",
            "thanks": "Thank you.",
            "iloveyou": "I love you."
        },
        "sad": {
            "hello": "Hello... I'm feeling sad today.",
            "thanks": "Thanks... I'm a bit down.",
            "iloveyou": "I love you... though I'm sad."
        },
        "surprise": {
            "hello": "Oh! Hello! That's surprising!",
            "thanks": "Wow! Thanks! I wasn't expecting that!",
            "iloveyou": "Oh my! I love you too! What a surprise!"
        }
    }
    
    # Return the emotion-modified text or just the gesture name if not found
    if emotion in emotion_gesture_mapping and gesture in emotion_gesture_mapping[emotion]:
        return emotion_gesture_mapping[emotion][gesture]
    return gesture

# Function for MediaPipe detection
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

# Extract keypoints for gesture recognition
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Prepare image for emotion detection
def extract_emotion_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Enhanced emotion detection function
def detect_emotion(face_img):
    """
    Enhanced emotion detection with additional preprocessing
    """
    try:
        # Ensure correct size for the model
        resized_face = cv2.resize(face_img, (48, 48))
        
        # Enhance contrast to make features more prominent
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_face = clahe.apply(resized_face)
        
        # Normalize and reshape for the model
        emotion_features = extract_emotion_features(enhanced_face)
        
        # Get emotion predictions
        emotion_pred = emotion_model.predict(emotion_features)[0]
        
        # Get top emotion and confidence
        max_index = np.argmax(emotion_pred)
        confidence = emotion_pred[max_index]
        emotion = emotion_labels[max_index]
        
        # Debug print
        print(f"Detected emotion: {emotion} with confidence: {confidence:.2f}")
        print(f"All emotions: {[(emotion_labels[i], emotion_pred[i]) for i in range(len(emotion_pred))]}")
        
        # Return emotion only if confidence is high enough
        if confidence > emotion_confidence_threshold:
            return emotion, confidence
        return "neutral", confidence  # Default to neutral if confidence is low
        
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "neutral", 0.0

# Visualization function for gesture probabilities
def prob_viz(res, actions, image, colors):
    output_image = image.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_image, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_image, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_image

# Main function
def main():
    global last_spoken_gesture, last_detected_emotion, last_spoken_time
    
    # Initialize variables for gesture recognition
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    
    # For emotion stability (smooth out rapid changes)
    emotion_history = ["neutral"] * 5  # Keep track of last 5 emotions
    
    # Start webcam capture
    cap = cv2.VideoCapture(0)
    
    # Set video resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Set MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break
                
            # Make MediaPipe detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # --- EMOTION DETECTION PROCESS ---
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            current_emotion = last_detected_emotion  # Default to last detected emotion
            emotion_confidence = 0.0
            
            for (x, y, w, h) in faces:
                try:
                    # Extract face region
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Enhanced emotion detection
                    detected_emotion, confidence = detect_emotion(face_roi)
                    
                    # Update emotion history
                    emotion_history.pop(0)
                    emotion_history.append(detected_emotion)
                    
                    # Get most common emotion in history for stability
                    from collections import Counter
                    emotion_counts = Counter(emotion_history)
                    stable_emotion, count = emotion_counts.most_common(1)[0]
                    
                    # Only update if the emotion is stable
                    if count >= 3:  # If emotion appears at least 3 times in history
                        current_emotion = stable_emotion
                        last_detected_emotion = stable_emotion
                        emotion_confidence = confidence
                    
                    # Draw rectangle around face using emotion-specific color
                    color = emotion_colors.get(current_emotion, (0, 255, 0))
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    
                    # Display emotion with confidence
                    emotion_text = f"{current_emotion} ({confidence:.2f})"
                    cv2.putText(image, emotion_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                except Exception as e:
                    print(f"Error processing face for emotion: {e}")
                    continue
            
            # --- GESTURE RECOGNITION PROCESS ---
            # Extract keypoints for gesture recognition
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            current_gesture = ""
            
            if len(sequence) == 30:
                # Predict gesture
                res = gesture_model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                # Visualization logic for gesture
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if gesture_actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(gesture_actions[np.argmax(res)])
                                current_gesture = gesture_actions[np.argmax(res)]
                        else:
                            sentence.append(gesture_actions[np.argmax(res)])
                            current_gesture = gesture_actions[np.argmax(res)]

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Visualize probabilities
                image = prob_viz(res, gesture_actions, image, gesture_colors)
            
            # Display gesture recognized
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display current emotion influencing speech
            emotion_color = emotion_colors.get(current_emotion, (16, 117, 245))
            cv2.rectangle(image, (0, image.shape[0]-40), (640, image.shape[0]), emotion_color, -1)
            cv2.putText(image, f"Voice emotion: {current_emotion}", (3, image.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- TEXT TO SPEECH INTEGRATION ---
            current_time = time.time()
            
            # Speak the gesture with emotion-based voice when a new gesture is detected
            if current_gesture and current_gesture != last_spoken_gesture and current_time - last_spoken_time > speaking_cooldown:
                # Get emotion-modified gesture text
                # speech_text = modify_gesture_text(current_gesture, current_emotion)
                
                # Speak the detected gesture with emotion-based voice
                speak_text_with_emotion(current_gesture, current_emotion)
                last_spoken_gesture = current_gesture
                last_spoken_time = current_time
            
            # Show output
            cv2.imshow('Emotion-Based Gesture Recognition', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()