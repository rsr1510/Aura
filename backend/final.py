import pickle
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS
import os

# Load the trained LSTM model
model = load_model("C:\\Aura\\backend\\sign_language_model.h5")

# Load LabelEncoder to map numeric labels back to characters
with open("C:\\Aura\\backend\\data.pickle", 'rb') as f:
    data_dict = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(data_dict['labels'])  # Fit label encoder with original labels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Open webcam
cap = cv2.VideoCapture(0)

# Buffers for constructing words and sentences
word_buffer = ""
sentence_buffer = []
stability_counter = 0
last_letter = ""
space_confirmed = False  # Flag to confirm space addition after stability

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        # Convert input to LSTM-compatible format
        data_aux = np.asarray([data_aux])
        data_aux = pad_sequences(data_aux, maxlen=model.input_shape[1], padding='post', dtype='float32')
        data_aux = np.expand_dims(data_aux, axis=2)

        # Model prediction
        prediction = model.predict(data_aux)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_letter = label_encoder.inverse_transform(predicted_label)[0]

        # Handle prediction stability
        if predicted_letter == last_letter and predicted_letter not in ["Uncertain", "No hands detected"]:
            stability_counter += 1
        else:
            stability_counter = 0
            last_letter = predicted_letter

        # Confirm letter when stability threshold reached
        if stability_counter >= 10 and predicted_letter not in ["Uncertain", "No hands detected"]:
            if predicted_letter == "space":
                word_buffer += " "  # Add space to word buffer when confirmed
                sentence_buffer.append(word_buffer)
                word_buffer = ""  
                print(f"Space added. Current word: {word_buffer}")
            elif predicted_letter == "del":
                word_buffer = word_buffer[:-1]  # Remove last letter
                print(f"Deleted last letter. Current word: {word_buffer}")
            else:
                word_buffer += predicted_letter
                print(f"Current word: {word_buffer}")

            # Reset stability after confirming
            stability_counter = 0

    # Display information on screen
    info_panel = np.zeros((200, W, 3), dtype=np.uint8)  # Create a black panel for text
    cv2.putText(info_panel, f"Current word: {word_buffer}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Format sentence display
    sentence_text = " ".join(sentence_buffer)
    if len(sentence_text) > 60:  # Truncate for display
        sentence_text = "..." + sentence_text[-60:]

    cv2.putText(info_panel, f"Sentence: {sentence_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display instructions
    cv2.putText(info_panel, "Press 'q' to quit, 'space' to speak, 'del' to delete", 
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Stack the info panel on top of the frame
    combined_display = np.vstack((info_panel, frame))
    cv2.imshow('Sign Language Recognition', combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If the spacebar is pressed, speak the whole sentence

    if cv2.waitKey(1) & 0xFF == ord(' '):  # Check for spacebar key press
        if sentence_buffer:  # Only speak if there is text in sentence_buffer
            tts = gTTS(text=" ".join(sentence_buffer), lang='en')
            tts.save("output.mp3")
            os.system("start output.mp3")  # Use "afplay output.mp3" on macOS or "mpg321 output.mp3" on Linux
            sentence_buffer = []  # Reset sentence buffer after speaking
        else:
            print("No text to speak")

    elif cv2.waitKey(1) & 0xFF == 8:  # Backspace pressed: remove last letter from word buffer
        if word_buffer:
            word_buffer = word_buffer[:-1] 

cap.release()
cv2.destroyAllWindows()
