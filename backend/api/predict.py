# from fastapi import APIRouter, UploadFile, File
# import cv2
# import numpy as np
# import pickle
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder

# router = APIRouter()

# # Load the trained model
# model = load_model("C:\\Aura\\backend\\sign_language_model.h5")

# # Load LabelEncoder
# with open("C:\\Aura\\backend\\data.pickle", "rb") as f:
#     data_dict = pickle.load(f)

# label_encoder = LabelEncoder()
# label_encoder.fit(data_dict["labels"])

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# @router.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if not results.multi_hand_landmarks:
#         return {"prediction": "No hands detected"}

#     data_aux = []
#     x_, y_ = [], []

#     for hand_landmarks in results.multi_hand_landmarks:
#         for i in range(len(hand_landmarks.landmark)):
#             x_.append(hand_landmarks.landmark[i].x)
#             y_.append(hand_landmarks.landmark[i].y)

#         for i in range(len(hand_landmarks.landmark)):
#             data_aux.append(hand_landmarks.landmark[i].x - min(x_))
#             data_aux.append(hand_landmarks.landmark[i].y - min(y_))

#     data_aux = np.asarray([data_aux])
#     data_aux = pad_sequences(data_aux, maxlen=model.input_shape[1], padding='post', dtype='float32')
#     data_aux = np.expand_dims(data_aux, axis=2)

#     prediction = model.predict(data_aux)
#     predicted_label = np.argmax(prediction, axis=1)
#     predicted_letter = label_encoder.inverse_transform(predicted_label)[0]

#     return {"prediction": predicted_letter}










from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import pickle
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import threading
import time
import os
import uvicorn
from typing import Dict, List, Union
from pydantic import BaseModel

app = FastAPI()

# Global variables to control video feed
global_frame = None
is_camera_active = False
recognition_thread = None
word_buffer = ""
sentence_buffer = []
stability_counter = 0
last_letter = ""

# Load the model and label encoder once
model = load_model("C:\\Aura\\backend\\sign_language_model.h5")

with open("C:\\Aura\\backend\\data.pickle", 'rb') as f:
    data_dict = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(data_dict['labels'])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

def process_hands():
    global global_frame, is_camera_active, word_buffer, sentence_buffer, stability_counter, last_letter
    
    cap = cv2.VideoCapture(0)
    
    while is_camera_active:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            continue

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
                    word_buffer += " "
                    sentence_buffer.append(word_buffer)
                    word_buffer = ""
                elif predicted_letter == "del":
                    word_buffer = word_buffer[:-1]
                else:
                    word_buffer += predicted_letter

                # Reset stability after confirming
                stability_counter = 0

        # Display information on frame
        info_panel = np.zeros((200, W, 3), dtype=np.uint8)
        cv2.putText(info_panel, f"Current word: {word_buffer}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        sentence_text = " ".join(sentence_buffer) + word_buffer
        if len(sentence_text) > 60:
            sentence_text = "..." + sentence_text[-60:]
            
        cv2.putText(info_panel, f"Sentence: {sentence_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combined display
        combined_display = np.vstack((info_panel, frame))
        
        # Update the global frame for streaming
        _, encoded_frame = cv2.imencode('.jpg', combined_display)
        global_frame = encoded_frame.tobytes()
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()

async def generate_frames():
    global global_frame
    
    while True:
        if global_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
        else:
            # Return a blank frame if no active frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start_recognition")
async def start_recognition(background_tasks: BackgroundTasks):
    global is_camera_active, recognition_thread, word_buffer, sentence_buffer
    
    if not is_camera_active:
        is_camera_active = True
        word_buffer = ""
        sentence_buffer = []
        
        # Use background task instead of threading for FastAPI
        background_tasks.add_task(process_hands)
        
        return {"status": "started"}
    
    return {"status": "already running"}

@app.post("/stop_recognition")
async def stop_recognition():
    global is_camera_active
    
    if is_camera_active:
        is_camera_active = False
        time.sleep(1)  # Give thread time to terminate
        return {"status": "stopped"}
    
    return {"status": "not running"}

@app.get("/get_text")
async def get_text():
    global word_buffer, sentence_buffer
    
    full_text = " ".join(sentence_buffer) + (" " + word_buffer if word_buffer else "")
    return {
        "word_buffer": word_buffer,
        "sentence_buffer": sentence_buffer,
        "full_text": full_text
    }

@app.post("/clear_text")
async def clear_text():
    global word_buffer, sentence_buffer
    
    word_buffer = ""
    sentence_buffer = []
    
    return {"status": "text cleared"}

@app.post("/delete_last")
async def delete_last():
    global word_buffer
    
    if word_buffer:
        word_buffer = word_buffer[:-1]
    
    return {"status": "deleted last character", "word_buffer": word_buffer}

# Import asyncio for async functionality
import asyncio

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)