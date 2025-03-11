from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
import asyncio
import base64
import tempfile
from gtts import gTTS
import uvicorn
from pathlib import Path
import socket
import logging
import tensorflow as tf
import subprocess
import json
import speech_recognition as sr
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to reduce warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Define model directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Define static directory
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Define 3D directory
THREED_DIR = BASE_DIR / "3D"
THREED_DIR.mkdir(exist_ok=True)

def find_free_port(start_port=8000, max_port=8100):
    """Find a free port to bind to."""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError("No free ports available in range {}-{}".format(start_port, max_port))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables for sign language recognition
global_frame = None
is_camera_active = False
word_buffer = ""
sentence_buffer = []
stability_counter = 0
last_letter = ""

# Add these global variables
speech_recognition_active = False
speech_queue = Queue()
recognizer = sr.Recognizer()

try:
    # Load the model and label encoder
    logger.info("Loading model and label encoder...")
    
    # Configure model for better CPU performance
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    model = load_model(str(BASE_DIR / "sign_language_model.h5"), compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with open(str(BASE_DIR / "data.pickle"), 'rb') as f:
        data_dict = pickle.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(data_dict['labels'])
    logger.info("Model and label encoder loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model or data: {e}")
    raise

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

def process_hands():
    global global_frame, is_camera_active, word_buffer, sentence_buffer, stability_counter, last_letter
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            raise Exception("Could not open camera")
        
        logger.info("Camera initialized successfully")
        frame_count = 0
        error_count = 0
        last_time = time.time()
        fps = 0
        
        while is_camera_active:
            try:
                current_time = time.time()
                data_aux = []
                x_ = []
                y_ = []

                ret, frame = cap.read()
                if not ret:
                    error_count += 1
                    if error_count > 5:  # If we get too many errors, restart the camera
                        logger.warning("Multiple frame capture failures, attempting to restart camera")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(0)
                        error_count = 0
                    continue

                error_count = 0  # Reset error count on successful frame capture
                frame_count += 1

                # Ensure consistent frame size
                frame = cv2.resize(frame, (640, 480))
                H, W, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Create info panel with consistent size
                info_panel = np.zeros((200, 640, 3), dtype=np.uint8)

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

                    try:
                        # Convert input to LSTM-compatible format
                        data_aux = np.asarray([data_aux])
                        data_aux = pad_sequences(data_aux, maxlen=model.input_shape[1], padding='post', dtype='float32')
                        data_aux = np.expand_dims(data_aux, axis=2)

                        # Model prediction
                        prediction = model.predict(data_aux, verbose=0)  # Disable verbose output
                        predicted_label = np.argmax(prediction, axis=1)
                        predicted_letter = label_encoder.inverse_transform(predicted_label)[0]

                        # Add prediction confidence to display
                        confidence = float(prediction[0][predicted_label[0]])
                        cv2.putText(info_panel, f"Confidence: {confidence:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
                                if word_buffer.strip():  # Only add non-empty words
                                    sentence_buffer.append(word_buffer.strip())
                                word_buffer = ""
                            elif predicted_letter == "del":
                                if word_buffer:
                                    word_buffer = word_buffer[:-1]
                                elif sentence_buffer:
                                    word_buffer = sentence_buffer.pop()
                            else:
                                word_buffer += predicted_letter

                            # Reset stability after confirming
                            stability_counter = 0
                    except Exception as e:
                        logger.error(f"Error in prediction processing: {e}")
                        stability_counter = 0  # Reset stability counter on error

                # Calculate and display FPS
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    fps = 30.0 / (current_time - last_time)
                    last_time = current_time

                cv2.putText(info_panel, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Current word: {word_buffer}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                sentence_text = " ".join(sentence_buffer) + (" " + word_buffer if word_buffer else "")
                if len(sentence_text) > 60:
                    sentence_text = "..." + sentence_text[-60:]
                    
                cv2.putText(info_panel, f"Sentence: {sentence_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combined display with consistent size
                combined_display = np.vstack((info_panel, frame))
                
                # Update the global frame for streaming
                _, encoded_frame = cv2.imencode('.jpg', combined_display, [cv2.IMWRITE_JPEG_QUALITY, 85])
                global_frame = encoded_frame.tobytes()
                
                # Maintain frame rate
                elapsed = time.time() - current_time
                sleep_time = max(0, 0.033 - elapsed)  # Target 30 FPS
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in frame processing loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    except Exception as e:
        logger.error(f"Fatal error in process_hands: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        logger.info("Camera released")

async def generate_frames():
    global global_frame
    
    while True:
        try:
            if global_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
            else:
                # Return a blank frame if no active frame
                blank_frame = np.zeros((680, 640, 3), dtype=np.uint8)  # Match the size of info_panel + frame
                cv2.putText(blank_frame, "Initializing camera...", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.033)  # Consistent with 30 FPS target
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            await asyncio.sleep(1)  # Wait longer on error

# Health check endpoint for video feed
@app.get("/api/video_feed/health")
async def video_feed_health():
    global is_camera_active, global_frame
    return {
        "status": "healthy" if is_camera_active and global_frame is not None else "initializing",
        "camera_active": is_camera_active,
        "frame_available": global_frame is not None
    }

@app.get("/api/video_feed")
async def video_feed():
    try:
        response = StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        
        return response
    except Exception as e:
        logger.error(f"Error in video feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# OPTIONS endpoint for video feed CORS
@app.options("/api/video_feed")
async def video_feed_options():
    response = JSONResponse(content={"status": "ok"})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.post("/api/start_recognition")
async def start_recognition(background_tasks: BackgroundTasks):
    global is_camera_active, word_buffer, sentence_buffer
    
    try:
        if not is_camera_active:
            is_camera_active = True
            word_buffer = ""
            sentence_buffer = []
            
            # Use background tasks for FastAPI
            thread = threading.Thread(target=process_hands)
            thread.daemon = True
            thread.start()
            
            return {"status": "started"}
        
        return {"status": "already running"}
    except Exception as e:
        is_camera_active = False
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop_recognition")
async def stop_recognition():
    global is_camera_active
    
    try:
        if is_camera_active:
            is_camera_active = False
            await asyncio.sleep(1)  # Give thread time to terminate
            return {"status": "stopped"}
        
        return {"status": "not running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_text")
async def get_text():
    global word_buffer, sentence_buffer
    
    try:
        full_text = " ".join(sentence_buffer) + (" " + word_buffer if word_buffer else "")
        return {
            "word_buffer": word_buffer,
            "sentence_buffer": sentence_buffer,
            "full_text": full_text.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear_text")
async def clear_text():
    global word_buffer, sentence_buffer
    
    try:
        word_buffer = ""
        sentence_buffer = []
        
        return {"status": "text cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/delete_last")
async def delete_last():
    global word_buffer, sentence_buffer
    
    try:
        if word_buffer:
            word_buffer = word_buffer[:-1]
        elif sentence_buffer:
            word_buffer = sentence_buffer.pop()
        
        return {
            "status": "deleted last character",
            "word_buffer": word_buffer,
            "sentence_buffer": sentence_buffer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-speech endpoint
class TextData(BaseModel):
    text: str

@app.post("/api/speak")
async def speak(data: TextData):
    if not data.text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        # Create TTS audio
        tts = gTTS(text=data.text, lang='en')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                tts.save(temp_path)
                
                # Convert to base64 for sending to frontend
                with open(temp_path, 'rb') as audio_file:
                    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                
                return {
                    "status": "success",
                    "audio": audio_data
                }
            finally:
                # Clean up the temp file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextToSignRequest(BaseModel):
    text: str

@app.post("/api/text_to_sign")
async def text_to_sign(request: TextToSignRequest):
    try:
        # Path to the Blender script
        blender_script = BASE_DIR / "3D" / "avatartest.py"
        
        # Create a temporary file for the text input
        temp_input = BASE_DIR / "3D" / "temp_input.txt"
        with open(temp_input, "w") as f:
            f.write(request.text)
        
        # Run Blender with the script
        blender_cmd = [
            "blender",
            "--background",  # Run in background mode
            "--python", str(blender_script),
            "--",  # Pass arguments to the script
            str(temp_input)
        ]
        
        process = subprocess.Popen(
            blender_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Blender process failed: {stderr.decode()}")
        
        # Check if the output video was created
        output_video = BASE_DIR / "3D" / "output_animation.mp4"
        if not output_video.exists():
            raise Exception("Output video was not created")
        
        # Return the video URL
        return {
            "status": "success",
            "video_url": f"/static/3D/output_animation.mp4"
        }
        
    except Exception as e:
        logger.error(f"Error in text_to_sign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start_speech_recognition")
async def start_speech_recognition():
    global speech_recognition_active, speech_queue
    try:
        if not speech_recognition_active:
            # Clear any existing items in the queue
            while not speech_queue.empty():
                speech_queue.get()
            logger.info("Cleared speech queue")
            
            # Test microphone access first
            try:
                with sr.Microphone() as source:
                    logger.info("Testing microphone access...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.info("Microphone test successful")
            except Exception as mic_error:
                logger.error(f"Microphone access error: {mic_error}")
                raise HTTPException(
                    status_code=500,
                    detail="Could not access microphone. Please ensure microphone is connected and permissions are granted."
                )

            speech_recognition_active = True
            # Start speech recognition in a background thread
            thread = threading.Thread(target=speech_recognition_thread)
            thread.daemon = True
            thread.start()
            logger.info("Speech recognition started successfully")
            return {"status": "started"}
        return {"status": "already running"}
    except Exception as e:
        logger.error(f"Error starting speech recognition: {e}")
        speech_recognition_active = False
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/stop_speech_recognition")
async def stop_speech_recognition():
    global speech_recognition_active, speech_queue
    try:
        logger.info("Stopping speech recognition...")
        # Set the flag to False first
        speech_recognition_active = False
        logger.info("Set speech_recognition_active to False")
        
        # Clear the speech queue
        while not speech_queue.empty():
            speech_queue.get()
        logger.info("Cleared speech queue")
        
        # Give the thread time to stop and ensure it's stopped
        for _ in range(3):  # Try up to 3 times
            await asyncio.sleep(1)
            if not speech_recognition_active:
                logger.info("Speech recognition stopped successfully")
                return {"status": "stopped"}
        
        # If we get here, something might be wrong
        logger.warning("Speech recognition may not have stopped properly")
        return {"status": "stopped", "warning": "May not have stopped properly"}
    except Exception as e:
        logger.error(f"Error stopping speech recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_speech_text")
async def get_speech_text():
    try:
        texts = []
        # Get all available texts from the queue
        while not speech_queue.empty():
            text = speech_queue.get()
            logger.info(f"Retrieved text from queue: {text}")
            texts.append(text)
        
        if texts:
            combined_text = " ".join(texts)
            logger.info(f"Returning combined text: {combined_text}")
            return {"text": combined_text}
            
        logger.info("No text available in queue")
        return {"text": ""}
    except Exception as e:
        logger.error(f"Error getting speech text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def speech_recognition_thread():
    global speech_recognition_active
    try:
        with sr.Microphone() as source:
            logger.info("Initializing speech recognition...")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Ambient noise adjustment complete")
            
            while speech_recognition_active:
                if not speech_recognition_active:  # Double check at loop start
                    logger.info("Speech recognition deactivated, exiting thread")
                    break
                    
                try:
                    logger.info("Listening for speech...")
                    # Shorter timeout to check stop flag more frequently
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    
                    if not speech_recognition_active:  # Check after audio capture
                        logger.info("Speech recognition stopped after audio capture")
                        break
                        
                    logger.info("Audio captured, recognizing...")
                    text = recognizer.recognize_google(audio)
                    logger.info(f"Recognized text: {text}")
                    
                    if text and speech_recognition_active:  # Only add text if still active
                        speech_queue.put(text)
                        logger.info(f"Added text to queue: {text}")
                        
                except sr.WaitTimeoutError:
                    if not speech_recognition_active:
                        logger.info("Speech recognition stopped during timeout")
                        break
                    continue
                except sr.UnknownValueError:
                    logger.info("Speech not understood")
                    continue
                except sr.RequestError as e:
                    logger.error(f"Could not request results from speech recognition service: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error in speech recognition: {e}")
                    if not speech_recognition_active:
                        break
                    continue
                    
    except Exception as e:
        logger.error(f"Fatal error in speech recognition thread: {e}")
    finally:
        speech_recognition_active = False
        logger.info("Speech recognition thread stopped")

if __name__ == "__main__":
    try:
        # Find an available port
        port = find_free_port()
        logger.info(f"Starting server on port {port}")
        
        # Start the server
        uvicorn.run(app, host="localhost", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise