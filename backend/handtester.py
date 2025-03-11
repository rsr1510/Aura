import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import mediapipe as mp
import os
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import pygame
from pygame import mixer
import time
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import pickle



class SignLanguageDetector:
    def __init__(self, model_path=None):
        # Initialize MediaPipe hands module for hand landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                         max_num_hands=2, 
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define class labels (ASL alphabet)
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                      'Y', 'Z', 'space', 'del', 'nothing']
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = None
            print("No model loaded. Use train_model() to train a new model.")
        
        # Initialize text-to-speech components
        pygame.init()
        mixer.init()
        
        # Text buffer for word formation
        self.text_buffer = ""
        self.last_prediction = None
        self.prediction_stability_count = 0
        self.min_stability_count = 10  # Frames with same prediction to confirm
    
    def download_dataset(self, dataset_dir="asl_dataset"):
        """
        Download or setup the ASL dataset. In this case, we'll use a placeholder for the dataset path.
        In real implementation, you would download from a specific source or use a locally available dataset.
        
        Note: You should replace this with actual dataset downloading code relevant to your chosen dataset.
        """
        print("Please download one of the following datasets manually:")
        print("1. ASL Alphabet Dataset (Kaggle): https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("2. ASL Lexicon Video Dataset: https://www.bu.edu/asllrp/av/dai-asllvd.html")
        print("3. MS-ASL Dataset: https://www.microsoft.com/en-us/research/project/ms-asl/")
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"Created directory {dataset_dir}. Please place your dataset files here.")
        else:
            print(f"Directory {dataset_dir} already exists.")
        
        return dataset_dir
    
    # def preprocess_dataset(self, dataset_dir, output_dir="processed_dataset"):
    #     """
    #     Preprocess the dataset by extracting hand landmarks from images and saving them.
    #     This is useful to speed up training by pre-extracting features.
    #     """
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
        
    #     for label in self.labels:
    #         label_dir = os.path.join(dataset_dir, label)
    #         output_label_dir = os.path.join(output_dir, label)
            
    #         if not os.path.exists(output_label_dir):
    #             os.makedirs(output_label_dir)
            
    #         if not os.path.exists(label_dir):
    #             print(f"Directory for label {label} not found. Skipping.")
    #             continue
            
    #         image_files = glob.glob(os.path.join(label_dir, "*.jpg")) + \
    #                      glob.glob(os.path.join(label_dir, "*.png"))
            
    #         for i, image_file in enumerate(image_files):
    #             if i % 100 == 0:
    #                 print(f"Processing {label}: {i}/{len(image_files)}")
                
    #             image = cv2.imread(image_file)
    #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
    #             results = self.hands.process(image_rgb)
                
    #             # Create a blank image to draw landmarks
    #             h, w, c = image.shape
    #             landmark_image = np.zeros((h, w, c), dtype=np.uint8)
                
    #             # Draw landmarks if detected
    #             if results.multi_hand_landmarks:
    #                 for hand_landmarks in results.multi_hand_landmarks:
    #                     self.mp_drawing.draw_landmarks(
    #                         landmark_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
    #                 # Save the landmark image
    #                 output_file = os.path.join(output_label_dir, os.path.basename(image_file))
    #                 cv2.imwrite(output_file, landmark_image)
        
    #     print("Preprocessing completed.")
    #     return output_dir
    




    def preprocess_dataset(self, dataset_dir, output_file="C:\\Aura\\backend\\data.pickle"):
        """
        Extract hand landmarks from images and save them as a pickle file.
        """
        data = []
        labels = []

        if not os.path.exists(dataset_dir):
            print(f"Dataset directory '{dataset_dir}' not found.")
            return None

        for label in self.labels:
            label_dir = os.path.join(dataset_dir, label)

            if not os.path.exists(label_dir):
                print(f"Directory for label '{label}' not found. Skipping.")
                continue

            image_files = glob.glob(os.path.join(label_dir, "*.jpg")) + \
                        glob.glob(os.path.join(label_dir, "*.png"))

            for i, image_file in enumerate(image_files):
                if i % 100 == 0:
                    print(f"Processing {label}: {i}/{len(image_files)}")

                image = cv2.imread(image_file)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                data_aux = []
                x_, y_ = [], []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(label)

        # Save extracted data to a pickle file
        with open(output_file, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)

        print(f"Preprocessing completed. Data saved to {output_file}")
        return output_file



















    def build_model(self):
        """
        Build and compile the CNN model for ASL sign detection
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model




    # def build_model(self):
    #     """
    #     Build and compile a neural network for ASL recognition using hand landmarks.
    #     """
    #     model = Sequential([
    #         Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks * 3 (x, y, z)
    #         Dropout(0.5),
    #         Dense(64, activation='relu'),
    #         Dropout(0.4),
    #         Dense(len(self.labels), activation='softmax')  # Output layer for classification
    #     ])
        
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     self.model = model
    #     return model
    
    def train_model(self, dataset_dir, batch_size=32, epochs=10, validation_split=0.2):
        """
        Train the model using the provided dataset
        """
        if not self.model:
            self.build_model()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # Setup data generators
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Setup callbacks
        checkpoint = ModelCheckpoint(
            'asl_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save the final model
        self.model.save('asl_model_final.h5')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        print(f"Model trained and saved as 'asl_model_final.h5'")
        return history



    # def train_model(self, dataset_csv, batch_size=32, epochs=10, validation_split=0.2):
    #     """
    #     Train the model using the provided landmark dataset (CSV).
    #     """
    #     if not self.model:
    #         self.build_model()
        
    #     # Load dataset from CSV
    #     df = pd.read_csv(dataset_csv)
        
    #     # Separate features (landmarks) and labels
    #     X = df.iloc[:, :-1].values  # All columns except the last (features)
    #     y = df.iloc[:, -1].values   # Last column (labels)
        
    #     # Convert labels to categorical (one-hot encoding)
    #     from sklearn.preprocessing import LabelEncoder
    #     label_encoder = LabelEncoder()
    #     y_encoded = label_encoder.fit_transform(y)
    #     y_categorical = to_categorical(y_encoded, num_classes=len(self.labels))

    #     # Split into training and validation sets
    #     X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=validation_split, random_state=42)

    #     # Normalize features (scaling between 0 and 1)
    #     X_train = np.array(X_train, dtype=np.float32)
    #     X_val = np.array(X_val, dtype=np.float32)

    #     # Setup callbacks
    #     checkpoint = ModelCheckpoint(
    #         'asl_model_best.h5',
    #         monitor='val_accuracy',
    #         save_best_only=True,
    #         mode='max',
    #         verbose=1
    #     )
        
    #     early_stopping = EarlyStopping(
    #         monitor='val_loss',
    #         patience=5,
    #         restore_best_weights=True,
    #         verbose=1
    #     )

    #     # Train the model
    #     history = self.model.fit(
    #         X_train, y_train,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         validation_data=(X_val, y_val),
    #         callbacks=[checkpoint, early_stopping]
    #     )

    #     # Save the final model
    #     self.model.save('land_model_final.h5')

    #     # Plot training history
    #     plt.figure(figsize=(12, 4))
        
    #     plt.subplot(1, 2, 1)
    #     plt.plot(history.history['accuracy'])
    #     plt.plot(history.history['val_accuracy'])
    #     plt.title('Model Accuracy')
    #     plt.ylabel('Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Validation'], loc='upper left')
        
    #     plt.subplot(1, 2, 2)
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('Model Loss')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Validation'], loc='upper left')
        
    #     plt.tight_layout()
    #     plt.savefig('training_history.png')
    #     plt.close()
        
    #     print(f"Model trained and saved as 'land_model_final.h5'")
    #     return history





    
    def extract_hand_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # Create a black image for landmark visualization
        landmark_image = np.zeros_like(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Determine hand type (left or right)
                hand_type = None
                if results.multi_handedness:
                    for idx, classification in enumerate(results.multi_handedness):
                        if classification.classification[0].label == 'Left':
                            hand_type = 'Left'
                        else:
                            hand_type = 'Right'
                
                # Draw landmarks with different colors for left and right hands
                color = (255, 0, 0) if hand_type == 'Left' else (0, 255, 0)
                
                self.mp_drawing.draw_landmarks(
                    landmark_image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2)
                )
        
        return landmark_image
    
    def predict_sign(self, frame):
        """
        Predict the sign from the given frame
        """
        if not self.model:
            print("Error: Model not loaded. Train or load a model first.")
            return None
        
        # Extract hand landmarks
        landmark_image = self.extract_hand_landmarks(frame)
        
        # Check if any landmarks were detected
        if np.sum(landmark_image) == 0:
            return "No hands detected"
        
        # Resize to match model input size
        processed_image = cv2.resize(frame, (128, 128))
        processed_image = processed_image / 255.0  # Normalize
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(processed_image)[0]
        
        # Print out detailed prediction probabilities
        top_indices = prediction.argsort()[-5:][::-1]  # Top 5 predictions
        print("\nTop 5 Predictions:")
        for idx in top_indices:
            print(f"{self.labels[idx]}: {prediction[idx]:.4f}")
        
        # Get the top prediction
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Only return prediction if confidence is high enough
        if confidence > 0.7:
            return self.labels[predicted_class_idx]
        else:
            return "Uncertain"
    
    def text_to_speech(self, text):
        """
        Convert text to speech and play it
        """
        if not text or text == "":
            return
        
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save("temp_speech.mp3")
        
        # Play the speech
        mixer.music.load("temp_speech.mp3")
        mixer.music.play()
        
        # Wait for audio to finish
        while mixer.music.get_busy():
            time.sleep(0.1)

    def print_model_details(self):
        """Print detailed model information"""
        if self.model:
            print("Model Summary:")
            self.model.summary()
            
            print("\nModel Input Shape:")
            print(self.model.input_shape)
            
            print("\nModel Output Shape:")
            print(self.model.output_shape)
            
            print("\nModel Layers:")
            for layer in self.model.layers:
                print(f"{layer.name}: Input Shape {layer.input_spec.shape if hasattr(layer, 'input_spec') else 'N/A'} -> Output Shape {layer.output_shape}")

    def analyze_dataset_distribution(self, dataset_dir):
        """Analyze class distribution in the dataset"""
        import os
        
        class_counts = {}
        for label in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, label)
            if os.path.isdir(label_path):
                class_counts[label] = len(os.listdir(label_path))
        
        print("Dataset Class Distribution:")
        for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{label}: {count} images")
    
    def run_detection(self):
        """
        Run real-time detection using webcam with robust image handling and dimension debugging
        """
        if not self.model:
            print("Error: Model not loaded. Train or load a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        # Explicitly set frame size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # UI setup
        word_buffer = ""
        sentence_buffer = []
        stability_counter = 0
        last_letter = None
        
        print("ASL Detection started. Press 'q' to quit, 'space' to speak sentence, 'backspace' to delete.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from camera.")
                break
            
            # Flip horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Ensure frame is exactly the size we want
            frame = cv2.resize(frame, (640, 480))
            print(f"Frame shape: {frame.shape}")
            
            # Create a copy of the frame for landmark visualization
            frame_for_landmarks = frame.copy()
            
            # Extract landmarks
            landmark_image = self.extract_hand_landmarks(frame_for_landmarks)
            print(f"Landmark image shape before resize: {landmark_image.shape}")
            
            # Ensure landmark_image is the same size as frame
            landmark_image = cv2.resize(landmark_image, (640, 480))
            print(f"Landmark image shape after resize: {landmark_image.shape}")
            
            # Predict sign
            predicted_letter = self.predict_sign(frame)
            
            # Create info panel with exact width
            info_panel = np.zeros((200, 640, 3), dtype=np.uint8)
            print(f"Info panel shape: {info_panel.shape}")
            
            # Display current prediction
            cv2.putText(info_panel, f"Detected: {predicted_letter}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Handle prediction stability for letter confirmation
            if predicted_letter == last_letter and predicted_letter not in ["Uncertain", "No hands detected"]:
                stability_counter += 1
                # Show stability meter
                cv2.rectangle(info_panel, (20, 50), (20 + min(stability_counter * 10, 100), 60), 
                            (0, 255, 0), -1)
            else:
                stability_counter = 0
                last_letter = predicted_letter
            
            # Confirm letter when stability threshold reached
            if stability_counter >= 10 and predicted_letter not in ["Uncertain", "No hands detected"]:
                # Process special commands
                if predicted_letter == "space":
                    if word_buffer:
                        sentence_buffer.append(word_buffer)
                        print(f"Word added: {word_buffer}")
                        self.text_to_speech(word_buffer)
                        word_buffer = ""
                elif predicted_letter == "del":
                    if word_buffer:
                        word_buffer = word_buffer[:-1]
                else:
                    # Add letter to word buffer
                    word_buffer += predicted_letter
                    print(f"Current word: {word_buffer}")
                
                # Reset stability after confirming
                stability_counter = 0
            
            # Display current word and sentence
            cv2.putText(info_panel, f"Current word: {word_buffer}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            sentence_text = " ".join(sentence_buffer)
            if len(sentence_text) > 60:  # Truncate for display
                sentence_text = "..." + sentence_text[-60:]
            
            cv2.putText(info_panel, f"Sentence: {sentence_text}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(info_panel, "Press 'q' to quit, 'space' to speak, 'backspace' to delete", 
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Modify the image combination section to ensure consistent dimensions
            try:
                # Ensure consistent dimensions
                frame = cv2.resize(frame, (640, 480))
                landmark_image = cv2.resize(landmark_image, (640, 480))
                
                # Create side-by-side view using NumPy
                side_by_side = np.concatenate([frame, landmark_image], axis=1)
                
                # Resize info panel to match width
                info_panel_resized = cv2.resize(info_panel, (side_by_side.shape[1], 200))
                
                # Combine images vertically using NumPy
                combined_view = np.concatenate([side_by_side, info_panel_resized], axis=0)
                
                # Display the combined view
                cv2.imshow('ASL Detection', combined_view)
            
            except Exception as e:
                print(f"Error combining images: {e}")
                # Fallback displays
                cv2.imshow('Frame', frame)
                cv2.imshow('Landmarks', landmark_image)
                cv2.imshow('Info Panel', info_panel)
                    
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Speak the entire sentence
                full_sentence = " ".join(sentence_buffer)
                if word_buffer:
                    full_sentence += " " + word_buffer
                
                if full_sentence:
                    print(f"Speaking: {full_sentence}")
                    self.text_to_speech(full_sentence)
                    # Reset buffers after speaking
                    sentence_buffer = []
                    word_buffer = ""
            elif key == 8:  # Backspace
                if word_buffer:
                    word_buffer = word_buffer[:-1]
                elif sentence_buffer:
                    word_buffer = sentence_buffer.pop()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    
        
    def evaluate_model(self, test_data_dir):
        """
        Evaluate the model on test data
        """
        if not self.model:
            print("Error: Model not loaded. Train or load a model first.")
            return
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate the model
        evaluation = self.model.evaluate(test_generator)
        print(f"Test Loss: {evaluation[0]:.4f}")
        print(f"Test Accuracy: {evaluation[1]:.4f}")
        
        # Generate a classification report
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Get class labels from the generator
        class_indices = test_generator.class_indices
        labels = {v: k for k, v in class_indices.items()}
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=list(class_indices.keys())))
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 15))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        num_classes = len(class_indices)
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, list(class_indices.keys()), rotation=90)
        plt.yticks(tick_marks, list(class_indices.keys()))
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print("Evaluation complete. Confusion matrix saved as 'confusion_matrix.png'")





    # def evaluate_model(self, dataset_csv):
    #     """
    #     Evaluate the model using a test dataset (CSV with landmarks).
    #     """
    #     if not self.model:
    #         print("Error: Model not loaded. Train or load a model first.")
    #         return

    #     # Load test dataset
    #     df = pd.read_csv(dataset_csv)

    #     # Separate features (landmarks) and labels
    #     X_test = df.iloc[:, :-1].values  # All columns except last (features)
    #     y_test = df.iloc[:, -1].values   # Last column (labels)

    #     # Encode labels
    #     label_encoder = LabelEncoder()
    #     y_encoded = label_encoder.fit_transform(y_test)

    #     # 🔹 Ensure num_classes matches model's expected output
    #     num_classes = self.model.output_shape[1]  # Get number of classes from model
    #     print(f"Model expects {num_classes} classes. Found {len(set(y_encoded))} in dataset.")

    #     y_test_categorical = to_categorical(y_encoded, num_classes=num_classes)  # Fix class count

    #     # Convert to NumPy arrays
    #     X_test = np.array(X_test, dtype=np.float32)

    #     # Normalize landmarks (if required)
    #     X_test = X_test / np.max(X_test)

    #     # Evaluate model
    #     evaluation = self.model.evaluate(X_test, y_test_categorical)
    #     print(f"Test Loss: {evaluation[0]:.4f}")
    #     print(f"Test Accuracy: {evaluation[1]:.4f}")

    #     return evaluation


    # def preprocess_dataset(self, dataset_dir, output_csv="landmarks_dataset.csv"):
    #     """
    #     Preprocess the dataset by extracting hand landmarks from images and saving them.
    #     """
    #     mp_hands = mp.solutions.hands
    #     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        
    #     data = []
    #     labels = []

    #     # ✅ Ensure we are processing all 29 labels
    #     all_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
    #                 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
    #                 'Y', 'Z', 'space', 'del', 'nothing']
        
    #     missing_labels = set(all_labels)  # Track missing labels

    #     for label in all_labels:  # ✅ Loop through all labels to avoid missing data
    #         label_dir = os.path.join(dataset_dir, label)
    #         if not os.path.isdir(label_dir):
    #             print(f"⚠ Warning: No folder found for '{label}', skipping.")
    #             continue  # Skip missing folders

    #         image_files = glob.glob(os.path.join(label_dir, "*.jpg")) + \
    #                     glob.glob(os.path.join(label_dir, "*.png"))

    #         if len(image_files) == 0:
    #             print(f"⚠ Warning: No images found for '{label}', skipping.")
    #             continue

    #         for image_file in image_files:
    #             image = cv2.imread(image_file)
    #             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #             results = hands.process(image_rgb)

    #             # If no hand detected, skip but log the issue
    #             if not results.multi_hand_landmarks:
    #                 print(f"⚠ Warning: No hand detected in {image_file}, skipping.")
    #                 continue

    #             # Extract hand landmarks
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 landmarks = []
    #                 for lm in hand_landmarks.landmark:
    #                     landmarks.extend([lm.x, lm.y, lm.z])  # Store x, y, z coordinates
                    
    #                 data.append(landmarks)
    #                 labels.append(label)

    #         # ✅ Remove label from missing list if at least one image was processed
    #         if len(image_files) > 0:
    #             missing_labels.discard(label)

    #     # ✅ Check if any labels are still missing
    #     if missing_labels:
    #         print(f"❌ ERROR: The following labels are missing from the dataset: {missing_labels}")

    #     # Convert to DataFrame and save as CSV
    #     df = pd.DataFrame(data)
    #     df['label'] = labels
    #     df.to_csv(output_csv, index=False)
    #     print(f"✅ Saved processed dataset to {output_csv}")

    # def predict_sign(self, frame):
    #     """
    #     Predict the sign from the given frame using hand landmarks instead of images.
    #     """
    #     if not self.model:
    #         print("Error: Model not loaded. Train or load a model first.")
    #         return None

    #     # Convert frame to RGB
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)

    #     # Check if any hands were detected
    #     if not results.multi_hand_landmarks:
    #         return "No hands detected"

    #     # Extract hand landmarks
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         landmarks = []
    #         for lm in hand_landmarks.landmark:
    #             landmarks.extend([lm.x, lm.y, lm.z])  # Flatten x, y, z coordinates

    #         # Convert to NumPy array and expand dimensions for model input
    #         landmarks = np.expand_dims(np.array(landmarks), axis=0)

    #         # Make prediction
    #         prediction = self.model.predict(landmarks)[0]

    #         # Print detailed prediction probabilities
    #         top_indices = prediction.argsort()[-5:][::-1]  # Top 5 predictions
    #         print("\nTop 5 Predictions:")
    #         for idx in top_indices:
    #             print(f"{self.labels[idx]}: {prediction[idx]:.4f}")

    #         # Get the top prediction
    #         predicted_class_idx = np.argmax(prediction)
    #         confidence = prediction[predicted_class_idx]

    #         # Only return prediction if confidence is high enough
    #         if confidence > 0.5:
    #             return self.labels[predicted_class_idx]
        
    #     return "Uncertain"



# Example usage
if __name__ == "__main__":
    detector = SignLanguageDetector()
    
    # Step 1: Download or set up dataset path
    #dataset_dir = detector.download_dataset()
    
    # Step 2: Preprocess the dataset (if needed)
    #processed_dir = detector.preprocess_dataset(dataset_dir)

    detector.preprocess_dataset( "C:\\Aura\\backend\\asl_dataset", output_file="C:\\Aura\\backend\\data.pickle")
    
    # Step 3: Build and train the model
    # Note: Uncomment to train a new model
    #detector.build_model()
    #detector.train_model("C:\\Aura\\backend\\asl_dataset", epochs=10)
    #detector.train_model("C:\\Aura\\backend\\landmarks_dataset.csv", epochs=10) 
    
    # Step 4: Load a pre-trained model
    #detector = SignLanguageDetector("C:\\Aura\\backend\\land_model_final.h5")

    #detector.preprocess_dataset( "C:\\Aura\\backend\\asl_dataset")


    
    
    # Step 5: Run real-time detection
    #detector.run_detection()
    #detector.evaluate_model("C:\\Aura\\backend\\landmarks_dataset.csv")



    #detector.print_model_details()
    #detector.analyze_dataset_distribution("C:\\Aura\\backend\\processed_dataset")