# AURA - ASL Understanding and Recognition Assistant

AURA is an innovative ASL (American Sign Language) communication platform that bridges the gap between the deaf and hearing communities through advanced AI and computer vision technologies.

## 🌟 Key Features

### 1. Bi-Directional Communication
- **Sign Language to Speech**: Real-time translation of ASL gestures to spoken words
- **Speech to Sign Language**: Conversion of spoken words to ASL demonstrations (via 3D avatar)

### 2. Emotion-Based Voice Generation
- Intelligent analysis of signed words and emotions
- Natural and toned voice generation based on detected emotions
- Dynamic voice modulation for more natural-sounding speech

### 3. Custom Gesture Recognition
- Support for custom ASL gestures
- Real-time hand tracking and gesture interpretation
- High-accuracy word gesture recognition

### 4. 3D Avatar Visualization (Experimental)
- Text-to-Sign Language visualization using 3D avatars
- Blender integration for animation generation
- Note: Currently experiencing rendering issues in the animation output

## 🗂️ Project Structure

```
aura/
├── word_gesture/          # Emotion detection module
│   ├── gesture_training.py    # Code to train the gestures for detection
│   ├── action (1).h5          # Trained model for gesture recognition
│   ├── emotion_training.py     # Code to train for detection of emotioins while detecting gestures
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main FastAPI server with API endpoints
│   ├── training.py            # Model training script using ASL dataset
│   ├── testing.py             # Model testing and validation
│   ├── soundtest.py           # Voice modulation and emotion-based generation
│   ├── sign_language_model.h5 # Pre-trained ASL recognition model
│   ├── data.pickle           # Label encoder data
│   ├── requirements.txt      # Python dependencies
│   └── static/               # Static files and generated content
├── frontend/                  # React frontend
│   ├── src/
│   │   └── pages/features/   # Core functionality implementation
│   ├── public/               # Public assets
│   └── package.json          # Node dependencies
└── asl_gestures_data.json    # Custom gesture mappings
```

## 🛠️ Technical Implementation

### Machine Learning Pipeline
1. **Data Collection**: ASL dataset from Kaggle
2. **Preprocessing**: Landmark generation from hand gestures using MediaPipe
3. **Model Training**: Custom neural network for gesture recognition
4. **Emotion Detection**: Integrated emotion analysis for voice modulation
5. **Voice Generation**: Emotion-aware text-to-speech synthesis

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- Blender (for 3D avatar generation)
- OpenCV
- TensorFlow
- MediaPipe

### Installation

1. Clone and setup backend:
```bash
git clone https://github.com/rsr1510/Aura.git
cd Aura
python -m venv .venv
.venv\Scripts\activate  # On Windows
cd backend
pip install -r requirements.txt
```

2. Setup frontend:
```bash
cd ../frontend
npm install
cp .env.example .env  # Configure REACT_APP_API_URL if needed
```

### Running the Application
1. Start backend server:
```bash
cd backend
python app.py
```

2. Start frontend development server:
```bash
cd frontend
npm start
```

## 🔄 Current Status
- Core ASL recognition system: ✅ Complete
- Speech-to-Text conversion: ✅ Complete
- Emotion-based voice generation: ✅ Complete
- 3D Avatar rendering: ⚠️ Partially complete (rendering bug pending)
- Word gesture with emotion detection: ⚠️ Ready but pending integration

## 📝 Note
This project is part of an academic demonstration showcasing the possibilities of AI-powered communication assistance for the deaf and hard of hearing community. The repository includes pre-trained models (`sign_language_model.h5` and `data.pickle`) to ensure immediate functionality after cloning.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details. 
