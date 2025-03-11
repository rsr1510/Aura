# AURA - ASL Understanding and Recognition Assistant

AURA is a web application that provides real-time American Sign Language (ASL) recognition and speech-to-sign translation capabilities.

## Features

- ASL to Speech Translation
- Speech to Sign Language Translation
- Real-time Video Recognition
- Text-to-Speech Output

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- OpenCV
- TensorFlow
- MediaPipe

## Project Structure

```
aura/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main application file
│   ├── sign_language_model.h5 # Pre-trained ASL recognition model
│   ├── data.pickle           # Label encoder data
│   ├── requirements.txt      # Python dependencies
│   └── static/               # Static files
├── frontend/                  # React frontend
│   ├── src/                  # Source files
│   ├── public/               # Public assets
│   └── package.json          # Dependencies
└── README.md                 # This file
```

## Important Note

This repository includes pre-trained machine learning models required for ASL recognition:
- `backend/sign_language_model.h5`: The main ASL recognition model
- `backend/data.pickle`: Label encoder data for ASL recognition

These files are included to ensure the project works immediately after cloning.

## Setup Instructions

### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Create environment file:
   ```bash
   cp .env.example .env
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Environment Variables

### Frontend (.env)
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8000)

## Deployment

### Backend Deployment
1. Ensure all requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. The ML models (sign_language_model.h5 and data.pickle) are included in the repository
3. Configure your production server (e.g., Gunicorn, Uvicorn) with the appropriate host and port

### Frontend Deployment
1. Build the production version:
   ```bash
   npm run build
   ```
2. Deploy the contents of the `build` directory to your web server
3. Configure your web server to serve the static files
4. Update the `.env` file with the production backend URL

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 