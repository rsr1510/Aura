# from fastapi import APIRouter
# from gtts import gTTS
# import os

# router = APIRouter()

# @router.post("/speak/")
# async def speak_text(text: str):
#     tts = gTTS(text=text, lang="en")
#     tts.save("output.mp3")
#     os.system("start output.mp3")  # Use 'start' for Windows, 'afplay' for Mac, 'mpg321' for Linux
#     return {"status": "spoken"}






from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from gtts import gTTS
import os
import base64
import tempfile
import uvicorn

app = FastAPI()

class TextData(BaseModel):
    text: str

@app.post("/speak")
async def speak(data: TextData):
    if not data.text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        # Create TTS audio
        tts = gTTS(text=data.text, lang='en')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        tts.save(temp_path)
        
        # Convert to base64 for sending to frontend
        with open(temp_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up the temp file
        os.remove(temp_path)
        
        return {
            "status": "success",
            "audio": audio_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("speak:app", host="0.0.0.0", port=8001, reload=True)