import gtts
import subprocess
import os

# Define keyword-based emotion detection
emotion_keywords = {
    "joy": ["happy", "excited", "joyful", "awesome", "great", "love"],
    "anger": ["angry", "furious", "annoyed", "rage", "mad"],
    "sadness": ["sad", "depressed", "upset", "unhappy", "down"],
    "neutral": ["okay", "fine", "normal", "alright"],
    "fear": ["scared", "afraid", "nervous", "anxious"],
    "surprise": ["wow", "amazing", "shocked", "unbelievable"],
}

# Emotion-to-Pitch Mapping (Final Adjustments)
emotion_pitch_map = {
    "joy": "asetrate=44100*0.65",  # Slightly high pitch, but not too fast
    "anger": "asetrate=44100*1.02",  # Mild increase
    "sadness": "asetrate=44100*0.55",  # Slower and lower pitch
    "neutral": "asetrate=44100*1.0",  # Normal pitch
    "fear": "asetrate=44100*0.90",  # Deeper, slightly slower
    "surprise": "asetrate=44100*1.05",  # Boosted, but controlled
}



def detect_emotion(text):
    text_lower = text.lower()
    for emotion, words in emotion_keywords.items():
        if any(word in text_lower for word in words):
            return emotion
    return "neutral"

# Input Text
#text = "good morning everyone this is a great day to start with!"
text = "i feel so down today"
# Detect Emotion
emotion = detect_emotion(text)
print(f"Detected Emotion: {emotion}")

# Get Adjusted Pitch Filter
pitch_filter = emotion_pitch_map.get(emotion, "asetrate=44100*1.0")

# Generate Speech with gTTS
tts = gtts.gTTS(text, lang="en")
output_file = "soundoutput.mp3"
tts.save(output_file)

# Adjust Pitch using FFmpeg
adjusted_output = "adjusted_output.mp3"
ffmpeg_command = ["ffmpeg", "-i", output_file, "-af", pitch_filter, adjusted_output]

try:
    subprocess.run(ffmpeg_command, check=True)
    print(f"Pitch-adjusted audio saved as: {adjusted_output}")
except subprocess.CalledProcessError as e:
    print(f"Error in pitch adjustment: {e}")

# Optional: Play the Adjusted Audio
try:
    subprocess.run(["ffplay", "-nodisp", "-autoexit", adjusted_output])
except FileNotFoundError:
    print("FFplay not found. Install FFmpeg to play the audio.")
