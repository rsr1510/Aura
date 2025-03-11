from transformers import pipeline

speech_engine = pipeline("text-to-speech", model="espnet/fastspeech2_en_ljspeech")



def generate_speech(text, emotion, posture):
    """ Generates speech with tone modulation based on emotion. """
    pitch = 1.0
    speed = 1.0
    
    if emotion == "happy":
        pitch = 1.2
        speed = 1.2
    elif emotion == "sad":
        pitch = 0.8
        speed = 0.8
    elif emotion == "angry":
        pitch = 1.1
        speed = 1.0
    elif emotion == "neutral":
        pitch = 1.0
        speed = 1.0
    
    audio = speech_engine(text, pitch=pitch, speed=speed)
    audio.save("output_speech.wav")
