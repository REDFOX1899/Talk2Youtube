import os
import openai
from config import OPENAI_API_KEY

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using OpenAI's Whisper API
    Returns the transcribed text
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        
        return response.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Alternative function for local transcription using whisper library
# Uncomment and install whisper if you prefer this method
"""
import whisper

def transcribe_audio_locally(audio_file_path):
    try:
        model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
        result = model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio locally: {e}")
        return None
"""