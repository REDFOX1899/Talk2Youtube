from pytube import YouTube
import os
from utils.helpers import generate_temp_filename

def download_youtube_audio(youtube_url):
    """
    Downloads the audio from a YouTube video and saves it to a temporary file
    Returns the path to the audio file
    """
    try:
        yt = YouTube(youtube_url)
        
        # Get the first audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Generate a temporary filename
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        output_file = generate_temp_filename("audio", "mp4")
        output_path = os.path.join(temp_dir, output_file)
        
        # Download the audio
        audio_stream.download(output_path=temp_dir, filename=output_file)
        
        return output_path
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None