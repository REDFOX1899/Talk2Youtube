import uuid
import os
import time

def generate_temp_filename(prefix, extension):
    """
    Generates a unique temporary filename
    prefix: a string prefix for the filename
    extension: the file extension (without the dot)
    """
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{unique_id}.{extension}"

def sanitize_filename(filename):
    """
    Sanitizes a filename to make it safe for all operating systems
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Ensure the filename isn't too long
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename

def get_video_id_from_url(url):
    """
    Extracts the video ID from a YouTube URL
    """
    if "youtu.be" in url:
        # Handle shortened URLs
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        # Handle regular URLs
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    return None