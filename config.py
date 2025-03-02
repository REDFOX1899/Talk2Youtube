import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "0.0.0.0")

# Temporary Directory for Audio Files
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)