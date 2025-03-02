# YouTube Video Summarizer API

A simple API that takes a YouTube video URL, transcribes the audio, and generates a summary using LLM (Deepseek or OpenAI).

## Features

- Download audio from YouTube videos
- Transcribe audio using OpenAI's Whisper API
- Generate summaries using Deepseek or OpenAI
- Customize summary style with prompts

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your API keys (see `.env.example`)
6. Run the application:
   ```
   python main.py
   ```

## API Endpoints

### Health Check

```
GET /health
```

Returns the API status.

### Summarize Video

```
POST /summarize
```

Body:

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "prompt": "Give me a bullet point summary of the key points in this video"
}
```

## Example Usage

Using curl:

```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "prompt": "Give me a bullet point summary of the key points in this video"}'
```

Using Python requests:

```python
import requests

response = requests.post(
    "http://localhost:5000/summarize",
    json={
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "prompt": "Give me a bullet point summary of the key points in this video"
    }
)

print(response.json())
```

## Requirements

- Python 3.8+
- OpenAI API key
- Deepseek API key (optional)
