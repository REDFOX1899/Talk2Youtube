import requests
from config import DEEPSEEK_API_KEY, OPENAI_API_KEY
import openai

def summarize_with_llm(transcript, prompt, provider="openai"):
    """
    Summarizes the transcript using an LLM (OpenAI or Deepseek)
    Returns the generated summary
    """
    if provider == "deepseek":
        return summarize_with_deepseek(transcript, prompt)
    else:
        return summarize_with_openai(transcript, prompt)

def summarize_with_openai(transcript, prompt):
    """
    Summarizes the transcript using OpenAI's API
    Falls back to this if Deepseek isn't configured
    """
    try:
        openai.api_key = OPENAI_API_KEY
        
        system_prompt = f"You are a helpful assistant that provides concise video summaries based on transcripts. {prompt}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def summarize_with_deepseek(transcript, prompt):
    """
    Summarizes the transcript using Deepseek's API
    """
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant that provides concise video summaries based on transcripts. {prompt}"},
                {"role": "user", "content": transcript}
            ],
            "model": "deepseek-chat"  # Update with correct model name
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",  # Update with correct endpoint
            headers=headers,
            json=data
        )
        
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error with Deepseek API: {e}")
        # Fall back to OpenAI if Deepseek fails
        return summarize_with_openai(transcript, prompt)