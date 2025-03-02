import pandas as pd
import numpy as np
import streamlit as st
import whisper
import os
import yt_dlp
from streamlit_chat import message
from openai import OpenAI
import time

# Create a local cache directory
WHISPER_CACHE = os.path.join(os.path.dirname(__file__), 'whisper_cache')
os.makedirs(WHISPER_CACHE, exist_ok=True)  # Create if doesn't exist

# Load model with custom cache location
model = whisper.load_model('base', download_root=WHISPER_CACHE)
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

# Function to calculate distances between embeddings
def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
    distance_metric: str = "cosine"
) -> list[float]:
    """Return distances between a query embedding and a list of embeddings."""
    if distance_metric == "cosine":
        return [1 - np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) 
                for emb in embeddings]
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

# yt-dlp function to download YouTube videos
def download_youtube_audio(youtube_url, output_path="youtube_video.mp4"):
    """Download audio from YouTube video using yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            return {
                'title': info.get('title', 'Unknown Title'),
                'id': info.get('id', 'Unknown ID')
            }
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

st.markdown('<h1>Youtube GPT 🤖<small> by <a href="https://codegpt.co">Code GPT</a></small></h1>', unsafe_allow_html=True)
st.write("Start a chat with a YouTube video. Add your OpenAI API Key and paste a YouTube URL in the 'Chat with the video' tab.")

DEFAULT_WIDTH = 80
VIDEO_DATA = "https://youtu.be/bsFXgfbj8Bc"

width = 40

width = max(width, 0.01)
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, 47, side])
container.video(data=VIDEO_DATA)
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
with tab1:
    st.markdown("### How does it work?")
    st.markdown('Read the article to know how it works: [Medium Article](https://medium.com/@dan.avila7/youtube-gpt-start-a-chat-with-a-video-efe92a499e60)')
    st.write("Youtube GPT was written with the following tools:")
    st.markdown("#### Code GPT")
    st.write("All code was written with the help of Code GPT. Visit [codegpt.co](https://codegpt.co) to get the extension.")
    st.markdown("#### Streamlit")
    st.write("The design was written with [Streamlit](https://streamlit.io/).")
    st.markdown("#### Whisper")
    st.write("Video transcription is done by [OpenAI Whisper](https://openai.com/blog/whisper/).")
    st.markdown("#### Embedding")
    st.write('[Embedding](https://platform.openai.com/docs/guides/embeddings) is done via the OpenAI API with "text-embedding-ada-002"')
    st.markdown("#### GPT-3.5")
    st.write('The chat uses the OpenAI API with the GPT-3.5-turbo model')
    st.markdown("""---""")
    st.write('Author: [Daniel Ávila](https://www.linkedin.com/in/daniel-avila-arias/)')
    st.write('Repo: [Github](https://github.com/davila7/youtube-gpt)')
    st.write("This software was developed with Code GPT, for more information visit: https://codegpt.co")
with tab2: 
    st.header("Transcription:")
    if os.path.exists("youtube_video.mp4.mp3"):
        audio_file = open('youtube_video.mp4.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
    if os.path.exists("transcription.csv"):
        df = pd.read_csv('transcription.csv')
        st.write(df)
with tab3:
    st.header("Embedding:")
    if os.path.exists("word_embeddings.csv"):
        df = pd.read_csv('word_embeddings.csv')
        st.write(df)
with tab4:
    # Initialize all session states
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    user_secret = st.text_input(label=":blue[OpenAI API key]",
                                placeholder="Paste your OpenAI API key, sk-",
                                type="password")
    youtube_link = st.text_input(label=":red[YouTube Video Link]",
                                placeholder="Paste YouTube URL here")
    
    # Add video processing section
    if youtube_link and user_secret:
        if not user_secret.startswith('sk-'):
            st.error("Invalid OpenAI API key format")
            st.stop()
            
        if st.button("Analyze Video"):
            try:
                with st.spinner('Processing video...'):
                    # Clear previous data
                    files_to_remove = [
                        "word_embeddings.csv",
                        "transcription.csv",
                        "youtube_video.mp4.mp3",
                        "youtube_video.mp4"
                    ]
                    
                    for file in files_to_remove:
                        if os.path.exists(file):
                            os.remove(file)

                    # Download video using yt-dlp
                    video_info = download_youtube_audio(youtube_link)
                    
                    st.session_state.video_processed = True
                    st.write(f"**Video Title:** {video_info['title']}")
                    st.video(youtube_link)

                    # Check audio file validity
                    audio_file_path = "youtube_video.mp4.mp3"
                    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                        st.error("Failed to download audio. Please try a different video.")
                        st.stop()

                    # Transcribe audio with progress tracking
                    output = model.transcribe(audio_file_path)
                    st.write(f"📝 Transcribed {len(output['segments'])} segments")
                    
                    # Save transcription
                    transcription = {
                        "title": video_info['title'].strip(),
                        "transcription": output['text']
                    }
                    pd.DataFrame([transcription]).to_csv('transcription.csv')
                    segments = output['segments']

                    # Generate embeddings with progress tracking
                    data = []
                    client = OpenAI(api_key=user_secret)
                    progress_bar = st.progress(0)
                    for i, segment in enumerate(segments):
                        response = client.embeddings.create(
                            input=segment["text"].strip(),
                            model="text-embedding-ada-002"
                        )
                        embeddings = response.data[0].embedding
                        meta = {
                            "text": segment["text"].strip(),
                            "start": segment['start'],
                            "end": segment['end'],
                            "embedding": embeddings
                        }
                        data.append(meta)
                        progress_bar.progress((i + 1) / len(segments))
                    
                    pd.DataFrame(data).to_csv('word_embeddings.csv')
                    st.success('✅ Video analysis completed! You can now chat with the video.')
                    st.balloons()

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.info("Make sure you have FFmpeg installed and accessible in your PATH")

    # Chat interface
    st.markdown("---")
    
    def get_text():
        if user_secret and (os.path.exists("word_embeddings.csv") or st.session_state.get('video_processed', False)):
            st.header("Ask me something about the video:")
            input_text = st.text_input("You: ", "", key="input")
            return input_text
        return None
    
    user_input = get_text()

    def get_embedding_text(api_key, prompt):
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=prompt.strip(),
            model="text-embedding-ada-002"
        )
        q_embedding = response.data[0].embedding
        
        df = pd.read_csv('word_embeddings.csv', index_col=0)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)
        
        df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, 'cosine')
        
        returns = []
        for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
            returns.append(row["text"])
        return "\n\n###\n\n".join(returns)

    def generate_response(api_key, prompt):
        client = OpenAI(api_key=api_key)
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are YoutubeGPT, a highly intelligent question answering bot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {str(e)}"

    if user_input:
        try:
            with st.spinner('Finding relevant content...'):
                text_embedding = get_embedding_text(user_secret, user_input)
            
            with st.spinner('Generating response...'):
                df_title = pd.read_csv('transcription.csv')
                string_title = df_title['title'].iloc[0] if not df_title.empty else "Video Title"
                user_input_embedding = f'Using this context: "{string_title}. {text_embedding}", answer the following question: \n{user_input}'
                output = generate_response(user_secret, user_input_embedding)
                
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            
    if st.session_state.get('generated'):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

if __name__ == '__main__':
    pass