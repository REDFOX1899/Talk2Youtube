from flask import Flask, Blueprint, request, jsonify
import os
from services.youtube import download_youtube_audio
from services.transcription import transcribe_audio
from services.llm import summarize_with_llm
from utils.helpers import generate_temp_filename
from config import DEBUG, PORT, HOST

# Initialize the Flask application
app = Flask(__name__)

# Create a Blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return jsonify({"status": "healthy"}), 200

@main_bp.route('/summarize', methods=['POST'])
def summarize_video():
    """Main endpoint to summarize a YouTube video given its URL and custom prompt"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    youtube_url = data.get('youtube_url')
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
    
    prompt = data.get('prompt', 'Summarize this video concisely')
    
    try:
        # Step 1: Download YouTube audio
        audio_file = download_youtube_audio(youtube_url)
        if not audio_file:
            return jsonify({"error": "Failed to download YouTube video"}), 400
        
        # Step 2: Transcribe the audio
        transcript = transcribe_audio(audio_file)
        if not transcript:
            # Clean up temp file
            os.remove(audio_file)
            return jsonify({"error": "Failed to transcribe audio"}), 400
        
        # Step 3: Summarize the transcript using LLM
        summary = summarize_with_llm(transcript, prompt)
        if not summary:
            # Clean up temp file
            os.remove(audio_file)
            return jsonify({"error": "Failed to generate summary"}), 400
        
        # Clean up temp file
        os.remove(audio_file)
        
        # Return the results
        return jsonify({
            "youtube_url": youtube_url,
            "transcript_preview": transcript[:500] + "..." if len(transcript) > 500 else transcript,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Register the blueprint with the app
app.register_blueprint(main_bp)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({"error": "Method not allowed"}), 405

def create_app():
    app = Flask(__name__)
    
    # Register blueprint
    app.register_blueprint(main_bp)
    
    # Register error handlers
    app.errorhandler(404)(not_found)
    app.errorhandler(500)(server_error)
    app.errorhandler(400)(bad_request)
    app.errorhandler(405)(method_not_allowed)
    
    return app

# Run the application
if __name__ == '__main__':
    print(f"Starting YouTube Summarizer API on {HOST}:{PORT}")
    app.run(debug=DEBUG, host=HOST, port=PORT)