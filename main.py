from app import create_app
from config import DEBUG, PORT, HOST

app = create_app()

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT)