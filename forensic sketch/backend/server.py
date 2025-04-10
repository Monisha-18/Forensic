from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import sys
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/run-recognition', methods=['GET'])
def run_recognition():
    try:
        print("Recognition endpoint hit!")  # Debugging log

        # Get the full Python path
        python_path = sys.executable  
        script_path = os.path.join(os.getcwd(), "recognize_video.py")

        # Run recognize_video.py and capture output
        result = subprocess.run([python_path, script_path], check=True, capture_output=True, text=True)

        print("Recognition Output:", result.stdout)  # Debugging
        return jsonify({"message": "Recognition started!", "output": result.stdout}), 200
    
    except subprocess.CalledProcessError as e:
        print("Error running recognize_video.py:", e.stderr)
        return jsonify({"error": e.stderr}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)