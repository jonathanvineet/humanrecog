from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from processor import RecognitionPipeline
from video_stream import VideoProcessor
import threading
import json

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Store processing status
processing_tasks = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_background(task_id, input_path, output_path):
    """Process video in background thread"""
    try:
        processing_tasks[task_id]['status'] = 'processing'
        
        # Initialize pipeline
        pipeline = RecognitionPipeline()
        
        # Process video
        video_processor = VideoProcessor(input_path)
        video_processor.process_with_pipeline(pipeline, output_path)
        
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result_url'] = f'/api/video/{task_id}'
        
    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload and process video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f'{task_id}_{filename}')
        file.save(input_path)
        
        # Prepare output path
        output_filename = f'{task_id}_processed.mp4'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Initialize task tracking
        processing_tasks[task_id] = {
            'status': 'queued',
            'input_file': filename,
            'uploaded_at': str(Path(input_path).stat().st_ctime)
        }
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_background,
            args=(task_id, input_path, output_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'status': 'queued',
            'message': 'Video upload successful. Processing started.'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def check_status(task_id):
    """Check processing status"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'result_url': task.get('result_url', None),
        'error': task.get('error', None)
    }), 200

@app.route('/api/video/<task_id>', methods=['GET'])
def get_processed_video(task_id):
    """Serve processed video"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Video not found'}), 404
    
    if processing_tasks[task_id]['status'] != 'completed':
        return jsonify({'error': 'Video processing not completed'}), 400
    
    # Find the output file
    output_folder = OUTPUT_FOLDER
    for filename in os.listdir(output_folder):
        if filename.startswith(task_id):
            filepath = os.path.join(output_folder, filename)
            return send_file(filepath, mimetype='video/mp4'), 200
    
    return jsonify({'error': 'Processed video file not found'}), 404

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean up old files"""
    try:
        import time
        current_time = time.time()
        retention_seconds = 3600 * 24  # 24 hours
        
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    if current_time - os.path.getmtime(filepath) > retention_seconds:
                        os.remove(filepath)
        
        return jsonify({'status': 'cleanup completed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
