import os
import time
import uuid
import cv2
from flask import Blueprint, request, jsonify, current_app, send_from_directory, url_for
from werkzeug.utils import secure_filename

api = Blueprint('api', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mock database for video sessions
video_sessions = {}

@api.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        # Create a unique video ID for this session
        video_id = uuid.uuid4().hex[:12]
        
        filepath = os.path.join(upload_folder, f"{video_id}_{filename}")
        file.save(filepath)
        
        video_sessions[video_id] = {
            'filename': filename,
            'filepath': filepath,
            'status': 'uploaded'
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'video_id': video_id,
            'filename': filename
        }), 200
        
    return jsonify({'error': 'File type not allowed'}), 400

@api.route('/api/trim', methods=['POST'])
def trim_video():
    data = request.json
    video_id = data.get('video_id')
    start = data.get('start')
    end = data.get('end')
    
    if not video_id or start is None or end is None:
        return jsonify({'error': 'Missing parameters'}), 400
        
    if video_id not in video_sessions:
        return jsonify({'error': 'Video session not found'}), 404
        
    video_sessions[video_id]['trim'] = {'start': start, 'end': end}
    video_sessions[video_id]['status'] = 'trimmed'
    
    return jsonify({
        'message': 'Trim settings saved',
        'video_id': video_id
    }), 200

@api.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.json
    video_id = data.get('video_id')
    time_in_seconds = float(data.get('time_in_seconds', 0))
    
    if not video_id or video_id not in video_sessions:
        return jsonify({'error': 'Video session not found'}), 404
        
    filepath = video_sessions[video_id]['filepath']
    print(f"=== analyze_frame Started ===")
    print(f"Video Path: {filepath}, Time requested: {time_in_seconds}")
    
    # 1. Use OpenCV to locate and read the specific frame
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video file', 'success': False}), 500
        
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        print("❌ Failed to extract frame")
        return jsonify({'error': 'Could not read frame at specified time'}), 500
        
    print("✅ Frame extracted successfully")
    height, width, _ = frame.shape
    
    # 2. Save the frame to a temporary location
    frames_dir = os.path.join(os.path.dirname(current_app.config['UPLOAD_FOLDER']), 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        
    timestamp = int(time.time() * 1000)
    frame_filename = f"frame_{video_id}_{timestamp}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)
    cv2.imwrite(frame_path, frame)
    cap.release()
    print(f"✅ Frame image saved: {frame_path}")
    
    # 3. Use Roboflow YOLO approach (Fallback to mock if not configured)
    players_data = []
    
    try:
        import requests
        import base64
        
        rf_api_key = os.environ.get("ROBOFLOW_API_KEY")
        rf_model_id = os.environ.get("ROBOFLOW_MODEL_ID")
        rf_version = os.environ.get("ROBOFLOW_VERSION", "1")
        
        if rf_api_key and rf_model_id:
            print("🤖 Starting Roboflow Analysis via REST API...")
            
            # Encode image to base64
            with open(frame_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('ascii')
                
            # Construct Roboflow Infer API URL
            # Note: The model_id format is usually "workspace/project" or just "project"
            url = f"https://detect.roboflow.com/{rf_model_id}/{rf_version}?api_key={rf_api_key}"
            
            # Post request
            resp = requests.post(
                url, 
                data=image_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if resp.status_code != 200:
                raise Exception(f"Roboflow API Error {resp.status_code}: {resp.text}")
                
            prediction = resp.json()
            predictions_list = prediction.get("predictions", [])
            print(f"✅ Roboflow analysis complete, detected {len(predictions_list)} objects")
            
            for index, pred in enumerate(predictions_list):
                # Only keep 'player' class — skip referees, balls, etc.
                if pred.get("class", "").lower() == "player":
                    x_center, y_center = pred["x"], pred["y"]
                    w, h = pred["width"], pred["height"]
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    players_data.append({
                        "id": index + 1,
                        "name": f"Player {index+1}",
                        "number": "?",
                        "avatar": "⚽",
                        "bbox": [x1, y1, x2, y2]
                    })
        else:
            raise ImportError("Roboflow keys not provided in ENV")
            
    except Exception as e:
        print(f"⚠️ Roboflow not configured or failed ({str(e)}), generating mock fallback...")
        # Generate some plausible bounding boxes based on image dimensions
        players_data = [
            { "id": 1, "name": "Marcus R.", "number": 10, "avatar": "⚽", "bbox": [int(width*0.2), int(height*0.4), int(width*0.25), int(height*0.55)] },
            { "id": 2, "name": "James K.", "number": 7, "avatar": "🏃", "bbox": [int(width*0.4), int(height*0.5), int(width*0.45), int(height*0.65)] },
            { "id": 3, "name": "Oscar T.", "number": 4, "avatar": "🧤", "bbox": [int(width*0.6), int(height*0.3), int(width*0.65), int(height*0.45)] },
            { "id": 4, "name": "Daniel P.", "number": 9, "avatar": "🦵", "bbox": [int(width*0.8), int(height*0.6), int(width*0.85), int(height*0.75)] }
        ]
    
    # 4. Return results for frontend split-view
    public_url = url_for('api.get_frame', filename=frame_filename, _external=False)
    
    return jsonify({
        "success": True,
        "annotated_frame_url": public_url,
        "players_data": players_data,
        "image_dimensions": {"width": width, "height": height}
    }), 200

@api.route('/api/frames/<filename>', methods=['GET'])
def get_frame(filename):
    frames_dir = os.path.join(os.path.dirname(current_app.config['UPLOAD_FOLDER']), 'frames')
    return send_from_directory(frames_dir, filename)

@api.route('/api/frame/<video_id>', methods=['GET'])
def extract_frame_on_fly(video_id):
    """
    GET /api/frame/123?t=4.5
    Extracts frame at t seconds and returns directly.
    """
    t = float(request.args.get('t', 0))
    if video_id not in video_sessions:
        return jsonify({'error': 'Video not found'}), 404
        
    filepath = video_sessions[video_id]['filepath']
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 500
        
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': 'Could not read frame'}), 500
        
    # Encode to memory and return
    _, buffer = cv2.imencode('.jpg', frame)
    from flask import Response
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@api.route('/api/players/<video_id>', methods=['GET'])
def get_detected_players(video_id):
    # This might not be used anymore if we pass data via router, but keep for fallback
    time.sleep(0.5)
    mock_players = [
        { 'id': 1, 'name': 'Marcus R.', 'number': 10, 'avatar': '⚽', 'detected': True },
        { 'id': 2, 'name': 'James K.', 'number': 7, 'avatar': '🏃', 'detected': True },
        { 'id': 3, 'name': 'Oscar T.', 'number': 4, 'avatar': '🧤', 'detected': True },
        { 'id': 4, 'name': 'Daniel P.', 'number': 9, 'avatar': '🦵', 'detected': True }
    ]
    return jsonify({'video_id': video_id, 'players': mock_players}), 200

@api.route('/api/analyze', methods=['POST'])
def analyze_player():
    data = request.json
    video_id = data.get('video_id')
    player_id = data.get('player_id')
    coordinates = data.get('coordinates', None) # New optional field
    
    if not video_id or not player_id:
        return jsonify({'error': 'Missing parameters'}), 400
        
    print(f"Starting analysis for video {video_id}, player {player_id}, coords: {coordinates}")
    
    return jsonify({
        'message': 'Analysis started',
        'task_id': f"task_{int(time.time())}"
    }), 200

@api.route('/api/videos/<video_id>', methods=['GET'])
def get_video(video_id):
    if video_id not in video_sessions:
        return jsonify({'error': 'Video not found'}), 404
        
    session = video_sessions[video_id]
    directory = os.path.dirname(session['filepath'])
    filename = os.path.basename(session['filepath'])
    
    return send_from_directory(directory, filename)
