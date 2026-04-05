import os, sys, threading, uuid, subprocess as sp
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

sys.path.insert(0, '/content/pitchlogic/backend')
sys.path.insert(0, '/content/drive/MyDrive/samurai_env/samurai')
sys.path.insert(0, '/content/drive/MyDrive/samurai_env/samurai/sam2')

os.environ.setdefault('YOLO_MODEL_PATH',     '/content/pitchlogic/backend/weights/football/best.pt')
os.environ.setdefault('KEYPOINT_MODEL_PATH', '/content/pitchlogic/backend/weights/keypoints/best.pt')
os.environ.setdefault('SAMURAI_SCRIPT',      '/content/drive/MyDrive/samurai_env/samurai/scripts/demo.py')

from app.pipeline.session_manager import SessionManager
from app.pipeline import tasks

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

OUTPUT_ROOT = Path('/content/pitchlogic/outputs')
UPLOAD_ROOT = Path('/content/pitchlogic/uploads')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
sm = SessionManager(output_root=OUTPUT_ROOT)

def _fire(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()

def _s404(sid):
    s = sm.get_session(sid)
    return (s, None) if s else (None, (jsonify({'error': 'Session not found'}), 404))

_STAGE_LABELS = {
    'auto_detect':        'Auto-detecting players...',
    'samurai_init':       'Initializing tracker...',
    'samurai_running':    'SAMURAI tracking...',
    'samurai_done':       'Tracking complete',
    'loading_video':      'Loading video...',
    'yolo_detection':     'YOLO detection...',
    'camera_motion':      'Camera motion compensation...',
    'keypoint_detection': 'Keypoint detection...',
    'ball_interpolation': 'Ball trajectory interpolation...',
    'speed_calculation':  'Calculating speed & distance...',
    'team_assignment':    'Identifying team colors...',
    'computing_summary':  'Generating summary...',
    'done':               'Done',
}

def _available(session):
    st = session['status']
    if st == 'analysis_done': return ['heatmap', 'speed_chart', 'possession', 'minimap_replay', 'full_replay', 'summary']
    if st == 'tracking_done': return ['analyze']
    if st == 'uploaded':      return ['track']
    return []

# ── Health ────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    gpu = {'available': False, 'name': 'CPU only'}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {'available': True, 'name': torch.cuda.get_device_name(0),
               'memory_gb': round(p.total_memory / 1e9, 1),
               'free_gb': round((p.total_memory - torch.cuda.memory_allocated(0)) / 1e9, 1)}
    return jsonify({'status': 'ok', 'gpu': gpu})

# ── Upload & Frame Analysis ───────────────────────────────────────────────────

@app.route('/api/<sid>/receive_video', methods=['POST'])
def receive_video(sid):
    if 'file' in request.files:
        f = request.files['file']
        dest = UPLOAD_ROOT / f'{sid}_{f.filename}'
        f.save(str(dest))
        video_path = str(dest)
    else:
        data = request.get_json(silent=True) or {}
        video_path = data.get('video_path', '')
        if not os.path.exists(video_path):
            return jsonify({'error': f'video_path not found: {video_path}'}), 400
    if not sm.get_session(sid):
        sm.create_session(sid, video_path)
        sm.session_output_dir(sid).mkdir(parents=True, exist_ok=True)
    else:
        sm.update_status(sid, 'uploaded', stage='video_received')
    print(f'✅ [receive_video] sid={sid} path={video_path}')
    return jsonify({'status': 'received', 'session_id': sid})

@app.route('/api/<sid>/analyze_frame', methods=['POST'])
def analyze_frame(sid):
    session, err = _s404(sid)
    if err: return err
    import cv2
    from ultralytics import YOLO
    data = request.get_json(silent=True) or {}
    t_s = float(data.get('time_in_seconds', 0))
    video_path = session['video_path']
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found: {video_path}'}), 404
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': 'Extract frame failed'}), 500
    model = YOLO(os.environ.get('YOLO_MODEL_PATH', 'weights/football/best.pt'))
    results = model.predict(frame, conf=0.25, verbose=False)[0]
    players = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if results.names[cls].lower() in ('player', 'person', 'referee', 'goalkeeper'):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            players.append({'id': len(players) + 1, 'name': results.names[cls].capitalize(),
                            'bbox': [x1, y1, x2, y2], 'avatar': '👤'})
    print(f'✅ [analyze_frame] sid={sid} detected {len(players)} players')
    return jsonify({'players_data': players,
                    'image_dimensions': {'width': frame.shape[1], 'height': frame.shape[0]}})

# ── Trim ──────────────────────────────────────────────────────────────────────

@app.route('/api/<sid>/trim', methods=['POST'])
def trim_video(sid):
    session, err = _s404(sid)
    if err: return err
    data = request.get_json(silent=True) or {}
    start_t = float(data.get('start', 0))
    end_t   = float(data.get('end', 0))
    original = session['video_path']
    out_path = os.path.join(os.path.dirname(original), f'trimmed_{sid}.mp4')
    print(f'✂️ [trim] sid={sid} {start_t}s -> {end_t}s')
    sp.run(['ffmpeg', '-y', '-i', original, '-ss', str(start_t), '-to', str(end_t), '-c', 'copy', out_path],
           stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    session['video_path'] = out_path
    return jsonify({'status': 'success', 'trimmed_path': out_path})

# ── Pipeline ──────────────────────────────────────────────────────────────────

@app.route('/api/<sid>/auto_start', methods=['POST'])
def auto_start(sid):
    """Auto-pipeline: YOLO detect first frame -> random pick player -> SAMURAI tracking."""
    session, err = _s404(sid)
    if err: return err
    if session['status'] not in ('uploaded',):
        return jsonify({'error': f"Cannot auto_start from: {session['status']}"}), 400
    sm.update_status(sid, 'tracking', progress=0, stage='auto_detect')
    _fire(tasks.run_auto_detect_and_track, sid, session, sm)
    return jsonify({'status': 'auto_started', 'session_id': sid})

@app.route('/api/<sid>/track', methods=['POST'])
def start_tracking(sid):
    session, err = _s404(sid)
    if err: return err
    if session['status'] not in ('uploaded', 'tracking_failed'):
        return jsonify({'error': f"Cannot track from: {session['status']}"}), 400
    body = request.get_json(silent=True) or {}
    if 'x1' in body:
        x1, y1, x2, y2 = int(body['x1']), int(body['y1']), int(body['x2']), int(body['y2'])
        bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
    elif all(k in body for k in ('x', 'y', 'w', 'h')):
        bbox = {k: int(body[k]) for k in ('x', 'y', 'w', 'h')}
    else:
        return jsonify({'error': 'Missing bbox (need x1/y1/x2/y2 or x/y/w/h)'}), 400
    bbox['frame'] = int(body.get('frame', 0))
    sm.update_status(sid, 'tracking', progress=0, stage='samurai_init')
    _fire(tasks.run_samurai_tracking, sid, session, bbox, sm)
    return jsonify({'status': 'tracking_started'})

@app.route('/api/<sid>/analyze', methods=['POST'])
def start_analysis(sid):
    session, err = _s404(sid)
    if err: return err
    if session['status'] != 'tracking_done':
        return jsonify({'error': f"Tracking must complete first. Current: {session['status']}"}), 400
    sm.update_status(sid, 'analyzing', progress=0, stage='yolo_global')
    _fire(tasks.run_global_analysis, sid, session, sm)
    return jsonify({'status': 'analysis_started'})

# ── Generate ──────────────────────────────────────────────────────────────────

def _require_analysis_done(sid):
    s, err = _s404(sid)
    if err: return None, err
    if s['status'] != 'analysis_done':
        return None, (jsonify({'error': 'Analysis not complete'}), 400)
    return s, None

@app.route('/api/<sid>/generate/heatmap', methods=['POST'])
def generate_heatmap(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'heatmap')
    _fire(tasks.run_heatmap, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/speed_chart', methods=['POST'])
def generate_speed_chart(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'speed_chart')
    _fire(tasks.run_speed_chart, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/possession', methods=['POST'])
def generate_possession(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'possession')
    _fire(tasks.run_possession_stats, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/minimap_replay', methods=['POST'])
def generate_minimap_replay(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'minimap_replay')
    _fire(tasks.run_minimap_replay, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/full_replay', methods=['POST'])
def generate_full_replay(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'full_replay')
    _fire(tasks.run_full_replay, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

# ── Status & Files ────────────────────────────────────────────────────────────

@app.route('/api/<sid>/status')
def get_status(sid):
    session, err = _s404(sid)
    if err: return err
    return jsonify({
        'session_id':         sid,
        'status':             session['status'],
        'progress':           session.get('progress', 0),
        'stage':              session.get('stage', ''),
        'stage_label':        _STAGE_LABELS.get(session.get('stage', ''), session.get('stage', '')),
        'error':              session.get('error'),
        'available_features': _available(session),
    })

@app.route('/api/<sid>/task/<tid>')
def task_status(sid, tid):
    t = sm.get_task(sid, tid)
    return (jsonify(t), 200) if t else (jsonify({'error': 'Task not found'}), 404)

@app.route('/api/<sid>/summary')
def summary(sid):
    session, err = _s404(sid)
    if err: return err
    s = session.get('player_summary')
    return (jsonify(s), 200) if s else (jsonify({'error': 'Not available'}), 404)

@app.route('/api/<sid>/file/<tid>')
def serve_file(sid, tid):
    t = sm.get_task(sid, tid)
    if not t or t['status'] != 'done':
        return jsonify({'error': 'File not ready'}), 404
    fp = t.get('file_path')
    if not fp or not os.path.exists(fp):
        return jsonify({'error': 'File missing'}), 404
    return send_file(fp, conditional=True)

@app.route('/api/<sid>/download/<tid>')
def download_file(sid, tid):
    t = sm.get_task(sid, tid)
    if not t or t['status'] != 'done':
        return jsonify({'error': 'File not ready'}), 404
    fp = t.get('file_path')
    if not fp or not os.path.exists(fp):
        return jsonify({'error': 'File missing'}), 404
    return send_file(fp, as_attachment=True, download_name=Path(fp).name)

@app.route('/api/<sid>/download_raw')
def download_raw_video(sid):
    """下载原始上传视频（full_replay 生成失败时的兜底）"""
    session, err = _s404(sid)
    if err: return err
    vp = session.get('video_path', '')
    if not vp or not os.path.exists(vp):
        return jsonify({'error': 'Video file not found'}), 404
    return send_file(vp, as_attachment=True,
                     download_name=f'original_{sid}.mp4',
                     conditional=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
