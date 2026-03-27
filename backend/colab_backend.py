import os, sys, threading, uuid, subprocess as sp
from pathlib import Path
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import torch

# ── 1. 环境配置 ──
sys.path.insert(0, '/content/pitchlogic/backend')
# SAMURAI & SAM2 依赖路径
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

def _get_available_features(session: dict) -> list:
    status = session['status']
    if status == 'analysis_done':
        return ['heatmap', 'speed_chart', 'possession', 'minimap_replay', 'full_replay', 'summary']
    if status == 'tracking_done':
        return ['analyze']
    if status == 'uploaded':
        return ['track']
    return []

@app.route('/health')
def health(): return jsonify({'status': 'ok'})

@app.route('/api/<sid>/receive_video', methods=['POST'])
def receive_video(sid):
    if 'file' in request.files:
        f = request.files['file']
        dest = UPLOAD_ROOT / f'{sid}_{f.filename}'
        f.save(str(dest))
        video_path = str(dest)
    else:
        video_path = (request.get_json(silent=True) or {}).get('video_path', '')
    if not sm.get_session(sid):
        sm.create_session(sid, video_path)
        sm.session_output_dir(sid).mkdir(parents=True, exist_ok=True)
    return jsonify({'status': 'received', 'session_id': sid})

@app.route('/api/<sid>/analyze_frame', methods=['POST'])
def analyze_frame(sid):
    session, err = _s404(sid)
    if err: return err
    import cv2
    from ultralytics import YOLO
    t_s = float((request.get_json(silent=True) or {}).get('time_in_seconds', 0))
    cap = cv2.VideoCapture(session['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * (cap.get(cv2.CAP_PROP_FPS) or 24)))
    ret, frame = cap.read(); cap.release()
    if not ret: return jsonify({'error': 'frame error'}), 500
    res = YOLO(os.environ.get('YOLO_MODEL_PATH','weights/football/best.pt')).predict(frame, conf=0.25, verbose=False)[0]
    ps = []
    for b in res.boxes:
        c = int(b.cls[0])
        if res.names[c].lower() in ('player','person','referee','goalkeeper'):
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            ps.append({'id':len(ps)+1, 'name':res.names[c].capitalize(), 'bbox':[x1,y1,x2,y2], 'avatar':'👤'})
    return jsonify({'players_data': ps, 'image_dimensions': {'width': frame.shape[1], 'height': frame.shape[0]}})

@app.route('/api/<sid>/trim', methods=['POST'])
def trim_video(sid):
    session, err = _s404(sid)
    if err: return err
    data = request.get_json(silent=True) or {}
    out = os.path.join(os.path.dirname(session['video_path']), f'trimmed_{sid}.mp4')
    sp.run(['ffmpeg','-y','-i',session['video_path'],'-ss',str(data.get('start',0)),'-to',str(data.get('end',0)),'-c','copy',out], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    session['video_path'] = out
    return jsonify({'status': 'success'})

@app.route('/api/<sid>/track', methods=['POST'])
def start_tracking(sid):
    session, err = _s404(sid)
    if err: return err
    body = request.get_json(silent=True) or {}
    x1,y1,x2,y2 = int(body['x1']), int(body['y1']), int(body['x2']), int(body['y2'])
    bbox = {'x':x1, 'y':y1, 'w':x2-x1, 'h':y2-y1, 'frame': int(body.get('frame',0))}
    sm.update_status(sid, 'tracking', progress=0, stage='samurai_init')
    _fire(tasks.run_samurai_tracking, sid, session, bbox, sm)
    return jsonify({'status': 'tracking_started'})

@app.route('/api/<sid>/analyze', methods=['POST'])
def start_analysis(sid):
    session, err = _s404(sid)
    if err: return err
    sm.update_status(sid, 'analyzing', progress=0, stage='yolo_global')
    _fire(tasks.run_global_analysis, sid, session, sm)
    return jsonify({'status': 'analysis_started'})

@app.route('/api/<sid>/status')
def get_status(sid):
    s, err = _s404(sid)
    if not s: return err
    return jsonify({
        'status':             s['status'],
        'progress':           s.get('progress', 0),
        'stage':              s.get('stage', ''),
        'stage_label':        _STAGE_LABELS.get(s.get('stage', ''), s.get('stage', '')),
        'available_features': _get_available_features(s),
        'error':              s.get('error')
    })

@app.route('/api/<sid>/task/<tid>')
def task_status(sid, tid):
    t = sm.get_task(sid, tid)
    return jsonify(t) if t else (jsonify({'error': '404'}), 404)

@app.route('/api/<sid>/summary')
def summary(sid):
    s, err = _s404(sid)
    return jsonify(s.get('player_summary')) if s else err

# ── On-demand Generation ───────────────────────────────────────────────────

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

@app.route('/api/<sid>/file/<tid>')
def serve_file(sid, tid):
    t = sm.get_task(sid, tid)
    if t and t['status'] == 'done': return send_file(t['file_path'])
    return (jsonify({'error': '404'}), 404)

@app.route('/api/<sid>/download/<tid>')
def download_file(sid, tid):
    t = sm.get_task(sid, tid)
    if t and t['status'] == 'done':
        return send_file(t['file_path'], as_attachment=True, download_name=Path(t['file_path']).name)
    return (jsonify({'error': '404'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
