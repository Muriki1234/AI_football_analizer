import os, sys, threading, uuid, shutil, subprocess as sp
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

sys.path.insert(0, '/content/pitchlogic/backend')
sys.path.insert(0, '/content/drive/MyDrive/samurai_env/samurai')
sys.path.insert(0, '/content/drive/MyDrive/samurai_env/samurai/sam2')

os.environ.setdefault('YOLO_MODEL_PATH',     '/content/pitchlogic/soccana_best.pt')
os.environ.setdefault('KEYPOINT_MODEL_PATH', '/content/pitchlogic/soccana_kpts_best.pt')
os.environ.setdefault('SAMURAI_SCRIPT',      '/content/drive/MyDrive/samurai_env/samurai/scripts/demo.py')

from app.pipeline.session_manager import SessionManager
from app.pipeline import tasks

app = Flask(__name__)
CORS(app)  # 覆盖所有路由，包括 /health

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

# ── 分块上传：绕开 Cloudflare Tunnel 的单请求超时 ──────────────────────────
# 前端把视频切成 5-10MB 的块顺序 POST，每块是独立 HTTP 请求（几秒完成），
# 所有块上传完后 POST /upload_complete 触发合并。
#
# 安全性要点：
#   - chunk 写入使用临时文件名 + os.replace 原子落盘，避免半写文件
#   - 合并前校验 chunk 数量完整
#   - 每个 sid 有独立 chunks 目录，不会互相污染

def _chunks_dir(sid: str) -> Path:
    return UPLOAD_ROOT / f'{sid}_chunks'

@app.route('/api/<sid>/upload_chunk', methods=['POST'])
def upload_chunk(sid):
    """接收一个视频分块。字段：chunk_index (int)，file (bytes)"""
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400
    try:
        idx = int(request.form.get('chunk_index', '-1'))
    except ValueError:
        return jsonify({'error': 'chunk_index must be int'}), 400
    if idx < 0:
        return jsonify({'error': 'chunk_index required'}), 400

    d = _chunks_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    final_path = d / f'{idx:05d}'
    tmp_path   = d / f'{idx:05d}.tmp'
    request.files['file'].save(str(tmp_path))
    os.replace(tmp_path, final_path)  # atomic
    return jsonify({'received': idx, 'size': final_path.stat().st_size})

@app.route('/api/<sid>/upload_status', methods=['GET'])
def upload_status(sid):
    """断点续传探针：返回已收到的 chunk index 列表"""
    d = _chunks_dir(sid)
    if not d.exists():
        return jsonify({'received': []})
    received = sorted(int(p.name) for p in d.iterdir()
                      if p.name.isdigit() and len(p.name) == 5)
    return jsonify({'received': received})

@app.route('/api/<sid>/upload_complete', methods=['POST'])
def upload_complete(sid):
    """所有块上传完后调用：校验 + 合并 + 创建 session。

    body: { total_chunks: int, filename: str }
    """
    data = request.get_json(silent=True) or {}
    try:
        total = int(data.get('total_chunks', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'total_chunks must be int'}), 400
    if total <= 0:
        return jsonify({'error': 'total_chunks required'}), 400

    filename = data.get('filename') or f'{sid}.mp4'
    # 防目录穿越：只留 basename
    filename = os.path.basename(filename)

    d = _chunks_dir(sid)
    if not d.exists():
        return jsonify({'error': 'No chunks directory for this session'}), 404

    missing = [i for i in range(total) if not (d / f'{i:05d}').exists()]
    if missing:
        return jsonify({'error': 'Missing chunks', 'missing': missing}), 400

    dest = UPLOAD_ROOT / f'{sid}_{filename}'
    tmp  = UPLOAD_ROOT / f'{sid}_{filename}.merging'
    try:
        with open(tmp, 'wb') as out:
            for i in range(total):
                with open(d / f'{i:05d}', 'rb') as cf:
                    shutil.copyfileobj(cf, out, length=1024 * 1024)
        os.replace(tmp, dest)  # atomic
    except Exception as e:
        if tmp.exists():
            try: tmp.unlink()
            except Exception: pass
        return jsonify({'error': f'Merge failed: {e}'}), 500
    finally:
        # 合并完清理分块目录（失败也清，因为前端会重新上传）
        shutil.rmtree(d, ignore_errors=True)

    video_path = str(dest)
    if not sm.get_session(sid):
        sm.create_session(sid, video_path)
        sm.session_output_dir(sid).mkdir(parents=True, exist_ok=True)
    else:
        sm.update_status(sid, 'uploaded', stage='video_received')
    print(f'✅ [upload_complete] sid={sid} merged {total} chunks -> {video_path} '
          f'({dest.stat().st_size / 1e6:.1f} MB)')
    return jsonify({'status': 'received', 'session_id': sid,
                    'size': dest.stat().st_size})

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
    model = YOLO(os.environ.get('YOLO_MODEL_PATH', '/content/pitchlogic/soccana_best.pt'))
    # conf 0.59 + 只过 player/goalkeeper（裁判/球被过滤掉）
    results = model.predict(frame, conf=0.59, iou=0.45, imgsz=1280, verbose=False)[0]
    players = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if results.names[cls].lower() in ('player', 'goalkeeper'):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pid = len(players) + 1
            players.append({
                'id': pid,
                'number': pid,                 # 前端 #{number} 显示
                'name': f'Player {pid}',
                'bbox': [x1, y1, x2, y2],
                'avatar': '👤',
            })
    # 返回原始帧（不画框）— 前端自己 overlay bbox，保持它原本的颜色样式
    import base64
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_b64 = base64.b64encode(buf).decode('utf-8')
    annotated_frame_url = f'data:image/jpeg;base64,{frame_b64}'
    print(f'✅ [analyze_frame] sid={sid} detected {len(players)} players (conf=0.59)')
    return jsonify({'players_data': players,
                    'image_dimensions': {'width': frame.shape[1], 'height': frame.shape[0]},
                    'annotated_frame_url': annotated_frame_url})

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

@app.route('/api/<sid>/generate/sprint_analysis', methods=['POST'])
def generate_sprint_analysis(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'sprint_analysis')
    _fire(tasks.run_sprint_analysis, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/defensive_line', methods=['POST'])
def generate_defensive_line(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'defensive_line')
    _fire(tasks.run_defensive_line, sid, s, tid, sm)
    return jsonify({'task_id': tid, 'status': 'queued'})

@app.route('/api/<sid>/generate/ai_summary', methods=['POST'])
def generate_ai_summary(sid):
    s, err = _require_analysis_done(sid)
    if err: return err
    tid = sm.create_task(sid, 'ai_summary')
    _fire(tasks.run_ai_summary, sid, s, tid, sm)
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

@app.route('/api/<sid>/raw_video')
def raw_video(sid):
    """直接流式提供原始视频（支持 range 请求，供前端 <video> 标签使用）"""
    session, err = _s404(sid)
    if err: return err
    vp = session.get('video_path', '')
    if not vp or not os.path.exists(vp):
        return jsonify({'error': 'Video file not found'}), 404
    return send_file(vp, mimetype='video/mp4', conditional=True)

@app.route('/api/<sid>/overlay_data')
def overlay_data(sid):
    """
    返回逐帧标注数据（JSON），供前端 Canvas 实时渲染叠加层。
    分析完成即可调用，无需等待视频生成。
    格式:
      { fps, video_w, video_h, t1, t2, frames: [{p, b, t, ctrl}, ...] }
      p = [[id, x1, y1, x2, y2, team, has_ball, speed], ...]
      b = [x1, y1, x2, y2] | null
      t = [x1, y1, x2, y2] | null  (SAMURAI tracked bbox)
      ctrl = 1 | 2 | 0
    """
    session, err = _s404(sid)
    if err: return err
    if session.get('status') not in ('analysis_done',):
        return jsonify({'error': 'Analysis not complete yet'}), 425

    import pickle, cv2 as _cv2
    tracks_path = session.get('tracks_cache_path')
    if not tracks_path or not os.path.exists(tracks_path):
        return jsonify({'error': 'Tracks cache not found'}), 404

    with open(tracks_path, 'rb') as f:
        data = pickle.load(f)

    tracks       = data['tracks']
    tracked_bboxes = data['tracked_bboxes']
    team_control = data['team_control']
    hex_t1       = data.get('team_colors_hex', {}).get(1, '#00BFFF')
    hex_t2       = data.get('team_colors_hex', {}).get(2, '#FF1493')

    # 获取视频尺寸
    vp = session.get('video_path', '')
    video_w, video_h, fps = 1920, 1080, 24.0
    if vp and os.path.exists(vp):
        cap = _cv2.VideoCapture(vp)
        video_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = cap.get(_cv2.CAP_PROP_FPS) or 24.0
        cap.release()

    total = len(tracks['players'])
    frames_out = []
    for i in range(total):
        # 球员
        players_list = []
        for pid, info in tracks['players'][i].items():
            if not info or 'bbox' not in info: continue
            b = info['bbox']
            spd = info.get('speed')
            players_list.append([
                int(pid),
                round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1),
                int(info.get('team', 0)),
                1 if info.get('has_ball') else 0,
                round(float(spd), 1) if spd is not None else 0.0
            ])

        # 足球
        ball_info = tracks['ball'][i].get(1, {})
        ball = None
        if ball_info and 'bbox' in ball_info:
            bb = ball_info['bbox']
            ball = [round(bb[0],1), round(bb[1],1), round(bb[2],1), round(bb[3],1)]

        # 追踪目标（SAMURAI bbox → xyxy）
        tracked = None
        if i in tracked_bboxes:
            sx, sy, sw, sh = tracked_bboxes[i]
            tracked = [round(sx,1), round(sy,1), round(sx+sw,1), round(sy+sh,1)]

        ctrl = int(team_control[i]) if i < len(team_control) else 0
        frames_out.append({'p': players_list, 'b': ball, 't': tracked, 'ctrl': ctrl})

    return jsonify({
        'fps':     fps,
        'video_w': video_w,
        'video_h': video_h,
        't1':      hex_t1,
        't2':      hex_t2,
        'frames':  frames_out,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)
