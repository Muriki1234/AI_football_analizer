"""
analysis.py — Analysis pipeline routes blueprint
Handles: tracking, global analysis, on-demand generation, status, summary, file serving
"""

import os
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file

from ..pipeline.session_manager import SessionManager
from ..pipeline import tasks

analysis = Blueprint('analysis', __name__)

OUTPUT_ROOT = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs'))
OUTPUT_ROOT.mkdir(exist_ok=True)

sm = SessionManager(output_root=OUTPUT_ROOT)


def _fire_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


def _session_or_404(session_id: str):
    session = sm.get_session(session_id)
    if not session:
        return None, (jsonify({"error": "Session not found"}), 404)
    return session, None


# ── Stage 0.5: Register session from existing upload ────────────────────────
# Called right before tracking starts; links the existing video_id to a new
# pipeline session so the two systems share one session_id.

@analysis.route('/api/<session_id>/register', methods=['POST'])
def register_session(session_id: str):
    """
    Links an already-uploaded video (video_id from api.py) to the analysis pipeline.
    Body: { "video_path": "/abs/path/to/video.mp4" }
    """
    data = request.get_json(silent=True) or {}
    video_path = data.get('video_path')
    
    if not video_path:
        # Fallback to in-memory video_sessions from api
        try:
            from .api import video_sessions
            if session_id in video_sessions:
                video_path = video_sessions[session_id].get('filepath')
        except Exception as e:
            print(f"Error fetching from video_sessions: {e}")

    if not video_path or not os.path.exists(str(video_path)):
        return jsonify({'error': 'video_path missing or file not found'}), 400


    # Create session in pipeline (idempotent)
    if not sm.get_session(session_id):
        sm.create_session(session_id, video_path)
        # Pre-create output dir
        sm.session_output_dir(session_id).mkdir(parents=True, exist_ok=True)

    return jsonify({'status': 'registered', 'session_id': session_id}), 200



# ── Stage 1: SAMURAI Tracking ────────────────────────────────────────────────

@analysis.route('/api/<session_id>/track', methods=['POST'])
def start_tracking(session_id: str):
    """
    Frontend sends selected player bbox after frame detection.
    Body: { "x1": 100, "y1": 200, "x2": 180, "y2": 380, "frame": 0 }
    We accept [x1,y1,x2,y2] format and convert to {x,y,w,h} for SAMURAI.
    """
    session, err = _session_or_404(session_id)
    if err: return err

    if session['status'] not in ('uploaded', 'tracking_failed'):
        return jsonify({'error': f"Cannot track from status: {session['status']}"}), 400

    body = request.get_json(silent=True) or {}

    # Support both [x1,y1,x2,y2] and {x,y,w,h} formats
    if 'x1' in body:
        x1, y1, x2, y2 = int(body['x1']), int(body['y1']), int(body['x2']), int(body['y2'])
        player_bbox = {'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1}
    elif all(k in body for k in ('x', 'y', 'w', 'h')):
        player_bbox = {k: int(body[k]) for k in ('x', 'y', 'w', 'h')}
    else:
        return jsonify({'error': 'Missing bbox fields. Send x1,y1,x2,y2 or x,y,w,h'}), 400

    player_bbox['frame'] = int(body.get('frame', 0))

    sm.update_status(session_id, 'tracking', progress=0, stage='samurai_init')
    _fire_thread(tasks.run_samurai_tracking, session_id, session, player_bbox, sm)

    return jsonify({'status': 'tracking_started'}), 200


# ── Stage 2: Global YOLO Analysis ───────────────────────────────────────────

@analysis.route('/api/<session_id>/analyze', methods=['POST'])
def start_analysis(session_id: str):
    session, err = _session_or_404(session_id)
    if err: return err

    if session['status'] != 'tracking_done':
        return jsonify({'error': f"Tracking must complete first. Current: {session['status']}"}), 400

    sm.update_status(session_id, 'analyzing', progress=0, stage='yolo_global')
    _fire_thread(tasks.run_global_analysis, session_id, session, sm)

    return jsonify({'status': 'analysis_started'}), 200


# ── Stage 3: On-demand Generation ───────────────────────────────────────────

def _require_analysis_done(session_id):
    session, err = _session_or_404(session_id)
    if err: return None, err
    if session['status'] != 'analysis_done':
        return None, (jsonify({'error': f"Analysis must complete first. Status: {session['status']}"}), 400)
    return session, None


@analysis.route('/api/<session_id>/generate/heatmap', methods=['POST'])
def generate_heatmap(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'heatmap')
    _fire_thread(tasks.run_heatmap, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/speed_chart', methods=['POST'])
def generate_speed_chart(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'speed_chart')
    _fire_thread(tasks.run_speed_chart, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/possession', methods=['POST'])
def generate_possession(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'possession')
    _fire_thread(tasks.run_possession_stats, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/minimap_replay', methods=['POST'])
def generate_minimap_replay(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'minimap_replay')
    _fire_thread(tasks.run_minimap_replay, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/full_replay', methods=['POST'])
def generate_full_replay(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'full_replay')
    _fire_thread(tasks.run_full_replay, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/sprint_analysis', methods=['POST'])
def generate_sprint_analysis(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'sprint_analysis')
    _fire_thread(tasks.run_sprint_analysis, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/defensive_line', methods=['POST'])
def generate_defensive_line(session_id: str):
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'defensive_line')
    _fire_thread(tasks.run_defensive_line, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


@analysis.route('/api/<session_id>/generate/ai_summary', methods=['POST'])
def generate_ai_summary(session_id: str):
    """Gemini 多模态战术分析报告（带追踪框视频 + 统计 JSON）"""
    session, err = _require_analysis_done(session_id)
    if err: return err
    task_id = sm.create_task(session_id, 'ai_summary')
    _fire_thread(tasks.run_ai_summary, session_id, session, task_id, sm)
    return jsonify({'task_id': task_id, 'status': 'queued'}), 200


# ── Status & Summary ─────────────────────────────────────────────────────────

_STAGE_LABELS = {
    'auto_detect':        'Auto-detecting players...',
    'samurai_init':       'Initializing tracker...',
    'extracting_frames':  'Extracting frames (FFmpeg)...',
    'samurai_running':    'SAMURAI tracking...',
    'samurai_done':       'Tracking complete',
    'loading_video':      'Loading video...',
    'yolo_detection':     'YOLO detection...',
    'camera_motion':      'Camera motion compensation...',
    'keypoint_detection': 'Keypoint detection...',
    'ball_interpolation': 'Ball trajectory interpolation...',
    'speed_calculation':  'Calculating speed & distance...',
    'team_assignment':    'Identifying team colors...',
    'scene_segmentation': 'Detecting match segments...',
    'computing_summary':  'Generating summary...',
    'done':               'Done',
}


def _translate_stage(stage: str) -> str:
    """Map internal stage key → human-readable label.
    Handles static keys and dynamic 'yolo_detection (X/Y frames, ETA ...)' suffix."""
    if not stage:
        return ''
    if stage in _STAGE_LABELS:
        return _STAGE_LABELS[stage]
    # Dynamic YOLO progress label already readable — pass through
    if stage.startswith('yolo_detection ('):
        return stage
    return stage


def _get_available_features(session: dict) -> list:
    status = session['status']
    if status == 'analysis_done':
        return ['heatmap', 'speed_chart', 'possession', 'minimap_replay', 'full_replay',
                'sprint_analysis', 'defensive_line', 'ai_summary', 'summary']
    if status == 'tracking_done':
        return ['analyze']
    if status == 'uploaded':
        return ['track']
    return []


@analysis.route('/api/<session_id>/status', methods=['GET'])
def get_session_status(session_id: str):
    session, err = _session_or_404(session_id)
    if err: return err
    return jsonify({
        'session_id':         session_id,
        'status':             session['status'],
        'progress':           session.get('progress', 0),
        'stage':              session.get('stage', ''),
        'stage_label':        _translate_stage(session.get('stage', '')),
        'error':              session.get('error'),
        'available_features': _get_available_features(session),
    }), 200


@analysis.route('/api/<session_id>/task/<task_id>', methods=['GET'])
def get_task_status(session_id: str, task_id: str):
    task = sm.get_task(session_id, task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task), 200


@analysis.route('/api/<session_id>/summary', methods=['GET'])
def get_player_summary(session_id: str):
    session, err = _session_or_404(session_id)
    if err: return err
    if session['status'] != 'analysis_done':
        return jsonify({'error': 'Analysis not complete'}), 400
    summary = session.get('player_summary')
    if not summary:
        return jsonify({'error': 'Summary not available yet'}), 404
    return jsonify(summary), 200


# ── File Serving ─────────────────────────────────────────────────────────────

@analysis.route('/api/<session_id>/file/<task_id>', methods=['GET'])
def serve_task_file(session_id: str, task_id: str):
    task = sm.get_task(session_id, task_id)
    if not task or task['status'] != 'done':
        return jsonify({'error': 'File not ready'}), 404
    file_path = task.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File missing on disk'}), 404
    return send_file(file_path, conditional=True)


@analysis.route('/api/<session_id>/download/<task_id>', methods=['GET'])
def download_task_file(session_id: str, task_id: str):
    task = sm.get_task(session_id, task_id)
    if not task or task['status'] != 'done':
        return jsonify({'error': 'File not ready'}), 404
    file_path = task.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File missing on disk'}), 404
    return send_file(file_path, as_attachment=True,
                     download_name=Path(file_path).name)
