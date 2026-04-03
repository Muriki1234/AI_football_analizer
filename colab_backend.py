import os, sys, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, make_response
import torch

sys.path.insert(0, '/content/pitchlogic/backend')
sys.path.insert(0, '/content/drive/MyDrive/samurai_env/samurai/sam2')
os.environ.setdefault('YOLO_MODEL_PATH',     '/content/pitchlogic/backend/weights/football/best.pt')
os.environ.setdefault('KEYPOINT_MODEL_PATH', '/content/pitchlogic/backend/weights/keypoints/best.pt')
os.environ.setdefault('SAMURAI_SCRIPT',      '/content/drive/MyDrive/samurai_env/samurai/scripts/demo.py')

from app.pipeline.session_manager import SessionManager
from app.pipeline import tasks

app = Flask(__name__)
OUTPUT_ROOT = Path('/content/pitchlogic/outputs')
UPLOAD_ROOT = Path('/content/pitchlogic/uploads')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
sm = SessionManager(output_root=OUTPUT_ROOT)

def _fire(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()

def _s404(sid):
    s = sm.get_session(sid)
    return (s, None) if s else (None, (jsonify({"error": "Session not found"}), 404))

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        r = make_response()
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        return r

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

@app.route("/health")
def health():
    gpu = {"available": False, "name": "CPU only"}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"available": True, "name": torch.cuda.get_device_name(0),
               "memory_gb": round(p.total_memory/1e9,1),
               "free_gb": round((p.total_memory-torch.cuda.memory_allocated(0))/1e9,1)}
    return jsonify({"status": "ok", "gpu": gpu})

@app.route("/api/<sid>/receive_video", methods=["POST"])
def receive_video(sid):
    if "file" in request.files:
        f = request.files["file"]
        dest = UPLOAD_ROOT / f"{sid}_{f.filename}"
        f.save(str(dest))
        video_path = str(dest)
    else:
        data = request.get_json(silent=True) or {}
        video_path = data.get("video_path", "")
        if not os.path.exists(video_path):
            return jsonify({"error": f"video_path not found: {video_path}"}), 400
    if not sm.get_session(sid):
        sm.create_session(sid, video_path)
        sm.session_output_dir(sid).mkdir(parents=True, exist_ok=True)
    else:
        sm.update_status(sid, "uploaded", stage="video_received")
    print(f"✅ [receive_video] sid={sid} path={video_path}")
    return jsonify({"status": "received", "session_id": sid})

@app.route("/api/<sid>/analyze_frame", methods=["POST"])
def analyze_frame(sid):
    session, err = _s404(sid)
    if err: return err
    import cv2
    from ultralytics import YOLO
    data = request.get_json(silent=True) or {}
    time_s = float(data.get("time_in_seconds", 0))
    video_path = session["video_path"]
    if not os.path.exists(video_path):
        return jsonify({"error": f"Video not found: {video_path}"}), 404
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Extract frame failed"}), 500
    model = YOLO(os.environ.get("YOLO_MODEL_PATH", "weights/football/best.pt"))
    results = model.predict(frame, conf=0.25, verbose=False)[0]
    players = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if results.names[cls].lower() in ("player", "person", "referee", "goalkeeper"):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            players.append({"id": len(players)+1, "name": results.names[cls].capitalize(),
                            "bbox": [x1,y1,x2,y2], "avatar": "👤"})
    print(f"✅ [analyze_frame] sid={sid} 检测到 {len(players)} 人")
    return jsonify({"players_data": players,
                    "image_dimensions": {"width": frame.shape[1], "height": frame.shape[0]}})

@app.route("/api/<sid>/trim", methods=["POST"])
def trim_video(sid):
    session, err = _s404(sid)
    if err: return err
    import subprocess as sp
    data = request.get_json(silent=True) or {}
    start_t = float(data.get("start", 0))
    end_t   = float(data.get("end", 0))
    original_path = session["video_path"]
    out_path = os.path.join(os.path.dirname(original_path), f"trimmed_{sid}.mp4")
    print(f"✂️ [trim] sid={sid} {start_t}s -> {end_t}s")
    sp.run(["ffmpeg", "-y", "-i", original_path,
            "-ss", str(start_t), "-to", str(end_t), "-c", "copy", out_path],
           stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    session["video_path"] = out_path
    return jsonify({"status": "success", "trimmed_path": out_path})

@app.route("/api/<sid>/auto_start", methods=["POST"])
def auto_start(sid):
    """Auto-pipeline: YOLO detect first frame -> random pick player -> SAMURAI tracking."""
    session, err = _s404(sid)
    if err: return err
    if session["status"] not in ("uploaded",):
        return jsonify({"error": f"Cannot auto_start from: {session['status']}"}), 400
    sm.update_status(sid, "tracking", progress=0, stage="auto_detect")
    _fire(tasks.run_auto_detect_and_track, sid, session, sm)
    return jsonify({"status": "auto_started", "session_id": sid})

@app.route("/api/<sid>/track", methods=["POST"])
def start_tracking(sid):
    session, err = _s404(sid)
    if err: return err
    if session["status"] not in ("uploaded", "tracking_failed"):
        return jsonify({"error": f"Cannot track from: {session['status']}"}), 400
    body = request.get_json(silent=True) or {}
    if "x1" in body:
        x1,y1,x2,y2 = int(body["x1"]),int(body["y1"]),int(body["x2"]),int(body["y2"])
        bbox = {"x":x1,"y":y1,"w":x2-x1,"h":y2-y1}
    elif all(k in body for k in ("x","y","w","h")):
        bbox = {k:int(body[k]) for k in ("x","y","w","h")}
    else:
        return jsonify({"error": "Missing bbox"}), 400
    bbox["frame"] = int(body.get("frame", 0))
    sm.update_status(sid, "tracking", progress=0, stage="samurai_init")
    _fire(tasks.run_samurai_tracking, sid, session, bbox, sm)
    return jsonify({"status": "tracking_started"})

@app.route("/api/<sid>/analyze", methods=["POST"])
def start_analysis(sid):
    session, err = _s404(sid)
    if err: return err
    if session["status"] != "tracking_done":
        return jsonify({"error": f"Tracking must complete first. Current: {session['status']}"}), 400
    sm.update_status(sid, "analyzing", progress=0, stage="yolo_global")
    _fire(tasks.run_global_analysis, sid, session, sm)
    return jsonify({"status": "analysis_started"})

GENERATE_MAP = {
    "heatmap":        tasks.run_heatmap,
    "speed_chart":    tasks.run_speed_chart,
    "possession":     tasks.run_possession_stats,
    "minimap_replay": tasks.run_minimap_replay,
    "full_replay":    tasks.run_full_replay,
}

@app.route("/api/<sid>/generate/<gen_type>", methods=["POST"])
def generate(sid, gen_type):
    session, err = _s404(sid)
    if err: return err
    if session["status"] != "analysis_done":
        return jsonify({"error": f"Analysis not done. Status: {session['status']}"}), 400
    fn = GENERATE_MAP.get(gen_type)
    if not fn:
        return jsonify({"error": f"Unknown type: {gen_type}"}), 400
    task_id = sm.create_task(sid, gen_type)
    _fire(fn, sid, session, task_id, sm)
    return jsonify({"task_id": task_id, "status": "queued"})

STAGE_LABELS = {
    "auto_detect":        "Auto-detecting players...",
    "samurai_running":    "SAMURAI tracking...",
    "yolo_detection":     "YOLO detection...",
    "camera_motion":      "Camera motion compensation...",
    "keypoint_detection": "Keypoint detection...",
    "ball_interpolation": "Ball trajectory...",
    "speed_calculation":  "Speed & distance...",
    "team_assignment":    "Team colors...",
    "computing_summary":  "Generating summary...",
    "done":               "Done",
}

def _available(s):
    st = s["status"]
    if st == "analysis_done": return ["heatmap","speed_chart","possession","minimap_replay","full_replay","summary"]
    if st == "tracking_done": return ["analyze"]
    if st == "uploaded":      return ["track"]
    return []

@app.route("/api/<sid>/status")
def get_status(sid):
    session, err = _s404(sid)
    if err: return err
    return jsonify({
        "session_id":         sid,
        "status":             session["status"],
        "progress":           session.get("progress", 0),
        "stage":              session.get("stage", ""),
        "stage_label":        STAGE_LABELS.get(session.get("stage",""), session.get("stage","")),
        "error":              session.get("error"),
        "available_features": _available(session),
    })

@app.route("/api/<sid>/task/<task_id>")
def task_status(sid, task_id):
    t = sm.get_task(sid, task_id)
    return (jsonify(t), 200) if t else (jsonify({"error": "Task not found"}), 404)

@app.route("/api/<sid>/summary")
def summary(sid):
    session, err = _s404(sid)
    if err: return err
    s = session.get("player_summary")
    return (jsonify(s), 200) if s else (jsonify({"error": "Not available"}), 404)

@app.route("/api/<sid>/file/<task_id>")
def serve_file(sid, task_id):
    t = sm.get_task(sid, task_id)
    if not t or t["status"] != "done":
        return jsonify({"error": "File not ready"}), 404
    fp = t.get("file_path")
    if not fp or not os.path.exists(fp):
        return jsonify({"error": "File missing"}), 404
    return send_file(fp, conditional=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, use_reloader=False)