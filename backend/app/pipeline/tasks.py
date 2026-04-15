"""
tasks.py - 所有后台任务实现（Flask 线程版）

任务分三层：
  1. run_samurai_tracking  → SAMURAI 追踪
  2. run_global_analysis   → YOLO + 摄像机补偿 + 速度 + 队伍
  3. 按需任务（用户点什么生什么）：
       run_heatmap          → 热力图 PNG
       run_speed_chart      → 速度/距离图表 PNG
       run_possession_stats → 控球率饼图 PNG + JSON
       run_minimap_replay   → 小地图轨迹回放 MP4
       run_sprint_analysis  → 冲刺爆发统计 PNG + JSON
       run_defensive_line   → 防线渗透统计 PNG + JSON
"""

# Standard imports
import os
import subprocess
import pickle
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Try-wrapped heavy imports
try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = sns = None

from .session_manager import SessionManager
from .analysis_core import (
    Tracker,
    CameraMovementEstimator,
    stream_video_chunks,
    read_frames_at_indices,
    read_video,
    KeypointDetector,
    ViewTransformer,
    AccurateSpeedEstimator,
    TeamAssigner,
    SmartBallPossessionDetector,
    save_video,
    bgr_to_hex,
    render_minimap_frame,
    measure_distance,
)

try:
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from sports.configs.soccer import SoccerPitchConfiguration
    import supervision as sv
    HAS_SPORTS = True
except ImportError:
    HAS_SPORTS = False

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_model_path(env_var: str, preferred_name: str) -> str:
    configured = os.environ.get(env_var)
    if configured:
        return configured

    return str(REPO_ROOT / preferred_name)


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ── 环境变量配置（部署时设置）────────────────────────────────────────────────
MODEL_PATH          = _resolve_model_path("YOLO_MODEL_PATH", "soccana_best.pt")
KEYPOINT_MODEL_PATH = _resolve_model_path("KEYPOINT_MODEL_PATH", "soccana_kpts_best.pt")
SAMURAI_SCRIPT      = os.environ.get("SAMURAI_SCRIPT", "samurai/run_samurai.py")

SHORT_VIDEO_FRAMES = 3000  # ≤3000 frames (~2min@24fps): read once into RAM for speed
                            # >3000 frames: use streaming to avoid OOM


# ═══════════════════════════════════════════════════════════════════════
# 阶段 1：SAMURAI 追踪
# ═══════════════════════════════════════════════════════════════════════

def run_samurai_tracking(session_id: str, session: dict,
                         player_bbox: dict, sm: SessionManager):
    """
    调用 SAMURAI 脚本追踪前端选定的球员，结果存为 samurai_tracking.pkl。
    加速版：使用 FFmpeg 降分辨率（0.5）和跳帧（stride=5，目前设保守点，能快5x），
    提升追踪速度（实测提速10x-20x），然后插值还原。
    """
    try:
        import pandas as pd
        video_path   = session["video_path"]
        output_dir   = sm.session_output_dir(session_id)
        cache_path   = output_dir / "samurai_tracking.pkl"

        sm.update_status(session_id, "tracking", progress=10, stage="samurai_running")

        start_index = int(player_bbox.get("frame", 0))
        frames_dir = output_dir / "samurai_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        sm.update_status(session_id, "tracking", progress=15, stage="extracting_frames")
        if cv2 is None:
            raise ImportError("cv2 is required for frame extraction")
            
        # ── 高清变极速抽取：缩放0.5 & 第10帧抽1帧 ──
        RESIZE_FACTOR = 0.5
        SKIP_STEP = 10
        
        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_orig_frames <= 0:
            total_orig_frames = 1500
        cap.release()

        new_w, new_h = int(orig_w * RESIZE_FACTOR), int(orig_h * RESIZE_FACTOR)

        # e.g select='gte(n, 120)*not(mod(n-120, 5))'
        vf = f"select='gte(n\\,{start_index})*not(mod(n-{start_index}\\,{SKIP_STEP}))',scale={new_w}:{new_h}"
        cmd_ext = [
            "ffmpeg", "-i", video_path, "-y",
            "-vf", vf, "-vsync", "0", "-q:v", "2",
            str(frames_dir / "%05d.jpg")
        ]
        res = subprocess.run(cmd_ext, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg extract failed: {res.stderr}")

        sam_total_frames = len(list(frames_dir.glob("*.jpg")))
        if sam_total_frames == 0:
            raise RuntimeError("No frames extracted by FFmpeg")

        # Create temporary txt for demo.py, scale bbox
        bx, by = player_bbox['x'] * RESIZE_FACTOR, player_bbox['y'] * RESIZE_FACTOR
        bw, bh = player_bbox['w'] * RESIZE_FACTOR, player_bbox['h'] * RESIZE_FACTOR
        init_txt = output_dir / "input_bbox.txt"
        with open(init_txt, "w") as f:
            f.write(f"{bx},{by},{bw},{bh}\n")
            
        temp_video = output_dir / "samurai_temp.mp4"
        
        cmd = [
            "python", SAMURAI_SCRIPT,
            "--video_path",  str(frames_dir),
            "--txt_path",    str(init_txt),
            "--video_output_path", str(temp_video),
            "--model_path",  os.environ.get("SAM2_MODEL_PATH", "sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
        ]
        env = os.environ.copy()
        samurai_root = str(Path(SAMURAI_SCRIPT).parent.parent)
        env["PYTHONPATH"] = f"{samurai_root}:{samurai_root}/sam2:" + env.get("PYTHONPATH", "")
        env["HYDRA_FULL_ERROR"] = "1"
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, env=env, cwd=samurai_root
        )

        if result.returncode != 0:
            full_error_log = result.stderr if len(result.stderr) < 3000 else result.stderr[-3000:]
            raise RuntimeError(f"SAMURAI exited {result.returncode}:\n{full_error_log}")

        res_txt = output_dir / "samurai_temp_bboxes.txt"
        if not res_txt.exists():
            raise FileNotFoundError(f"SAMURAI did not produce output bboxes at {res_txt}")

        # 解析并插值
        scale_back = 1.0 / RESIZE_FACTOR
        sparse_bboxes = {}
        with open(res_txt, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    fid = int(parts[0])
                    original_fid = start_index + (fid * SKIP_STEP)
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if w > 0 and h > 0:
                        sparse_bboxes[original_fid] = [x * scale_back, y * scale_back, w * scale_back, h * scale_back]

        end_index = max(sparse_bboxes.keys()) if sparse_bboxes else start_index
        df = pd.DataFrame(index=range(start_index, end_index + 1), columns=['x', 'y', 'w', 'h'])
        for f_idx, box in sparse_bboxes.items():
            if f_idx <= end_index:
                df.loc[f_idx] = box
        
        df = df.astype(float).interpolate(method='linear', limit_direction='both').bfill().ffill()
        
        bboxes_dict = {}
        for f_idx, row in df.iterrows():
            bboxes_dict[f_idx] = (row['x'], row['y'], row['w'], row['h'])

        with open(cache_path, "wb") as f:
            pickle.dump({"bboxes": bboxes_dict}, f)

        sm.update_status(
            session_id, "tracking_done",
            progress=100, stage="samurai_done",
            samurai_cache_path=str(cache_path),
            samurai_tracked_frames=len(bboxes_dict),
        )

    except Exception as exc:
        sm.update_status(session_id, "tracking_failed",
                         error=str(exc), stage="samurai_error")
        _log_error("SAMURAI tracking", session_id, exc)


# ═══════════════════════════════════════════════════════════════════════
# 阶段 2：全局 YOLO 分析（所有按需功能的数据基础）
# ═══════════════════════════════════════════════════════════════════════

def run_global_analysis(session_id: str, session: dict, sm: SessionManager):
    """
    流水线：读取视频 → YOLO检测 → 摄像机补偿 → 关键点/透视变换
            → 速度计算 → 队伍颜色 → 球权检测 → 摘要
    结果缓存为 tracks.pkl，按需任务直接读取，不重复跑 YOLO。
    """
    try:
        import time as _time
        _t_total = _time.perf_counter()
        def _bench(label, t0):
            elapsed = _time.perf_counter() - t0
            print(f"[BENCH] {label:<25}: {elapsed:.1f}s")
            return elapsed

        video_path   = session["video_path"]
        samurai_pkl  = session["samurai_cache_path"]
        output_dir   = sm.session_output_dir(session_id)
        tracks_cache = output_dir / "tracks.pkl"

        # ── 1. 读取视频元数据（不加载帧到内存）────────────────────────
        sm.update_status(session_id, "analyzing", progress=5, stage="loading_video")
        cap_meta = cv2.VideoCapture(video_path)
        fps    = cap_meta.get(cv2.CAP_PROP_FPS) or 24
        total  = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        _vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
        _vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap_meta.release()
        if total <= 0:
            total = 1500
        print(f"[INFO] Video: {total} frames @ {fps:.1f}fps, {_vid_w}x{_vid_h}")

        with open(samurai_pkl, "rb") as f:
            samurai_data = pickle.load(f)
        tracked_bboxes = samurai_data["bboxes"]   # {frame_idx: (x,y,w,h)}
        tb_keys = sorted(tracked_bboxes.keys())
        print(f"[INFO] SAMURAI bboxes: {len(tracked_bboxes)} frames, "
              f"range [{tb_keys[0] if tb_keys else 'N/A'} – {tb_keys[-1] if tb_keys else 'N/A'}]")

        # ── 2-4. 检测 + 光流 + 关键点（短视频一次读入内存，长视频流式）────────
        sm.update_status(session_id, "analyzing", progress=10, stage="yolo_detection")
        _require_file(MODEL_PATH, "YOLO model")
        _require_file(KEYPOINT_MODEL_PATH, "Keypoint model")
        if not HAS_SPORTS:
            raise ImportError("sports library is required for keypoint and minimap analysis")
        tracker = Tracker(MODEL_PATH)

        if total <= SHORT_VIDEO_FRAMES:
            # 短视频：一次读入内存，三个模块共享同一份帧（省去2次重复IO）
            print(f"[INFO] Short video ({total} frames) — loading into RAM for speed")
            _t = _time.perf_counter()
            frames = read_video(video_path)
            tracks = tracker.get_object_tracks(frames)
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            tracker.add_position_to_tracks(tracks)
            _bench("yolo_detection", _t)

            sm.update_status(session_id, "analyzing", progress=35, stage="camera_motion")
            _t = _time.perf_counter()
            cam = CameraMovementEstimator(frames[0])
            cam_mov = cam.get_camera_movement(frames)
            cam.add_adjust_positions_to_tracks(tracks, cam_mov)
            _bench("camera_motion", _t)

            sm.update_status(session_id, "analyzing", progress=50, stage="keypoint_detection")
            _t = _time.perf_counter()
            kp  = KeypointDetector(KEYPOINT_MODEL_PATH)
            vt  = ViewTransformer()
            kps = kp.predict(frames, cam_movement=cam_mov)
            vt.add_transformed_position_to_tracks(tracks, kps)
            vt.interpolate_2d_positions(tracks)
            _bench("keypoint_detection", _t)

            del frames  # 释放内存
            import gc; gc.collect()

        else:
            # 长视频：流式处理，内存恒定
            print(f"[INFO] Long video ({total} frames) — using streaming mode")
            _t = _time.perf_counter()
            tracks = tracker.get_object_tracks_streamed(video_path, total)
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            tracker.add_position_to_tracks(tracks)
            _bench("yolo_detection", _t)

            sm.update_status(session_id, "analyzing", progress=35, stage="camera_motion")
            _t = _time.perf_counter()
            cam     = CameraMovementEstimator.from_video_path(video_path)
            cam_mov = cam.get_camera_movement_streamed(video_path, total)
            cam.add_adjust_positions_to_tracks(tracks, cam_mov)
            _bench("camera_motion", _t)

            sm.update_status(session_id, "analyzing", progress=50, stage="keypoint_detection")
            _t = _time.perf_counter()
            kp  = KeypointDetector(KEYPOINT_MODEL_PATH)
            vt  = ViewTransformer()
            kps = kp.predict_streamed(video_path, total, cam_movement=cam_mov)
            vt.add_transformed_position_to_tracks(tracks, kps)
            vt.interpolate_2d_positions(tracks)
            _bench("keypoint_detection", _t)

        # ── 5. 速度 & 距离 ───────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=70, stage="speed_calculation")
        _t = _time.perf_counter()
        speed_est = AccurateSpeedEstimator()
        speed_est.add_speed_and_distance_to_tracks(tracks)
        _bench("speed_calculation", _t)

        # ── 7. 队伍颜色 + 球权检测 ────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=80, stage="team_assignment")
        team_assigner = TeamAssigner()
        team_control  = []
        poss_detector = SmartBallPossessionDetector(fps=fps, video_w=_vid_w, video_h=_vid_h)
        ball_history  = []

        # 多帧聚合初始化队伍颜色（流式 seek 采样 8 帧）
        _t = _time.perf_counter()
        team_assigner.assign_team_color_from_video(video_path, tracks["players"], n_samples=8)
        team_color_initialized = bool(team_assigner.kmeans is not None)
        _bench("team_color_init", _t)
        if team_color_initialized:
            print(f"[INFO] Team colors initialized from 8 sampled frames")
        else:
            print("[WARN] Team color initialization failed — all players assigned to team 1")

        if team_color_initialized:
            # ── 多帧投票：按索引 seek 采样帧，不缓存全部帧 ──
            _t = _time.perf_counter()
            from collections import Counter
            player_vote_dict = {}
            SAMPLE_STEP = max(1, total // 20)   # 最多20帧投票，短视频无需120帧
            vote_indices = list(range(0, total, SAMPLE_STEP))

            # 分批 seek（每批50帧，避免长时间无进度）
            for chunk_start in range(0, len(vote_indices), 50):
                batch_idxs = vote_indices[chunk_start:chunk_start + 50]
                frame_dict = read_frames_at_indices(video_path, batch_idxs)
                for idx in batch_idxs:
                    frame = frame_dict.get(idx)
                    if frame is None or idx >= len(tracks["players"]):
                        continue
                    for pid, info in tracks["players"][idx].items():
                        if not info or 'bbox' not in info:
                            continue
                        try:
                            color = team_assigner._get_player_color(frame, info['bbox'])
                            if color is not None and not np.all(color == 0):
                                cluster_id = int(team_assigner.kmeans.predict(
                                    color.reshape(1, -1))[0])
                                # 用 _cluster_to_team 映射到 team 1/2，小簇按颜色距离归队
                                predicted = team_assigner._cluster_to_team.get(cluster_id)
                                if predicted is None:
                                    d1 = np.linalg.norm(color - team_assigner.team_colors.get(1, np.zeros(3)))
                                    d2 = np.linalg.norm(color - team_assigner.team_colors.get(2, np.zeros(3)))
                                    predicted = 1 if d1 <= d2 else 2
                                player_vote_dict.setdefault(pid, []).append(predicted)
                        except Exception:
                            pass

            player_final_team = {
                pid: Counter(votes).most_common(1)[0][0]
                for pid, votes in player_vote_dict.items() if votes
            }
            _bench("team_voting", _t)
            print(f"[INFO] Multi-frame voting done: {len(player_final_team)} players assigned")

            _t = _time.perf_counter()
            for i, p_tracks in enumerate(tracks["players"]):
                for pid, info in p_tracks.items():
                    if not info:
                        continue
                    # 投票结果优先，未覆盖的 ID 默认队伍1
                    tid = player_final_team.get(pid, 1)
                    info["team"]       = tid
                    info["team_color"] = team_assigner.team_colors.get(tid, np.array([0,0,0]))

                # 球权检测
                ball_info = tracks["ball"][i].get(1, {})
                ball_bbox = ball_info.get("bbox", [])
                if len(ball_bbox) == 4:
                    bp = ((ball_bbox[0]+ball_bbox[2])/2, (ball_bbox[1]+ball_bbox[3])/2)
                    ball_history.append(bp)
                else:
                    ball_history.append(None)
                if len(ball_history) > 10:
                    ball_history.pop(0)

                ball_transformed_pos = ball_info.get("position_transformed")
                pid_has_ball = poss_detector.detect_possession(
                    i, p_tracks, ball_bbox, ball_history,
                    ball_transformed_pos=ball_transformed_pos)
                conf = poss_detector.get_confidence()
                if (pid_has_ball != -1 and pid_has_ball in p_tracks
                        and conf > 0.3
                        and poss_detector.ball_state == "controlled"):
                    p_tracks[pid_has_ball]["has_ball"]             = True
                    p_tracks[pid_has_ball]["possession_confidence"] = conf
                    team_control.append(p_tracks[pid_has_ball].get("team", 0))
                else:
                    team_control.append(0)

            _bench("possession_detection", _t)

        # ── 8. 摘要 & 缓存 ────────────────────────────────────────────
        _bench("TOTAL", _t_total)
        sm.update_status(session_id, "analyzing", progress=92, stage="computing_summary")
        player_summary = _compute_player_summary(tracks, tracked_bboxes, team_control, fps=fps)

        cache_payload = {
            "tracks":              tracks,
            "tracked_bboxes":      tracked_bboxes,
            "team_control":        team_control,
            "possession_switches": player_summary.get("possession_switches", 0),
            # 颜色序列化（numpy array → list）
            "team_colors": {
                k: v.tolist() for k, v in team_assigner.team_colors.items()
            },
            "team_colors_hex": {
                k: bgr_to_hex(v) for k, v in team_assigner.team_colors.items()
            },
        }
        with open(tracks_cache, "wb") as f:
            pickle.dump(cache_payload, f)

        sm.update_status(
            session_id, "analysis_done",
            progress=100, stage="done",
            tracks_cache_path=str(tracks_cache),
            player_summary=player_summary,
            total_frames=total,
        )

    except Exception as exc:
        sm.update_status(session_id, "analysis_failed",
                         error=str(exc), stage="analysis_error")
        _log_error("Global analysis", session_id, exc)


def _compute_player_summary(tracks: dict, tracked_bboxes: dict,
                            team_control: list, fps: int = 24) -> dict:
    """从 tracks 中提取被追踪球员的关键数字（存入 session，立即可用）"""
    speeds, distances, has_ball_count = [], [], 0

    for i in range(len(tracks["players"])):
        if i not in tracked_bboxes:
            continue
        sx, sy, sw, sh = tracked_bboxes[i]
        matched = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
        if matched:
            speeds.append(matched.get("speed", 0))
            distances.append(matched.get("distance", 0))
            if matched.get("has_ball"):
                has_ball_count += 1

    arr = np.array(team_control)
    t1, t2 = int(np.sum(arr == 1)), int(np.sum(arr == 2))
    total_ctrl = t1 + t2 or 1

    # Count possession switches (team changes between non-zero values)
    possession_switches = 0
    prev_team = 0
    for t in team_control:
        if t != 0 and t != prev_team and prev_team != 0:
            possession_switches += 1
        if t != 0:
            prev_team = t

    return {
        "max_speed_kmh":        round(float(max(speeds)),        1) if speeds else 0,
        "avg_speed_kmh":        round(float(np.mean(speeds)),    1) if speeds else 0,
        "total_distance_m":     round(float(max(distances)),     0) if distances else 0,
        "possession_seconds":   round(has_ball_count / fps,      1),
        "team1_possession_pct": round(t1 / total_ctrl * 100,     1),
        "team2_possession_pct": round(t2 / total_ctrl * 100,     1),
        "possession_switches":  possession_switches,
    }


# ═══════════════════════════════════════════════════════════════════════
# 阶段 3：按需生成任务
# ═══════════════════════════════════════════════════════════════════════

# ── 3a. 热力图 ───────────────────────────────────────────────────────────────

def run_heatmap(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """生成被追踪球员的热力图 PNG（基于 minimap 坐标）"""
    try:
        sm.update_task(session_id, task_id, status="running", progress=10)
        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]

        # 收集 minimap 位置点
        heatmap_pts = []
        for i in range(len(tracks["players"])):
            if i not in tracked_bboxes:
                continue
            sx, sy, sw, sh = tracked_bboxes[i]
            info = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
            if info:
                pos = info.get("position_minimap") or info.get("position_transformed")
                if pos and len(pos) == 2 and not any(np.isnan(p) for p in pos):
                    heatmap_pts.append(pos)

        sm.update_task(session_id, task_id, progress=50)

        output_path = sm.session_output_dir(session_id) / "heatmap.png"

        if HAS_SPORTS and len(heatmap_pts) > 10:
            config = SoccerPitchConfiguration()
            _draw_heatmap_sports(heatmap_pts, config, output_path)
        else:
            _draw_heatmap_matplotlib(heatmap_pts, output_path)

        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("heatmap", session_id, exc)


def _draw_heatmap_sports(points: list, config, output_path: Path):
    """利用 roboflow/sports 库在球场图上绘制 KDE 热力图"""
    pitch = draw_pitch(config=config)
    h, w  = pitch.shape[:2]
    pts   = np.array(points)

    # 用白点构建灰度密度图
    blank  = np.zeros_like(pitch)
    white  = sv.Color.from_hex("#FFFFFF")
    sample = pts[::2] if len(pts) > 300 else pts
    dot_layer = draw_points_on_pitch(config=config, xy=sample,
                                     face_color=white, edge_color=white,
                                     radius=2, pitch=blank)
    gray    = cv2.cvtColor(dot_layer, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (101, 101), 0)
    normed  = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hmap    = cv2.applyColorMap(normed, cv2.COLORMAP_JET)

    mask   = normed > 20
    result = pitch.copy()
    result[mask] = cv2.addWeighted(pitch[mask], 0.35, hmap[mask], 0.65, 0)
    cv2.imwrite(str(output_path), result)


def _draw_heatmap_matplotlib(points: list, output_path: Path):
    """无 sports 库时的备用方案（matplotlib KDE）"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#2d6a1e")
    ax.set_facecolor("#2d6a1e")
    # 球场线条（简化版）
    for rect in [plt.Rectangle((0,0), 105, 68, fill=False, ec="white", lw=2),
                 plt.Rectangle((0,23.2), 16.5, 21.6, fill=False, ec="white"),
                 plt.Rectangle((88.5,23.2), 16.5, 21.6, fill=False, ec="white")]:
        ax.add_patch(rect)

    if len(points) > 10:
        pts = np.array(points)
        sns.kdeplot(x=pts[:,0], y=pts[:,1], fill=True, cmap="Reds",
                    alpha=0.75, ax=ax, bw_adjust=0.5)
        ax.scatter(pts[::8,0], pts[::8,1], c="white", s=4, alpha=0.25)

    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    ax.set_title("Player Heatmap", color="white", fontsize=16, fontweight="bold")
    ax.set_xlabel("Length (m)", color="white"); ax.set_ylabel("Width (m)", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


# ── 3b. 速度 & 距离图表 ────────────────────────────────────────────────────

def run_speed_chart(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """生成速度曲线 + 累计跑动距离双图 PNG"""
    try:
        sm.update_task(session_id, task_id, status="running", progress=10)
        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        cap = cv2.VideoCapture(session["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()

        speeds, distances, times = [], [], []
        for i in range(len(tracks["players"])):
            t = i / fps
            info = None
            if i in tracked_bboxes:
                sx, sy, sw, sh = tracked_bboxes[i]
                info = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
            speeds.append(   info.get("speed",    0) if info else 0)
            prev_d = distances[-1] if distances else 0
            distances.append(info.get("distance", prev_d) if info else prev_d)
            times.append(t)

        sm.update_task(session_id, task_id, progress=55)

        output_path = sm.session_output_dir(session_id) / "speed_chart.png"
        _draw_speed_chart(times, speeds, distances, output_path)

        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("speed_chart", session_id, exc)


def _draw_speed_chart(times: list, speeds: list, distances: list, output_path: Path):
    BG      = "#1a1a2e"
    PANEL   = "#16213e"
    RED     = "#e74c3c"
    BLUE    = "#3498db"
    YELLOW  = "#f1c40f"
    CYAN    = "#1abc9c"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG, sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # 速度图
    ax1.set_facecolor(PANEL)
    ax1.plot(times, speeds, color=RED, linewidth=1.4, label="Speed (km/h)")
    ax1.fill_between(times, speeds, alpha=0.12, color=RED)
    ax1.axhline(y=24, color=YELLOW, ls="--", lw=1.2, alpha=0.8, label="Sprint (24 km/h)")
    ax1.axhline(y=7,  color=CYAN,   ls=":",  lw=1.0, alpha=0.6, label="Walking (7 km/h)")
    # 高亮冲刺区域
    speeds_arr = np.array(speeds)
    ax1.fill_between(times, speeds_arr, 24,
                     where=speeds_arr >= 24, alpha=0.25, color=YELLOW, label="Sprint zone")
    ax1.set_ylabel("Speed (km/h)", color="white", fontsize=11)
    ax1.set_title("Tracked Player — Speed & Distance", color="white",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.legend(facecolor=PANEL, labelcolor="white", fontsize=9, loc="upper right")
    _style_ax(ax1)

    # 距离图（渐变色）
    ax2.set_facecolor(PANEL)
    ax2.plot(times, distances, color=BLUE, linewidth=2, label="Cumulative distance (m)")
    ax2.fill_between(times, distances, alpha=0.15, color=BLUE)
    ax2.set_ylabel("Distance (m)", color="white", fontsize=11)
    ax2.set_xlabel("Time (s)",     color="white", fontsize=11)
    ax2.legend(facecolor=PANEL, labelcolor="white", fontsize=9)
    _style_ax(ax2)

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def _style_ax(ax):
    ax.tick_params(colors="white", labelsize=9)
    ax.grid(alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_color("#444")


# ── 3c. 控球率统计 ────────────────────────────────────────────────────────

def run_possession_stats(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """生成控球率饼图 PNG，并把数字结果存入 task.result"""
    try:
        sm.update_task(session_id, task_id, status="running", progress=20)
        data = _load_cache(session)
        team_control = np.array(data["team_control"])
        hex_colors   = data["team_colors_hex"]

        t1  = int(np.sum(team_control == 1))
        cap = cv2.VideoCapture(session["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()
        t2  = int(np.sum(team_control == 2))
        neu = int(np.sum(team_control == 0))
        total = t1 + t2 + neu or 1

        result_data = {
            "team1_pct":     round(t1  / total * 100, 1),
            "team2_pct":     round(t2  / total * 100, 1),
            "neutral_pct":   round(neu / total * 100, 1),
            "team1_color":   hex_colors.get(1, "#3498db"),
            "team2_color":   hex_colors.get(2, "#e74c3c"),
            "team1_seconds": round(t1  / fps, 1),
            "team2_seconds": round(t2  / fps, 1),
        }

        output_path = sm.session_output_dir(session_id) / "possession_chart.png"
        _draw_possession_chart(result_data, t1, t2, neu, output_path)

        _finish_task(sm, session_id, task_id, output_path, result=result_data)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("possession_stats", session_id, exc)


def _draw_possession_chart(data: dict, t1: int, t2: int, neu: int, output_path: Path):
    BG = "#1a1a2e"
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)

    # 饼图
    ax_pie.set_facecolor(BG)
    sizes  = [t1, t2] + ([neu] if neu else [])
    colors = [data["team1_color"], data["team2_color"]] + (["#555555"] if neu else [])
    labels = [f"Team 1\n{data['team1_pct']}%",
              f"Team 2\n{data['team2_pct']}%"] + ([f"Neutral\n{data['neutral_pct']}%"] if neu else [])

    wedges, texts = ax_pie.pie(
        sizes, labels=labels, colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(color="white", fontsize=12)
    )
    ax_pie.set_title("Ball Possession", color="white", fontsize=14, fontweight="bold", pad=15)

    # 横条形图（时间）
    ax_bar.set_facecolor(BG)
    teams   = ["Team 1", "Team 2"]
    seconds = [data["team1_seconds"], data["team2_seconds"]]
    bars = ax_bar.barh(teams, seconds, color=[data["team1_color"], data["team2_color"]],
                       edgecolor="white", linewidth=0.8, height=0.45)
    for bar, sec, pct in zip(bars, seconds, [data["team1_pct"], data["team2_pct"]]):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{sec}s  ({pct}%)", va="center", color="white", fontsize=11)
    ax_bar.set_xlabel("Possession time (s)", color="white", fontsize=11)
    ax_bar.set_title("Possession by Time", color="white", fontsize=14, fontweight="bold", pad=15)
    ax_bar.tick_params(colors="white"); ax_bar.grid(axis="x", alpha=0.2, color="white")
    for sp in ax_bar.spines.values(): sp.set_color("#444")

    plt.tight_layout(pad=2.5)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── 3d. 小地图轨迹回放 ────────────────────────────────────────────────────

def run_minimap_replay(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    生成小地图轨迹回放 MP4。
    流式写入：逐帧渲染 → 直接写 ffmpeg pipe，内存占用恒定。
    """
    try:
        import subprocess as sp

        sm.update_task(session_id, task_id, status="running", progress=5)

        if not HAS_SPORTS:
            raise RuntimeError("roboflow/sports library required for minimap replay")

        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        team_control  = np.array(data["team_control"])
        hex_t1        = data["team_colors_hex"].get(1, "#3498db")
        hex_t2        = data["team_colors_hex"].get(2, "#e74c3c")
        config        = SoccerPitchConfiguration()
        total_frames  = len(tracks["players"])

        sm.update_task(session_id, task_id, progress=10)

        cap = cv2.VideoCapture(session["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        cap.release()

        # 先渲染第一帧拿到尺寸
        ball_trail: list = []
        ball_info_0 = tracks["ball"][0].get(1, {}) if len(tracks["ball"]) > 0 else {}
        ball_mp_0 = ball_info_0.get("position_minimap") if ball_info_0 else None
        if ball_mp_0:
            ball_trail.append((ball_mp_0[0], ball_mp_0[1]))
        first_frame = render_minimap_frame(0, tracks, tracked_bboxes, team_control,
                                            config, hex_t1, hex_t2,
                                            ball_trail=list(ball_trail))
        mh, mw = first_frame.shape[:2]

        output_path = sm.session_output_dir(session_id) / "minimap_replay.mp4"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{mw}x{mh}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "pipe:",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        proc = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE,
                        stdout=sp.DEVNULL, stderr=sp.DEVNULL)

        # 写第一帧
        proc.stdin.write(first_frame.tobytes())

        # 流式渲染后续帧
        for i in range(1, total_frames):
            ball_info = tracks["ball"][i].get(1, {}) if i < len(tracks["ball"]) else {}
            ball_mp = ball_info.get("position_minimap") if ball_info else None
            if ball_mp:
                ball_trail.append((ball_mp[0], ball_mp[1]))
                if len(ball_trail) > 30:
                    ball_trail.pop(0)
            try:
                frame = render_minimap_frame(i, tracks, tracked_bboxes, team_control,
                                              config, hex_t1, hex_t2,
                                              ball_trail=list(ball_trail))
            except Exception:
                frame = np.zeros((mh, mw, 3), dtype=np.uint8)
            proc.stdin.write(frame.tobytes())

            if i % 120 == 0:
                pct = int(10 + (i / total_frames) * 85)
                sm.update_task(session_id, task_id, progress=pct)

        proc.stdin.close()
        proc.wait()

        print(f"[MEM] Minimap replay done — streamed {total_frames} frames")
        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("minimap_replay", session_id, exc)


def _render_minimap_frame_worker(args):
    """线程安全的单帧渲染包装"""
    i, tracks, tracked_bboxes, team_control, config, hex_t1, hex_t2 = args[:7]
    ball_trail = args[7] if len(args) > 7 else None
    try:
        frame = render_minimap_frame(i, tracks, tracked_bboxes, team_control,
                                     config, hex_t1, hex_t2,
                                     ball_trail=ball_trail)
        return i, frame
    except Exception:
        # 出错返回黑帧，不中断整体渲染
        blank = np.zeros((400, 700, 3), dtype=np.uint8)
        return i, blank


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════


# ── 3e. 全景图合并回放 (Full Replay) ──────────────────────────────────

def run_full_replay(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    生成带有 YOLO 检测框、球员 ID、球权三角、小地图Overlay的全景视频 MP4。
    流式处理：逐帧读取 → 渲染 → 写入 ffmpeg pipe，内存占用恒定（~1帧）。
    """
    try:
        import subprocess as sp

        sm.update_task(session_id, task_id, status="running", progress=5)

        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        team_control  = data["team_control"]
        hex_t1        = data["team_colors_hex"].get(1, "#3498db")
        hex_t2        = data["team_colors_hex"].get(2, "#e74c3c")

        total_frames = len(tracks["players"])
        tb_keys = sorted(tracked_bboxes.keys())
        print(f"[DEBUG REPLAY] tracks_players_len={total_frames}")
        print(f"[DEBUG REPLAY] tracked_bboxes: count={len(tracked_bboxes)}, range=[{tb_keys[0] if tb_keys else 'N/A'}..{tb_keys[-1] if tb_keys else 'N/A'}]")

        sm.update_task(session_id, task_id, progress=10)

        # Prepare helpers (no frame data — lightweight)
        tracker_obj = Tracker(MODEL_PATH)
        team_assigner = TeamAssigner()
        team_colors = {
            1: np.array([int(hex_t1[5:7], 16), int(hex_t1[3:5], 16), int(hex_t1[1:3], 16)]),
            2: np.array([int(hex_t2[5:7], 16), int(hex_t2[3:5], 16), int(hex_t2[1:3], 16)])
        }
        team_assigner.team_colors = team_colors
        config = SoccerPitchConfiguration() if HAS_SPORTS else None

        # Open video for sequential reading
        cap = cv2.VideoCapture(session["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = sm.session_output_dir(session_id) / "full_replay.mp4"

        # Start ffmpeg pipe
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "pipe:",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        proc = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE,
                        stdout=sp.DEVNULL, stderr=sp.DEVNULL)

        sm.update_task(session_id, task_id, progress=15)

        # Stream: read frame → render → write → discard
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Render single frame (same logic as _render_single_frame_worker_full)
            args = (i, frame, tracks, tracked_bboxes, team_control,
                    team_assigner, tracker_obj, config, hex_t1, hex_t2)
            _, rendered = _render_single_frame_worker_full(args)

            proc.stdin.write(rendered.tobytes())

            if i % 60 == 0:
                pct = int(15 + (i / total_frames) * 80)
                sm.update_task(session_id, task_id, progress=pct)

        proc.stdin.close()
        proc.wait()
        cap.release()

        print(f"[MEM] Full replay done — streamed {total_frames} frames, no bulk load")
        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("full_replay", session_id, exc)


def _render_single_frame_worker_full(args):
    """单帧全景渲"""
    i, frame, tracks, tracked_bboxes, team_control, team_assigner, tracker, config, hex_t1, hex_t2 = args
    if frame is None: return i, frame
    frame = frame.copy()
    
    try:
        current_matched_yolo_id = None
        current_yolo_info = None
        samurai_bbox_xyxy = None

        if i in tracked_bboxes:
            sx, sy, sw, sh = tracked_bboxes[i]
            samurai_center = (sx + sw/2, sy + sh/2)
            samurai_bbox_xyxy = [sx, sy, sx+sw, sy+sh]

            min_dist = 100
            for pid, info in tracks['players'][i].items():
                if not info or 'bbox' not in info: continue
                y_bbox = info['bbox']
                y_center = ((y_bbox[0]+y_bbox[2])/2, (y_bbox[1]+y_bbox[3])/2)
                dist = measure_distance(samurai_center, y_center)
                if dist < min_dist:
                    min_dist = dist
                    current_matched_yolo_id = pid
                    current_yolo_info = info

        # Draw other players
        for pid, info in tracks['players'][i].items():
            if not info or pid == current_matched_yolo_id: continue
            if 'bbox' not in info or len(info['bbox']) < 4: continue
            bbox = info['bbox']
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]: continue

            # Use real jersey color stored per-player by run_global_analysis
            raw_color = info.get('team_color')
            if raw_color is not None:
                color = tuple(int(c) for c in raw_color[:3])
            else:
                color = (0, 255, 0)  # fallback green if no color assigned yet

            frame = tracker.draw_ellipse(frame, bbox, color, pid, is_tracked=False)

            if info.get('has_ball', False):
                confidence = info.get('possession_confidence', 1.0)
                if confidence > 0.6:
                    frame = tracker.draw_triangle(frame, bbox, (0, 0, 255))

        # Draw target player
        if samurai_bbox_xyxy is not None:
            y2 = int(samurai_bbox_xyxy[3])
            x_c = int((samurai_bbox_xyxy[0] + samurai_bbox_xyxy[2]) / 2)
            width = samurai_bbox_xyxy[2] - samurai_bbox_xyxy[0]

            if width <= 0:
                return i, frame

            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width), int(0.35*width)),
                       angle=0.0, startAngle=-45, endAngle=235, color=(0, 215, 255), thickness=6)

            team_color = (0, 215, 255)  # default gold outline
            if current_yolo_info:
                raw_color = current_yolo_info.get('team_color')
                if raw_color is not None:
                    team_color = tuple(int(c) for c in raw_color[:3])

            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width*0.9), int(0.35*width*0.9)),
                       angle=0.0, startAngle=-45, endAngle=235, color=team_color, thickness=3)

            if current_yolo_info and current_yolo_info.get('has_ball', False):
                confidence = current_yolo_info.get('possession_confidence', 1.0)
                if confidence > 0.6:
                    frame = tracker.draw_triangle(frame, samurai_bbox_xyxy, (0, 0, 255))

        # Draw ball with bbox rect + confidence label (referees intentionally not drawn)
        for _, info in tracks['ball'][i].items():
            if info and 'bbox' in info:
                bbox = info['bbox']
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    conf = info.get('conf')
                    label = f"Ball {conf:.2f}" if conf is not None else "Ball"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 255), -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        frame = tracker.draw_team_ball_control(frame, i, np.array(team_control), team_assigner.team_colors)

    except Exception as e:
        print(f"Error drawing overlay on frame {i}: {e}")
        pass
        
    return i, frame


# ── 3e. 冲刺爆发统计 ──────────────────────────────────────────────────────────

def run_sprint_analysis(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    统计被追踪球员的高强度冲刺（>25 km/h 且持续 ≥ 2s）：
    输出冲刺次数、平均持续时间及每次冲刺的路径叠加在球场小地图上。
    """
    try:
        sm.update_task(session_id, task_id, status="running", progress=10)
        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        cap = cv2.VideoCapture(session["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        cap.release()

        SPRINT_KMH     = 25.0   # 冲刺速度阈值
        MIN_SPRINT_SEC = 2.0    # 最短持续时间
        min_frames     = int(MIN_SPRINT_SEC * fps)

        # 收集每帧的速度和位置（基于 minimap 坐标）
        speeds, positions = [], []
        for i in range(len(tracks["players"])):
            info = None
            if i in tracked_bboxes:
                sx, sy, sw, sh = tracked_bboxes[i]
                info = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
            speeds.append(info.get("speed", 0) if info else 0)
            pos = None
            if info:
                pos = info.get("position_minimap") or info.get("position_transformed")
            positions.append(pos if (pos and len(pos) == 2
                                     and not any(np.isnan(p) for p in pos)) else None)

        sm.update_task(session_id, task_id, progress=40)

        # 识别冲刺段
        sprint_segments = []   # [(start_frame, end_frame, [positions])]
        in_sprint, sprint_start, sprint_pts = False, 0, []
        for i, spd in enumerate(speeds):
            if spd >= SPRINT_KMH:
                if not in_sprint:
                    in_sprint, sprint_start, sprint_pts = True, i, []
                if positions[i]:
                    sprint_pts.append(positions[i])
            else:
                if in_sprint:
                    if (i - sprint_start) >= min_frames and len(sprint_pts) >= 2:
                        sprint_segments.append((sprint_start, i - 1, list(sprint_pts)))
                    in_sprint, sprint_pts = False, []
        # 处理视频末尾的冲刺
        if in_sprint and (len(speeds) - sprint_start) >= min_frames and len(sprint_pts) >= 2:
            sprint_segments.append((sprint_start, len(speeds) - 1, list(sprint_pts)))

        sm.update_task(session_id, task_id, progress=65)

        durations = [(e - s) / fps for s, e, _ in sprint_segments]
        result_data = {
            "sprint_count":    len(sprint_segments),
            "avg_duration_s":  round(float(np.mean(durations)), 2) if durations else 0.0,
            "max_duration_s":  round(float(np.max(durations)),  2) if durations else 0.0,
            "max_speed_kmh":   round(float(np.max(speeds)),     1),
        }

        output_path = sm.session_output_dir(session_id) / "sprint_analysis.png"
        _draw_sprint_chart(sprint_segments, result_data, output_path)

        _finish_task(sm, session_id, task_id, output_path, result=result_data)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("sprint_analysis", session_id, exc)


def _draw_sprint_chart(segments: list, stats: dict, output_path: Path):
    BG    = "#1a1a2e"
    GREEN = "#2d6a1e"
    RED   = "#e74c3c"
    GOLD  = "#f1c40f"

    if HAS_SPORTS and segments:
        config = SoccerPitchConfiguration()
        pitch  = draw_pitch(config=config)
        colors = [
            sv.Color.from_hex(c) for c in
            ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db",
             "#9b59b6", "#1abc9c", "#e91e63", "#ff5722", "#607d8b"]
        ]
        for idx, (_, _, pts) in enumerate(segments):
            col = colors[idx % len(colors)]
            xy  = np.array(pts)
            pitch = draw_points_on_pitch(config=config, xy=xy,
                                          face_color=col, edge_color=col,
                                          radius=3, pitch=pitch)
            # 连线路径
            for j in range(len(pts) - 1):
                p1 = config.scale(pts[j])
                p2 = config.scale(pts[j+1])
                cv2.line(pitch, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                         (col.r, col.g, col.b), 2)

        # 文字叠加统计
        h = pitch.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"Sprint Bursts:  {stats['sprint_count']}",
            f"Avg Duration:   {stats['avg_duration_s']}s",
            f"Max Duration:   {stats['max_duration_s']}s",
            f"Max Speed:      {stats['max_speed_kmh']} km/h",
        ]
        for li, txt in enumerate(lines):
            cv2.putText(pitch, txt, (10, h - 20 - li * 24),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_path), pitch)
        return

    # 备用：matplotlib
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.set_facecolor(GREEN)
    for rect in [plt.Rectangle((0, 0), 105, 68, fill=False, ec="white", lw=2),
                 plt.Rectangle((0, 23.2), 16.5, 21.6, fill=False, ec="white"),
                 plt.Rectangle((88.5, 23.2), 16.5, 21.6, fill=False, ec="white")]:
        ax.add_patch(rect)
    palette = [RED, "#e67e22", GOLD, "#2ecc71", "#3498db"]
    for idx, (_, _, pts) in enumerate(segments):
        col  = palette[idx % len(palette)]
        pts_arr = np.array(pts)
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], color=col, lw=2, alpha=0.85)
        ax.scatter(pts_arr[0, 0], pts_arr[0, 1], c=col, s=60, zorder=5)
    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    info = (f"Sprint Bursts: {stats['sprint_count']}   "
            f"Avg: {stats['avg_duration_s']}s   "
            f"Max Speed: {stats['max_speed_kmh']} km/h")
    ax.set_title(f"Sprint Analysis\n{info}", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


# ── 3f. 防线渗透统计 ──────────────────────────────────────────────────────────

def run_defensive_line(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    防线渗透统计：在小地图上动态画出对方最后一道防线（按 x 坐标的前四名）。
    标记追踪球员成功越过防线的帧，并给出渗透次数与路径图。
    """
    try:
        sm.update_task(session_id, task_id, status="running", progress=10)
        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]

        sm.update_task(session_id, task_id, progress=25)

        # 确定追踪球员的队伍（取最频繁的队伍标签）
        tracked_teams = []
        for i in range(len(tracks["players"])):
            if i not in tracked_bboxes: continue
            sx, sy, sw, sh = tracked_bboxes[i]
            info = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
            if info and info.get("team"):
                tracked_teams.append(info["team"])
        tracked_team = int(np.median(tracked_teams)) if tracked_teams else 1
        opponent_team = 2 if tracked_team == 1 else 1

        # 每帧：计算追踪球员位置 + 对方防线 x（取对方最深4名的均值）
        frame_data = []   # [(tracked_x, defense_line_x, tracked_pos)]
        for i in range(len(tracks["players"])):
            tracked_pos = None
            if i in tracked_bboxes:
                sx, sy, sw, sh = tracked_bboxes[i]
                info = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
                if info:
                    pos = info.get("position_minimap") or info.get("position_transformed")
                    if pos and len(pos) == 2 and not any(np.isnan(p) for p in pos):
                        tracked_pos = pos

            # 对方球员 x 坐标列表（升序 = 离追踪球员的球门更远的方向）
            opp_xs = []
            for pid, pinfo in tracks["players"][i].items():
                if not pinfo or pinfo.get("team") != opponent_team: continue
                pp = pinfo.get("position_minimap") or pinfo.get("position_transformed")
                if pp and len(pp) == 2 and not any(np.isnan(v) for v in pp):
                    opp_xs.append(pp[0])

            if opp_xs and tracked_pos:
                # 取对方最靠近追踪球员的4名球员均值作为防线
                opp_xs.sort()
                # 假设追踪球员朝 x 增大方向进攻（如不符合可翻转）
                deepest = opp_xs[:4] if tracked_pos[0] < np.mean(opp_xs) else opp_xs[-4:]
                defense_x = float(np.mean(deepest))
                frame_data.append((tracked_pos[0], defense_x, tracked_pos))
            else:
                frame_data.append(None)

        sm.update_task(session_id, task_id, progress=60)

        # 识别渗透事件（追踪球员越过防线并保持 ≥ 3 帧）
        penetrations = []
        behind = False
        consec = 0
        pen_start_pos = None
        for i, fd in enumerate(frame_data):
            if fd is None:
                consec = 0; behind = False; continue
            tx, dx, tp = fd
            if tx > dx:   # 越过防线
                consec += 1
                if not behind:
                    pen_start_pos = tp
                if consec >= 3 and not behind:
                    penetrations.append(tp)
                    behind = True
            else:
                consec = 0; behind = False

        result_data = {
            "penetration_count": len(penetrations),
            "tracked_team":      tracked_team,
            "opponent_team":     opponent_team,
        }

        output_path = sm.session_output_dir(session_id) / "defensive_line.png"
        _draw_defensive_line_chart(frame_data, penetrations, result_data, output_path)

        _finish_task(sm, session_id, task_id, output_path, result=result_data)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("defensive_line", session_id, exc)


def _draw_defensive_line_chart(frame_data: list, penetrations: list,
                                stats: dict, output_path: Path):
    BG    = "#1a1a2e"
    GREEN = "#2d6a1e"
    RED   = "#e74c3c"
    CYAN  = "#00e5ff"
    GOLD  = "#f1c40f"

    # 提取追踪球员轨迹
    track_pts = [fd[2] for fd in frame_data if fd is not None]

    if HAS_SPORTS and track_pts:
        config = SoccerPitchConfiguration()
        pitch  = draw_pitch(config=config)

        # 画追踪球员路径（白色）
        white = sv.Color.from_hex("#FFFFFF")
        track_arr = np.array(track_pts[::3])  # 每3帧取一个点
        if len(track_arr) > 0:
            pitch = draw_points_on_pitch(config=config, xy=track_arr,
                                          face_color=white, edge_color=white,
                                          radius=2, pitch=pitch)

        # 画渗透点（红色爆炸点）
        if penetrations:
            red = sv.Color.from_hex(RED)
            pens_arr = np.array(penetrations)
            pitch = draw_points_on_pitch(config=config, xy=pens_arr,
                                          face_color=red, edge_color=red,
                                          radius=6, pitch=pitch)

        # 防线平均位置（取所有有效帧的中位数）
        dxs = [fd[1] for fd in frame_data if fd is not None]
        if dxs:
            median_dx = float(np.median(dxs))
            p_left  = config.scale([median_dx, 0])
            p_right = config.scale([median_dx, 68])
            cv2.line(pitch,
                     (int(p_left[0]),  int(p_left[1])),
                     (int(p_right[0]), int(p_right[1])),
                     (0, 229, 255), 2, cv2.LINE_AA)

        h = pitch.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"Penetrations:  {stats['penetration_count']}",
            f"Tracked Team:  {stats['tracked_team']}",
            f"Opponent Team: {stats['opponent_team']}",
        ]
        for li, txt in enumerate(lines):
            cv2.putText(pitch, txt, (10, h - 20 - li * 24),
                        font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_path), pitch)
        return

    # 备用 matplotlib
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.set_facecolor(GREEN)
    for rect in [plt.Rectangle((0, 0), 105, 68, fill=False, ec="white", lw=2),
                 plt.Rectangle((0, 23.2), 16.5, 21.6, fill=False, ec="white"),
                 plt.Rectangle((88.5, 23.2), 16.5, 21.6, fill=False, ec="white")]:
        ax.add_patch(rect)

    if track_pts:
        tp = np.array(track_pts[::3])
        ax.plot(tp[:, 0], tp[:, 1], color="white", lw=1.2, alpha=0.6)

    dxs = [fd[1] for fd in frame_data if fd is not None]
    if dxs:
        median_dx = float(np.median(dxs))
        ax.axvline(x=median_dx, color=CYAN, lw=2, ls="--", label="Avg defense line")

    if penetrations:
        px = [p[0] for p in penetrations]; py = [p[1] for p in penetrations]
        ax.scatter(px, py, c=RED, s=120, zorder=6, marker="*", label="Penetration")

    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    ax.set_title(f"Defensive Line Penetration — {stats['penetration_count']} penetration(s)",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor=BG, labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def _load_cache(session: dict) -> dict:
    cache_path = session.get("tracks_cache_path")
    if not cache_path or not Path(cache_path).exists():
        raise FileNotFoundError("tracks.pkl not found — global analysis may have failed")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _find_matched_player(player_frame: dict, target_center: tuple,
                          max_dist: float = 150):
    """在 YOLO 追踪结果中找最接近 SAMURAI 中心点的球员"""
    best_dist, best_info = max_dist, None
    for info in player_frame.values():
        if not info or "bbox" not in info:
            continue
        bbox = info["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        d  = ((cx - target_center[0])**2 + (cy - target_center[1])**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_info = info
    return best_info


def _finish_task(sm: SessionManager, session_id: str, task_id: str,
                 file_path: Path, result: dict = None):
    """任务成功完成时统一写入状态"""
    sm.update_task(
        session_id, task_id,
        status="done",
        progress=100,
        file_path=str(file_path),
        # URL 前端用来直接展示（对应 Flask route /api/<sid>/file/<tid>）
        url=f"/api/{session_id}/file/{task_id}",
        result=result,
    )


def _log_error(name: str, session_id: str, exc: Exception):
    print(f"[ERROR] {name} | session={session_id}")
    print(traceback.format_exc())
