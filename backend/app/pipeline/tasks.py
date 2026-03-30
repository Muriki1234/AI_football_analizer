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
    KeypointDetector,
    ViewTransformer,
    AccurateSpeedEstimator,
    TeamAssigner,
    SmartBallPossessionDetector,
    read_video,
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

# ── 环境变量配置（部署时设置）────────────────────────────────────────────────
MODEL_PATH         = os.environ.get("YOLO_MODEL_PATH",      "weights/football/best.pt")
KEYPOINT_MODEL_PATH= os.environ.get("KEYPOINT_MODEL_PATH",  "weights/keypoints/best.pt")
SAMURAI_SCRIPT     = os.environ.get("SAMURAI_SCRIPT",        "samurai/run_samurai.py")


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
            
        # ── 高清变极速抽取：缩放0.5 & 第5帧抽1帧 ──
        RESIZE_FACTOR = 0.5
        SKIP_STEP = 5
        
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
        video_path   = session["video_path"]
        samurai_pkl  = session["samurai_cache_path"]
        output_dir   = sm.session_output_dir(session_id)
        tracks_cache = output_dir / "tracks.pkl"

        # ── 1. 读视频 ────────────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=5, stage="loading_video")
        frames = read_video(video_path)
        total  = len(frames)

        with open(samurai_pkl, "rb") as f:
            samurai_data = pickle.load(f)
        tracked_bboxes = samurai_data["bboxes"]   # {frame_idx: (x,y,w,h)}

        # ── 2. YOLO 跳帧检测 + 插值 ──────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=10, stage="yolo_detection")
        tracker = Tracker(MODEL_PATH)
        tracks  = tracker.get_object_tracks(frames)
        tracker.add_position_to_tracks(tracks)

        # ── 3. 摄像机运动补偿 ────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=35, stage="camera_motion")
        cam = CameraMovementEstimator(frames[0])
        cam_mov = cam.get_camera_movement(frames)
        cam.add_adjust_positions_to_tracks(tracks, cam_mov)

        # ── 4. 关键点 + 透视变换（生成 minimap 坐标）───────────────
        sm.update_status(session_id, "analyzing", progress=50, stage="keypoint_detection")
        if os.path.exists(KEYPOINT_MODEL_PATH) and HAS_SPORTS:
            kp  = KeypointDetector(KEYPOINT_MODEL_PATH)
            vt  = ViewTransformer()
            kps = kp.predict(frames)
            vt.add_transformed_position_to_tracks(tracks, kps)
            vt.interpolate_2d_positions(tracks)
        else:
            print(f"[WARN] Keypoint model not found or sports lib missing — skipping perspective")

        # ── 5. 足球轨迹插值 ──────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=65, stage="ball_interpolation")
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # ── 6. 速度 & 距离 ───────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=70, stage="speed_calculation")
        speed_est = AccurateSpeedEstimator()
        speed_est.add_speed_and_distance_to_tracks(tracks)

        # ── 7. 队伍颜色 + 球权检测 ────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=80, stage="team_assignment")
        team_assigner = TeamAssigner()
        team_control  = []
        cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()
        poss_detector = SmartBallPossessionDetector(fps=fps)
        ball_history  = []

        if tracks["players"] and tracks["players"][0]:
            team_assigner.assign_team_color(frames[0], tracks["players"][0])

            for i, p_tracks in enumerate(tracks["players"]):
                # 分配队伍 ID 和颜色
                for pid, info in p_tracks.items():
                    if not info:
                        continue
                    tid = team_assigner.get_player_team(frames[i], info["bbox"], pid)
                    info["team"]       = tid
                    info["team_color"] = team_assigner.team_colors.get(tid, np.array([0,0,0]))

                # 球权检测
                ball_bbox = tracks["ball"][i].get(1, {}).get("bbox", [])
                if len(ball_bbox) == 4:
                    bp = ((ball_bbox[0]+ball_bbox[2])/2, (ball_bbox[1]+ball_bbox[3])/2)
                    ball_history.append(bp)
                else:
                    ball_history.append(None)
                if len(ball_history) > 10:
                    ball_history.pop(0)

                pid_has_ball = poss_detector.detect_possession(i, p_tracks, ball_bbox, ball_history)
                conf = poss_detector.get_confidence()
                if pid_has_ball != -1 and pid_has_ball in p_tracks and conf > 0.5:
                    p_tracks[pid_has_ball]["has_ball"]             = True
                    p_tracks[pid_has_ball]["possession_confidence"] = conf
                    team_control.append(p_tracks[pid_has_ball].get("team", 0))
                else:
                    team_control.append(team_control[-1] if team_control else 0)

        # ── 8. 摘要 & 缓存 ────────────────────────────────────────────
        sm.update_status(session_id, "analyzing", progress=92, stage="computing_summary")
        player_summary = _compute_player_summary(tracks, tracked_bboxes, team_control)

        cache_payload = {
            "tracks":         tracks,
            "tracked_bboxes": tracked_bboxes,
            "team_control":   team_control,
            # 颜色序列化（numpy array → list）
            "team_colors": {
                k: v.tolist() for k, v in team_assigner.team_colors.items()
            },
            "team_colors_hex": {
                1: bgr_to_hex(team_assigner.team_colors.get(1, [255,255,255])),
                2: bgr_to_hex(team_assigner.team_colors.get(2, [0,0,0])),
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

    return {
        "max_speed_kmh":        round(float(max(speeds)),        1) if speeds else 0,
        "avg_speed_kmh":        round(float(np.mean(speeds)),    1) if speeds else 0,
        "total_distance_m":     round(float(max(distances)),     0) if distances else 0,
        "possession_seconds":   round(has_ball_count / fps,      1),
        "team1_possession_pct": round(t1 / total_ctrl * 100,     1),
        "team2_possession_pct": round(t2 / total_ctrl * 100,     1),
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
    不含原始视频画面，只渲染球场图 + 球员点 + 被追踪目标 + 足球，体积小、渲染快。
    """
    try:
        sm.update_task(session_id, task_id, status="running", progress=5)

        if not HAS_SPORTS:
            raise RuntimeError("roboflow/sports library required for minimap replay")

        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        team_control  = data["team_control"]
        hex_t1        = data["team_colors_hex"].get(1, "#3498db")
        hex_t2        = data["team_colors_hex"].get(2, "#e74c3c")
        config        = SoccerPitchConfiguration()
        total_frames  = len(tracks["players"])

        sm.update_task(session_id, task_id, progress=10)

        # 渲染所有帧（并行）
        render_args = [
            (i, tracks, tracked_bboxes, np.array(team_control),
             config, hex_t1, hex_t2)
            for i in range(total_frames)
        ]
        frames = [None] * total_frames

        with ThreadPoolExecutor(max_workers=4) as pool:
            for idx, rendered in pool.map(_render_minimap_frame_worker, render_args):
                frames[idx] = rendered
                if idx % 120 == 0:
                    pct = int(10 + (idx / total_frames) * 80)
                    sm.update_task(session_id, task_id, progress=pct)

        output_path = sm.session_output_dir(session_id) / "minimap_replay.mp4"
        cap = cv2.VideoCapture(session["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()
        save_video(frames, str(output_path), fps=fps)

        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("minimap_replay", session_id, exc)


def _render_minimap_frame_worker(args):
    """线程安全的单帧渲染包装"""
    i, tracks, tracked_bboxes, team_control, config, hex_t1, hex_t2 = args
    try:
        frame = render_minimap_frame(i, tracks, tracked_bboxes, team_control,
                                     config, hex_t1, hex_t2)
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
    """生成带有 YOLO 检测框、球员 ID、球权三角、小地图Overlay的全景视频 MP4"""
    try:
        sm.update_task(session_id, task_id, status="running", progress=5)

        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        team_control  = data["team_control"]
        hex_t1        = data["team_colors_hex"].get(1, "#3498db")
        hex_t2        = data["team_colors_hex"].get(2, "#e74c3c")
        
        # We need original frames
        sm.update_task(session_id, task_id, progress=10)
        frames = read_video(session["video_path"])
        total_frames = min(len(frames), len(tracks["players"]))
        
        sm.update_task(session_id, task_id, progress=20)
        
        # Build kwargs for parallel
        tracker = Tracker(MODEL_PATH)
        team_assigner = TeamAssigner()
        team_colors = {
            1: np.array([int(hex_t1[5:7], 16), int(hex_t1[3:5], 16), int(hex_t1[1:3], 16)]),
            2: np.array([int(hex_t2[5:7], 16), int(hex_t2[3:5], 16), int(hex_t2[1:3], 16)])
        }
        team_assigner.team_colors = team_colors

        config = SoccerPitchConfiguration() if HAS_SPORTS else None

        render_args = [
            (i, frames[i], tracks, tracked_bboxes, team_control, team_assigner, tracker, config, hex_t1, hex_t2)
            for i in range(total_frames)
        ]
        
        output_frames = [None] * total_frames
        
        with ThreadPoolExecutor(max_workers=8) as pool:
            for idx, rendered in pool.map(_render_single_frame_worker_full, render_args):
                output_frames[idx] = rendered
                if idx % 60 == 0:
                    pct = int(20 + (idx / total_frames) * 75)
                    sm.update_task(session_id, task_id, progress=pct)

        output_path = sm.session_output_dir(session_id) / "full_replay.mp4"
        cap = cv2.VideoCapture(session["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()
        save_video(output_frames, str(output_path), fps=fps)
        
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

            # Determine color
            tc = info.get('team_color')
            if tc is not None and len(tc) == 3:
                color = (int(tc[0]), int(tc[1]), int(tc[2]))
            else:
                color = (0,0,255)
            
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

            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width), int(0.35*width)),
                       angle=0.0, startAngle=-45, endAngle=235, color=(0, 215, 255), thickness=6)

            team_color = (0, 215, 255)
            if current_yolo_info:
                tc = current_yolo_info.get('team_color', [0, 215, 255])
                if len(tc) == 3:
                    team_color = (int(tc[0]), int(tc[1]), int(tc[2]))

            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width*0.9), int(0.35*width*0.9)),
                       angle=0.0, startAngle=-45, endAngle=235, color=team_color, thickness=3)

            display_id = current_matched_yolo_id if current_matched_yolo_id else "Target"
            text_size, _ = cv2.getTextSize(str(display_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            w_txt, h_txt = text_size

            cv2.rectangle(frame, (int(x_c - w_txt/2 - 5), int(y2 - 10)),
                               (int(x_c + w_txt/2 + 5), int(y2 + 10)), (0, 215, 255), -1)
            cv2.putText(frame, str(display_id), (int(x_c - w_txt/2), int(y2 + 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            if current_yolo_info and 'speed' in current_yolo_info:
                speed = current_yolo_info['speed']
                text = f"{speed:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                box_x1, box_y2 = int(samurai_bbox_xyxy[0]), int(samurai_bbox_xyxy[3])
                cv2.rectangle(frame, (box_x1, box_y2 + 32 - th),
                            (box_x1 + tw + 6, box_y2 + 38), (180, 235, 255), -1)
                cv2.rectangle(frame, (box_x1, box_y2 + 32 - th),
                            (box_x1 + tw + 6, box_y2 + 38), (0, 0, 0), 1)
                cv2.putText(frame, text, (box_x1 + 3, box_y2 + 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if current_yolo_info and current_yolo_info.get('has_ball', False):
                confidence = current_yolo_info.get('possession_confidence', 1.0)
                if confidence > 0.6:
                    frame = tracker.draw_triangle(frame, samurai_bbox_xyxy, (0, 0, 255))

        # Draw referees & ball
        for _, info in tracks['referees'][i].items():
            if info and 'bbox' in info:
                bbox = info['bbox']
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    frame = tracker.draw_ellipse(frame, bbox, (0, 255, 255), None)

        for _, info in tracks['ball'][i].items():
            if info and 'bbox' in info:
                bbox = info['bbox']
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    frame = tracker.draw_triangle(frame, bbox, (0, 255, 0))

        frame = tracker.draw_team_ball_control(frame, i, np.array(team_control), team_assigner.team_colors)

    except Exception as e:
        print(f"Error drawing overlay on frame {i}: {e}")
        pass
        
    return i, frame


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
