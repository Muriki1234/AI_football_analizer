"""
analysis_core.py - 核心 CV/ML 类库（Flask 版）

在你的 Colab 代码基础上做了以下改动：
  1. 移除所有 Colab/Jupyter 专用调用（files.download 等）
  2. 新增 render_minimap_frame() 供 minimap 回放任务使用
  3. matplotlib 统一用 Agg 后端（在 tasks.py 开头设置）
"""

import os
import cv2

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.interpolate import interp1d, UnivariateSpline
except ImportError:
    interp1d = None
    UnivariateSpline = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    from sports.common.view import ViewTransformer as SportsViewTransformer
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from sports.configs.soccer import SoccerPitchConfiguration
    HAS_SPORTS = True
except ImportError:
    HAS_SPORTS = False

# ── 配置 ────────────────────────────────────────────────────────────────────
YOLO_DETECTION_STRIDE = 1    # 每帧都检测（不跳帧）
YOLO_BATCH_SIZE       = 60   # 单批处理帧数（60 = 更好GPU利用率）
KEYPOINT_STRIDE       = 20   # 每20帧检测一次关键点（提升小地图精度，原为60）
MINIMAP_SMOOTH_WINDOW = 25
SPEED_SMOOTH_WINDOW   = 7
PLAYER_CONF           = 0.25  # 球员/裁判检测置信度
BALL_CONF             = float(os.environ.get("BALL_CONF", "0.59"))


# ═══════════════════════════════════════════════════════════════════════
# 基础工具
# ═══════════════════════════════════════════════════════════════════════

def get_center_of_bbox(bbox):
    return int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)

def get_foot_position(bbox):
    return int((bbox[0]+bbox[2])/2), int(bbox[3])

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def bgr_to_hex(bgr) -> str:
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def clamp_pitch_position(x: float, y: float,
                          x_max: float = 105.0, y_max: float = 68.0):
    """Clamp a transformed pitch coordinate to valid field bounds."""
    return (max(0.0, min(float(x), x_max)),
            max(0.0, min(float(y), y_max)))


def _linear_fill_keypoints(sampled: dict, total_frames: int) -> list:
    """
    Fill un-sampled frames by linearly interpolating each keypoint ID
    between known frames. Nearest-neighbor fallback for edge gaps.

    Uses numpy for interpolation (pandas optional for richer fill).

    Args:
        sampled: {frame_idx: {keypoint_id: [x, y]}} — only sampled frames
        total_frames: total number of frames in the video

    Returns:
        list of length total_frames, each element is {keypoint_id: [x, y]}
    """
    if not sampled:
        return [{} for _ in range(total_frames)]

    all_kids = set()
    for kps in sampled.values():
        all_kids.update(kps.keys())

    result = [{} for _ in range(total_frames)]

    for fi, kps in sampled.items():
        if fi < total_frames:
            result[fi] = dict(kps)

    all_frames = np.arange(total_frames, dtype=float)

    for kid in all_kids:
        known_idx = sorted(fi for fi, kps in sampled.items() if kid in kps and fi < total_frames)
        if len(known_idx) < 2:
            if known_idx:
                fi0 = known_idx[0]
                val = sampled[fi0][kid]
                for fi in range(total_frames):
                    if kid not in result[fi]:
                        result[fi][kid] = list(val)
            continue

        kx = np.array([sampled[fi][kid][0] for fi in known_idx], dtype=float)
        ky = np.array([sampled[fi][kid][1] for fi in known_idx], dtype=float)
        ki = np.array(known_idx, dtype=float)

        interp_x = np.interp(all_frames, ki, kx)
        interp_y = np.interp(all_frames, ki, ky)

        for fi in range(total_frames):
            if kid not in result[fi]:
                result[fi][kid] = [float(interp_x[fi]), float(interp_y[fi])]

    return result


def read_video(video_path: str) -> list:
    """Read all frames from a video file. Uses CAP_FFMPEG backend for reliability."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def stream_video_chunks(video_path: str, chunk_size: int = 500):
    """Generator: yields (start_frame_idx, frames_chunk) without loading full video into RAM."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    start_idx = 0
    while True:
        chunk = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame)
        if not chunk:
            break
        yield start_idx, chunk
        start_idx += len(chunk)
        if len(chunk) < chunk_size:
            break
    cap.release()


def read_frames_at_indices(video_path: str, indices) -> dict:
    """Read specific frames from video by seeking. Returns {frame_idx: frame} dict."""
    indices = sorted(set(int(i) for i in indices if i >= 0))
    result = {}
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            result[idx] = frame
    cap.release()
    return result


def save_video(frames: list, output_path: str, fps: float = 24):
    """Save frames to H.264 mp4 via ffmpeg pipe — avoids mp4v temp-file FPS drift."""
    if not frames:
        return
    import subprocess
    h, w = frames[0].shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write((frame if frame is not None else black).tobytes())
    proc.stdin.close()
    proc.wait()

def put_text_pil(img, text: str, position: tuple, color: tuple, font_size: int = 28):
    """用 PIL 绘制文字（支持中文）"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("SimHei.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    fill = (int(color[2]), int(color[1]), int(color[0]))  # BGR → RGB
    draw.text(position, text, font=font, fill=fill)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def interpolate_ball_positions_spline(ball_positions: list) -> list:
    """
    Interpolate missing ball detections.
    - Gaps <= 30 frames: cubic spline (smooth curve matching ball physics)
    - Gaps > 30 frames: linear fallback (spline oscillates badly over long gaps)
    - Detected frames are preserved exactly (no smoothing of real detections)
    """
    n = len(ball_positions)
    raw = [ball_positions[i].get(1, {}).get("bbox") for i in range(n)]

    known_idx = [i for i, b in enumerate(raw) if b is not None]
    if len(known_idx) < 2:
        return ball_positions

    coords = {c: np.array([raw[i][j] for i in known_idx], dtype=float)
              for j, c in enumerate(["x1", "y1", "x2", "y2"])}
    idx_arr = np.array(known_idx, dtype=float)

    result = list(ball_positions)

    # Group missing indices into contiguous gap segments
    missing = sorted(set(range(n)) - set(known_idx))
    if not missing:
        return result

    gaps = []
    seg_start = prev = missing[0]
    for m in missing[1:]:
        if m != prev + 1:
            gaps.append((seg_start, prev))
            seg_start = m
        prev = m
    gaps.append((seg_start, prev))

    # Fit global cubic spline on known points (only if enough points)
    if len(known_idx) >= 4:
        splines = {c: UnivariateSpline(idx_arr, coords[c], k=3, s=0, ext=3)
                   for c in ["x1", "y1", "x2", "y2"]}
    else:
        splines = None

    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start + 1

        if splines is not None and gap_len <= 30:
            # Cubic spline for short gaps
            for fi in range(gap_start, gap_end + 1):
                bbox = [float(splines[c](fi)) for c in ["x1", "y1", "x2", "y2"]]
                result[fi] = {1: {"bbox": bbox}}
        else:
            # Linear fallback for long gaps or insufficient known points
            before = [k for k in known_idx if k < gap_start]
            after  = [k for k in known_idx if k > gap_end]
            if not before or not after:
                nearest = before[-1] if before else after[0]
                for fi in range(gap_start, gap_end + 1):
                    result[fi] = {1: {"bbox": list(raw[nearest])}}
            else:
                k0, k1 = before[-1], after[0]
                b0, b1 = raw[k0], raw[k1]
                span = k1 - k0
                for fi in range(gap_start, gap_end + 1):
                    t = (fi - k0) / span
                    bbox = [b0[j] + t * (b1[j] - b0[j]) for j in range(4)]
                    result[fi] = {1: {"bbox": bbox}}

    return result


# ═══════════════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════════════

class Tracker:
    def __init__(self, model_path: str):
        self.model        = YOLO(model_path)
        self.tracker      = sv.ByteTrack()
        self.class_names  = self.model.names
        self.player_id = self.ball_id = self.referee_id = None
        for i, name in self.class_names.items():
            n = name.lower()
            if   "player" in n or "person" in n: self.player_id   = i
            elif "ball"   in n:                   self.ball_id     = i
            elif "referee" in n:                  self.referee_id  = i
        if self.player_id is None: self.player_id = 0

    def get_object_tracks(self, frames: list) -> dict:
        total = len(frames)
        tracks = {"players":  [{} for _ in range(total)],
                  "referees": [{} for _ in range(total)],
                  "ball":     [{} for _ in range(total)]}

        det_indices = list(range(0, total, YOLO_DETECTION_STRIDE))
        det_dict    = {}

        for i in range(0, len(det_indices), YOLO_BATCH_SIZE):
            batch_idx    = det_indices[i:i+YOLO_BATCH_SIZE]
            batch_frames = [frames[idx] for idx in batch_idx]
            results      = self.model.predict(batch_frames, conf=PLAYER_CONF, iou=0.45, verbose=False, half=True, imgsz=1280)
            for res, fidx in zip(results, batch_idx):
                det_dict[fidx] = res

        for fidx, det in det_dict.items():
            ds = sv.Detections.from_ultralytics(det)
            for k, cid in enumerate(ds.class_id):
                if "goalkeeper" in self.class_names[cid].lower():
                    ds.class_id[k] = self.player_id

            # Frontend 不需要裁判检测，避免被画框或误参与后续展示
            if self.referee_id is not None and len(ds) > 0:
                ds = ds[ds.class_id != self.referee_id]

            d_tracks = self.tracker.update_with_detections(ds)
            tracks["players"][fidx]  = {}
            tracks["referees"][fidx] = {}
            tracks["ball"][fidx]     = {}

            for d in d_tracks:
                bbox, cid, tid = d[0].tolist(), d[3], d[4]
                if cid == self.player_id:
                    tracks["players"][fidx][tid] = {"bbox": bbox}

            # Ball: apply BALL_CONF post-filter
            ball_confs = det.boxes.conf.tolist() if det.boxes is not None else []
            for d, conf_val in zip(ds, ball_confs):
                if d[3] == self.ball_id and conf_val >= BALL_CONF:
                    tracks["ball"][fidx][1] = {"bbox": d[0].tolist(), "conf": conf_val}

        self._interpolate_tracks(tracks, total)
        return tracks

    def _interpolate_tracks(self, tracks: dict, total_frames: int):
        for obj in ("players", "referees"):
            all_ids = set()
            for fd in tracks[obj]: all_ids.update(fd.keys())
            for tid in all_ids:
                fidxs, bboxes = [], []
                for fi, fd in enumerate(tracks[obj]):
                    if fd and tid in fd:
                        b = fd[tid]["bbox"]
                        if len(b)==4 and b[2]>b[0] and b[3]>b[1]:
                            fidxs.append(fi); bboxes.append(b)
                if len(fidxs) < 2: continue
                bboxes = np.array(bboxes)
                for fi in range(fidxs[0], fidxs[-1]+1):
                    if not tracks[obj][fi]: tracks[obj][fi] = {}
                    if tid not in tracks[obj][fi]:
                        ib = [float(interp1d(fidxs, bboxes[:,c],
                                             kind="linear")(fi)) for c in range(4)]
                        if ib[2]>ib[0] and ib[3]>ib[1]:
                            tracks[obj][fi][tid] = {"bbox": ib}

    def get_object_tracks_streamed(self, video_path: str, total_frames: int,
                                    chunk_size: int = 500) -> dict:
        """流式版本：分块处理视频，不把所有帧加载到内存。ByteTrack 跨块保持连续状态。"""
        tracks = {"players":  [{} for _ in range(total_frames)],
                  "referees": [{} for _ in range(total_frames)],
                  "ball":     [{} for _ in range(total_frames)]}

        for start_idx, chunk in stream_video_chunks(video_path, chunk_size):
            # YOLO 推理：对块内帧批处理
            det_local = list(range(0, len(chunk), YOLO_DETECTION_STRIDE))
            det_dict  = {}
            for i in range(0, len(det_local), YOLO_BATCH_SIZE):
                batch_l  = det_local[i:i+YOLO_BATCH_SIZE]
                results  = self.model.predict([chunk[j] for j in batch_l],
                                              conf=PLAYER_CONF, iou=0.45, verbose=False, half=True, imgsz=1280)
                for res, local_idx in zip(results, batch_l):
                    det_dict[local_idx] = res

            # ByteTrack 更新：必须按帧序处理以维持追踪状态（跨块也连续）
            for local_idx in sorted(det_dict.keys()):
                global_idx = start_idx + local_idx
                if global_idx >= total_frames:
                    break
                ds = sv.Detections.from_ultralytics(det_dict[local_idx])
                for k, cid in enumerate(ds.class_id):
                    if "goalkeeper" in self.class_names[cid].lower():
                        ds.class_id[k] = self.player_id

                if self.referee_id is not None and len(ds) > 0:
                    ds = ds[ds.class_id != self.referee_id]

                d_tracks = self.tracker.update_with_detections(ds)
                for d in d_tracks:
                    bbox, cid, tid = d[0].tolist(), d[3], d[4]
                    if cid == self.player_id:
                        tracks["players"][global_idx][tid] = {"bbox": bbox}
                # Ball: apply BALL_CONF post-filter
                ball_confs = det_dict[local_idx].boxes.conf.tolist() if det_dict[local_idx].boxes is not None else []
                for d, conf_val in zip(ds, ball_confs):
                    if d[3] == self.ball_id and conf_val >= BALL_CONF:
                        tracks["ball"][global_idx][1] = {"bbox": d[0].tolist(), "conf": conf_val}

        self._interpolate_tracks(tracks, total_frames)
        return tracks

    def add_position_to_tracks(self, tracks: dict):
        for obj, otracks in tracks.items():
            for fnum, track in enumerate(otracks):
                if not track: continue
                for tid, info in track.items():
                    if "bbox" in info:
                        info["position"] = (get_center_of_bbox(info["bbox"])
                                            if obj == "ball"
                                            else get_foot_position(info["bbox"]))

    def interpolate_ball_positions(self, ball_positions: list) -> list:
        """Interpolate missing ball detections using cubic spline (short gaps) or linear (long gaps)."""
        return interpolate_ball_positions_spline(ball_positions)


    def draw_ellipse(self, frame, bbox, color, track_id, is_tracked=False):
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        width = int(bbox[2] - bbox[0])
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x_c = int((bbox[0] + bbox[2]) / 2)
        triangle_points = np.array([[x_c, y], [x_c - 10, y - 20], [x_c + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_colors):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        t1_color = team_colors.get(1, (0, 255, 0))
        t2_color = team_colors.get(2, (0, 0, 255))
        if hasattr(t1_color, "tolist"): t1_color = tuple(map(int, t1_color.tolist()))
        if hasattr(t2_color, "tolist"): t2_color = tuple(map(int, t2_color.tolist()))
        if team_1_num_frames + team_2_num_frames > 0:
            team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
            team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
            cv2.putText(frame, f"Team 1: {team_1*100:.0f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, t1_color, 3)
            cv2.putText(frame, f"Team 2: {team_2*100:.0f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, t2_color, 3)
        return frame

# ═══════════════════════════════════════════════════════════════════════
# CameraMovementEstimator
# ═══════════════════════════════════════════════════════════════════════

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask  = np.zeros_like(gray)
        mask[:, 0:20]     = 1
        mask[:, 900:1050] = 1
        self.features = dict(maxCorners=100, qualityLevel=0.3,
                             minDistance=3, blockSize=7, mask=mask)

    def get_camera_movement(self, frames: list) -> list:
        """Estimate camera movement with stride=3 for speed, then interpolate."""
        STRIDE = 3
        total = len(frames)
        sampled_movement = {0: [0.0, 0.0]}

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_pts  = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for i in range(1, total, STRIDE):
            gray     = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            new_pts, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, **self.lk_params)
            max_d, cx, cy = 0, 0.0, 0.0
            if new_pts is not None and old_pts is not None:
                for n, o in zip(new_pts, old_pts):
                    d = measure_distance(n.ravel(), o.ravel())
                    if d > max_d:
                        max_d = d
                        cx, cy = measure_xy_distance(o.ravel(), n.ravel())
            sampled_movement[i] = [cx, cy] if max_d > self.minimum_distance else [0.0, 0.0]
            if max_d > self.minimum_distance:
                old_pts = cv2.goodFeaturesToTrack(gray, **self.features)
            old_gray = gray.copy()

        # Interpolate back to full frame count
        df = pd.DataFrame(index=range(total), columns=["x", "y"], dtype=float)
        for idx, mv in sampled_movement.items():
            df.loc[idx] = mv
        df = df.interpolate(method="linear").bfill().ffill()
        df["x"] = df["x"].rolling(5, min_periods=1, center=True).mean()
        df["y"] = df["y"].rolling(5, min_periods=1, center=True).mean()
        return df.values.tolist()

    @classmethod
    def from_video_path(cls, video_path: str):
        """从视频第一帧创建实例（流式处理用）。"""
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame: {video_path}")
        return cls(frame)

    def get_camera_movement_streamed(self, video_path: str, total_frames: int,
                                      chunk_size: int = 500) -> list:
        """流式光流估计：跨块保持 old_gray/old_pts 状态，不需要全帧列表。"""
        STRIDE = 3
        sampled_movement = {0: [0.0, 0.0]}
        old_gray = None
        old_pts  = None

        for start_idx, chunk in stream_video_chunks(video_path, chunk_size):
            for local_idx, frame in enumerate(chunk):
                fi = start_idx + local_idx
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if old_gray is None:          # 第一帧初始化
                    old_gray = gray
                    old_pts  = cv2.goodFeaturesToTrack(old_gray, **self.features)
                    continue

                if fi % STRIDE != 0:          # 跳帧：只更新 old_gray，不计算
                    old_gray = gray
                    continue

                new_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray, old_pts, None, **self.lk_params)
                max_d, cx, cy = 0, 0.0, 0.0
                if new_pts is not None and old_pts is not None:
                    for n, o in zip(new_pts, old_pts):
                        d = measure_distance(n.ravel(), o.ravel())
                        if d > max_d:
                            max_d = d
                            cx, cy = measure_xy_distance(o.ravel(), n.ravel())
                sampled_movement[fi] = ([cx, cy] if max_d > self.minimum_distance
                                         else [0.0, 0.0])
                if max_d > self.minimum_distance:
                    old_pts = cv2.goodFeaturesToTrack(gray, **self.features)
                old_gray = gray.copy()

        # 插值回全帧数
        df = pd.DataFrame(index=range(total_frames), columns=["x", "y"], dtype=float)
        for idx, mv in sampled_movement.items():
            if idx < total_frames:
                df.loc[idx] = mv
        df = df.interpolate(method="linear").bfill().ffill()
        df["x"] = df["x"].rolling(5, min_periods=1, center=True).mean()
        df["y"] = df["y"].rolling(5, min_periods=1, center=True).mean()
        return df.values.tolist()

    def add_adjust_positions_to_tracks(self, tracks: dict, cam_movement: list):
        for obj, otracks in tracks.items():
            for fnum, track in enumerate(otracks):
                mv = cam_movement[fnum]
                for info in track.values():
                    pos = info.get("position")
                    if pos:
                        info["position_adjusted"] = (pos[0]-mv[0], pos[1]-mv[1])


# ═══════════════════════════════════════════════════════════════════════
# KeypointDetector
# ═══════════════════════════════════════════════════════════════════════

class KeypointDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    # 光流位移超过此像素值时强制触发关键点检测，防止快速平移期间单应性偏差
    _PAN_TRIGGER_PX = 15.0

    def predict(self, frames: list, cam_movement: list = None) -> list:
        """
        cam_movement: 可选，每帧 [dx, dy] 列表（来自 CameraMovementEstimator）。
        若某帧的镜头位移幅度超过 _PAN_TRIGGER_PX，强制插入该帧做关键点检测，
        打断 KEYPOINT_STRIDE 的冷却期，避免快速平移时单应性矩阵严重偏差。
        """
        sampled  = {}
        base_set = set(range(0, len(frames), KEYPOINT_STRIDE))

        # 检测高速平移帧并强制加入采样
        if cam_movement is not None:
            for fi, mv in enumerate(cam_movement):
                if mv and len(mv) >= 2:
                    if (mv[0]**2 + mv[1]**2) ** 0.5 > self._PAN_TRIGGER_PX:
                        base_set.add(fi)

        indices = sorted(base_set)
        for i in range(0, len(indices), YOLO_BATCH_SIZE):
            batch_idx = indices[i:i+YOLO_BATCH_SIZE]
            results   = self.model.predict([frames[j] for j in batch_idx],
                                           conf=0.1, verbose=False, half=True, imgsz=640)
            for res, fidx in zip(results, batch_idx):
                kps = {}
                if res.keypoints is not None and res.keypoints.xy.shape[1] > 0:
                    xy    = res.keypoints.xy[0].cpu().numpy()
                    confs = (res.keypoints.conf[0].cpu().numpy()
                             if res.keypoints.conf is not None
                             else np.ones(len(xy)))
                    for kid, (x, y) in enumerate(xy):
                        if confs[kid] > 0.5 and (x!=0 or y!=0):
                            kps[kid] = [float(x), float(y)]
                sampled[fidx] = kps

        # 展开到全帧列表（线性插值填充，比最近邻更平滑）
        return _linear_fill_keypoints(sampled, len(frames))


# ═══════════════════════════════════════════════════════════════════════
# ViewTransformer
# ═══════════════════════════════════════════════════════════════════════

    def predict_streamed(self, video_path: str, total_frames: int,
                          chunk_size: int = 500,
                          cam_movement: list = None) -> list:
        """
        流式关键点检测：KEYPOINT_STRIDE 采样，不加载全部帧。
        cam_movement: 可选，每帧 [dx, dy]，高速平移帧强制触发检测（同 predict）。
        """
        sampled = {}

        # 预计算需要强制检测的帧集合
        forced = set()
        if cam_movement is not None:
            for fi, mv in enumerate(cam_movement):
                if mv and len(mv) >= 2:
                    if (mv[0]**2 + mv[1]**2) ** 0.5 > self._PAN_TRIGGER_PX:
                        forced.add(fi)

        for start_idx, chunk in stream_video_chunks(video_path, chunk_size):
            # 落在步长边界 或 强制触发 的帧
            kp_local = [j for j in range(len(chunk))
                         if (start_idx + j) % KEYPOINT_STRIDE == 0
                         or (start_idx + j) in forced]
            if not kp_local:
                continue
            for i in range(0, len(kp_local), YOLO_BATCH_SIZE):
                batch_l  = kp_local[i:i+YOLO_BATCH_SIZE]
                results  = self.model.predict([chunk[j] for j in batch_l],
                                               conf=0.1, verbose=False, half=True, imgsz=640)
                for res, local_idx in zip(results, batch_l):
                    global_idx = start_idx + local_idx
                    kps = {}
                    if res.keypoints is not None and res.keypoints.xy.shape[1] > 0:
                        xy    = res.keypoints.xy[0].cpu().numpy()
                        confs = (res.keypoints.conf[0].cpu().numpy()
                                 if res.keypoints.conf is not None
                                 else np.ones(len(xy)))
                        for kid, (x, y) in enumerate(xy):
                            if confs[kid] > 0.5 and (x != 0 or y != 0):
                                kps[kid] = [float(x), float(y)]
                    sampled[global_idx] = kps

        # 展开到全帧列表（线性插值填充，比最近邻更平滑）
        return _linear_fill_keypoints(sampled, total_frames)


class ViewTransformer:
    # Soccana_Keypoint (29 pts) → physical pitch coordinates
    # Units match SoccerPitchConfiguration scale (12000 × 7000).
    # KP 10 / 26: penalty spots used as D-arc proxies (closest available point).
    # KP 15: field center (6000, 3500) — not in sports library, added manually.
    SOCCANA_PITCH_COORDS = {
        0:  (0,     0),      # sideline_top_left
        1:  (0,     1945),   # big_rect_left_top_pt1
        2:  (2015,  1945),   # big_rect_left_top_pt2
        3:  (0,     5055),   # big_rect_left_bottom_pt1
        4:  (2015,  5055),   # big_rect_left_bottom_pt2
        5:  (0,     2584),   # small_rect_left_top_pt1
        6:  (550,   2584),   # small_rect_left_top_pt2
        7:  (0,     4416),   # small_rect_left_bottom_pt1
        8:  (550,   4416),   # small_rect_left_bottom_pt2
        9:  (0,     7000),   # sideline_bottom_left
        10: (1100,  3500),   # left_semicircle_right  (penalty spot proxy)
        11: (6000,  0),      # center_line_top
        12: (6000,  7000),   # center_line_bottom
        13: (6000,  2085),   # center_circle_top
        14: (6000,  4915),   # center_circle_bottom
        15: (6000,  3500),   # field_center
        16: (12000, 0),      # sideline_top_right
        17: (12000, 1945),   # big_rect_right_top_pt1
        18: (9985,  1945),   # big_rect_right_top_pt2
        19: (12000, 5055),   # big_rect_right_bottom_pt1
        20: (9985,  5055),   # big_rect_right_bottom_pt2
        21: (12000, 2584),   # small_rect_right_top_pt1
        22: (11450, 2584),   # small_rect_right_top_pt2
        23: (12000, 4416),   # small_rect_right_bottom_pt1
        24: (11450, 4416),   # small_rect_right_bottom_pt2
        25: (12000, 7000),   # sideline_bottom_right
        26: (10900, 3500),   # right_semicircle_left  (penalty spot proxy)
        27: (4085,  3500),   # center_circle_left
        28: (7915,  3500),   # center_circle_right
    }

    def __init__(self):
        if not HAS_SPORTS:
            self.config = None; return
        self.config = SoccerPitchConfiguration()
        mc = max(abs(v[0]) for v in self.config.vertices) + \
             max(abs(v[1]) for v in self.config.vertices)
        if   mc > 500: self.scale_factor, self.minimap_scale = 0.01, 1.0
        elif mc > 50:  self.scale_factor, self.minimap_scale = 0.1,  10.0
        else:          self.scale_factor, self.minimap_scale = 1.0,  100.0
        self._last_transformer = None  # fallback when keypoints < 4


    def add_transformed_position_to_tracks(self, tracks: dict, kps_list: list):
        if not HAS_SPORTS: return

        for fnum, kps in enumerate(kps_list):
            src, dst = [], []
            for kid, pos in kps.items():
                target = self.SOCCANA_PITCH_COORDS.get(kid)
                if target is not None:
                    src.append(pos); dst.append(target)

            if len(src) >= 4:
                transformer = SportsViewTransformer(source=np.array(src), target=np.array(dst))
                self._last_transformer = transformer
            elif self._last_transformer is not None:
                # 用上一帧有效的 homography 做 fallback，避免小地图断续
                transformer = self._last_transformer
            else:
                continue

            for obj, otracks in tracks.items():
                if fnum >= len(otracks): continue
                for info in otracks[fnum].values():
                    pos = info.get("position_adjusted") or info.get("position")
                    if not pos: continue
                    t = transformer.transform_points(np.array([pos]))
                    if t is not None and len(t) > 0:
                        tx = t[0][0] * self.scale_factor
                        ty = t[0][1] * self.scale_factor
                        tx_c, ty_c = clamp_pitch_position(tx, ty)
                        info["position_transformed"] = [tx_c, ty_c]
                        info["position_minimap"]     = [tx_c * (self.minimap_scale / self.scale_factor),
                                                          ty_c * (self.minimap_scale / self.scale_factor)]


    def interpolate_2d_positions(self, tracks: dict):
        """平滑 minimap 坐标（自适应窗口：静止大窗口，移动小窗口）"""
        SPEED_THRESHOLD = 0.3   # m/frame — below this → stationary smoothing
        WINDOW_SLOW     = 15    # large window for stationary players
        WINDOW_FAST     = 5     # small window for fast-moving players

        for obj, otracks in tracks.items():
            if obj in ("ball", "referees"): continue
            all_ids = set()
            for fd in otracks: all_ids.update(fd.keys())

            for tid in all_ids:
                rows = []
                for fd in otracks:
                    info = fd.get(tid)
                    pt = info.get("position_transformed", [np.nan, np.nan]) if info else [np.nan, np.nan]
                    mp = info.get("position_minimap",     [np.nan, np.nan]) if info else [np.nan, np.nan]
                    rows.append({"x": pt[0], "y": pt[1], "mx": mp[0], "my": mp[1]})

                df = pd.DataFrame(rows).interpolate("linear").bfill().ffill()

                # Compute per-frame speed (m/frame) from position_transformed deltas
                dx = df["x"].diff().fillna(0)
                dy = df["y"].diff().fillna(0)
                speed = (dx**2 + dy**2).pow(0.5)  # Euclidean distance per frame

                # Choose window per frame: slow → 15, fast → 5
                windows = np.where(speed < SPEED_THRESHOLD, WINDOW_SLOW, WINDOW_FAST)

                # Apply rolling smooth with adaptive window
                # Strategy: split into segments by window size and smooth each
                mx_smooth = df["mx"].copy()
                my_smooth = df["my"].copy()
                x_smooth  = df["x"].copy()
                y_smooth  = df["y"].copy()

                for win in [WINDOW_SLOW, WINDOW_FAST]:
                    mask = (windows == win)
                    if not mask.any(): continue
                    # Apply uniform rolling window to the whole series, then keep only masked frames
                    r_mx = df["mx"].rolling(win, min_periods=1, center=True).mean()
                    r_my = df["my"].rolling(win, min_periods=1, center=True).mean()
                    r_x  = df["x"].rolling(min(win, SPEED_SMOOTH_WINDOW), min_periods=1, center=True).mean()
                    r_y  = df["y"].rolling(min(win, SPEED_SMOOTH_WINDOW), min_periods=1, center=True).mean()
                    mx_smooth = mx_smooth.where(~mask, r_mx)
                    my_smooth = my_smooth.where(~mask, r_my)
                    x_smooth  = x_smooth.where(~mask, r_x)
                    y_smooth  = y_smooth.where(~mask, r_y)

                df["mx"] = mx_smooth
                df["my"] = my_smooth
                df["x"]  = x_smooth
                df["y"]  = y_smooth

                for i, row in df.iterrows():
                    if tid in otracks[i] and not np.isnan(row["x"]):
                        otracks[i][tid]["position_transformed"] = [row["x"],  row["y"]]
                        otracks[i][tid]["position_minimap"]     = [row["mx"], row["my"]]


# ═══════════════════════════════════════════════════════════════════════
# AccurateSpeedEstimator
# ═══════════════════════════════════════════════════════════════════════

class AccurateSpeedEstimator:
    def __init__(self):
        self.fps          = 24
        self.frame_window = 5
        self.max_speed    = 38.0
        self._history     = {}

    def add_speed_and_distance_to_tracks(self, tracks: dict):
        total_dist = {}
        for obj, otracks in tracks.items():
            if obj in ("ball","referees"): continue
            for fi in range(len(otracks)):
                prev = max(0, fi - self.frame_window)
                for tid, info in otracks[fi].items():
                    if tid not in otracks[prev]: continue
                    cp = info.get("position_transformed")
                    pp = otracks[prev][tid].get("position_transformed")
                    if not cp or not pp: continue
                    dist    = measure_distance(cp, pp)
                    elapsed = (fi - prev) / self.fps
                    if elapsed == 0: continue
                    raw = (dist / elapsed) * 3.6
                    if raw <= self.max_speed:
                        self._history.setdefault(tid, []).append(raw)
                        if len(self._history[tid]) > 7: self._history[tid].pop(0)
                    smoothed = min(
                        float(np.median(self._history[tid])) if tid in self._history else raw,
                        self.max_speed
                    )
                    total_dist.setdefault(obj, {}).setdefault(tid, 0)
                    total_dist[obj][tid] += dist
                    info["speed"]    = smoothed
                    info["distance"] = total_dist[obj][tid]


# ═══════════════════════════════════════════════════════════════════════
# TeamAssigner
# ═══════════════════════════════════════════════════════════════════════

class TeamAssigner:
    # 对全场球员颜色聚类时使用 k=5，以分离门将/裁判的颜色簇
    _TEAM_CLUSTERS = 5

    def __init__(self):
        self.team_colors      = {}
        self.player_team_dict = {}
        self.kmeans           = None
        self._cluster_to_team = {}  # cluster_id → team_id (1 or 2)

    def _get_player_color(self, frame, bbox) -> np.ndarray:
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if img.shape[0]==0 or img.shape[1]==0: return np.array([0,0,0])
        top = img[:img.shape[0]//2, :]
        km  = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(top.reshape(-1,3))
        lbl = km.labels_.reshape(top.shape[0], top.shape[1])
        corners = [lbl[0,0], lbl[0,-1], lbl[-1,0], lbl[-1,-1]]
        return km.cluster_centers_[1 - max(set(corners), key=corners.count)]

    def _fit_team_kmeans(self, all_colors: list):
        """
        KMeans(k=5) 对全场球员颜色聚类，选出人数最多的两个簇作为两支球队，
        其余小簇（门将/裁判）不污染球队颜色中心。
        """
        if len(all_colors) < 2:
            return
        colors_arr = np.array(all_colors)
        n_clusters = min(self._TEAM_CLUSTERS, len(colors_arr))
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10,
                    random_state=42).fit(colors_arr)

        # 按簇大小降序排列，取前两个最大簇作为球队
        cluster_sizes = np.bincount(km.labels_, minlength=n_clusters)
        sorted_ids    = np.argsort(cluster_sizes)[::-1]
        team_ids      = sorted_ids[:2]

        self._cluster_to_team = {int(cid): (i + 1) for i, cid in enumerate(team_ids)}
        self.kmeans           = km
        self.team_colors[1]   = km.cluster_centers_[team_ids[0]]
        self.team_colors[2]   = km.cluster_centers_[team_ids[1]]

    def assign_team_color(self, frame, player_detections: dict):
        colors = [self._get_player_color(frame, d["bbox"])
                  for d in player_detections.values() if d]
        if not colors: return
        self._fit_team_kmeans(colors)

    def assign_team_color_multi(self, frames: list, tracks_players: list, n_samples: int = 8):
        """
        从视频中均匀采样 n_samples 帧，聚合所有球员颜色后拟合 KMeans。
        比单帧初始化鲁棒得多，解决白/黄等相近颜色无法区分的问题。
        """
        total = len(frames)
        if total == 0:
            return
        # 均匀采样（跳过开头/结尾5%，避免黑帧）
        start = max(0, int(total * 0.05))
        end   = min(total - 1, int(total * 0.95))
        sample_idx = np.linspace(start, end, n_samples, dtype=int)

        all_colors = []
        for idx in sample_idx:
            if idx >= len(tracks_players):
                continue
            for pid, info in tracks_players[idx].items():
                if not info or 'bbox' not in info:
                    continue
                try:
                    c = self._get_player_color(frames[idx], info['bbox'])
                    if c is not None and not np.all(c == 0):
                        all_colors.append(c)
                except Exception:
                    pass

        self._fit_team_kmeans(all_colors)

    def assign_team_color_from_video(self, video_path: str, tracks_players: list,
                                      n_samples: int = 8):
        """流式版 assign_team_color_multi：按需 seek 到采样帧，不需要完整 frames 列表。"""
        total = len(tracks_players)
        if total == 0:
            return
        start = max(0, int(total * 0.05))
        end   = min(total - 1, int(total * 0.95))
        sample_idx = np.linspace(start, end, n_samples, dtype=int).tolist()
        frame_dict = read_frames_at_indices(video_path, sample_idx)

        all_colors = []
        for idx in sample_idx:
            frame = frame_dict.get(idx)
            if frame is None or idx >= len(tracks_players):
                continue
            for pid, info in tracks_players[idx].items():
                if not info or 'bbox' not in info:
                    continue
                try:
                    c = self._get_player_color(frame, info['bbox'])
                    if c is not None and not np.all(c == 0):
                        all_colors.append(c)
                except Exception:
                    pass

        self._fit_team_kmeans(all_colors)

    def get_player_team(self, frame, bbox, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return 1  # fallback：kmeans 未初始化时默认队伍1
        color      = self._get_player_color(frame, bbox)
        cluster_id = int(self.kmeans.predict(color.reshape(1, -1))[0])
        tid        = self._cluster_to_team.get(cluster_id)
        if tid is None:
            # 小簇（门将/裁判）：分配到颜色最近的球队
            d1  = np.linalg.norm(color - self.team_colors.get(1, np.zeros(3)))
            d2  = np.linalg.norm(color - self.team_colors.get(2, np.zeros(3)))
            tid = 1 if d1 <= d2 else 2
        self.player_team_dict[player_id] = tid
        return tid


# ═══════════════════════════════════════════════════════════════════════
# SmartBallPossessionDetector
# ═══════════════════════════════════════════════════════════════════════

class SmartBallPossessionDetector:
    def __init__(self, fps: int = 24, video_w: int = 1920, video_h: int = 1080):
        self.fps                  = fps
        # 按视频宽度归一化距离阈值（基准：1920px）
        self._res_scale           = max(video_w, video_h) / 1920.0
        self.max_control_distance = int(70  * self._res_scale)
        self._possession_history  = []
        self._speed_history       = []
        self._current             = None
        self._frames_stable       = 0
        self.ball_state           = "controlled"

    def detect_possession(self, frame_num: int, players: dict,
                          ball_bbox: list, ball_history: list,
                          ball_transformed_pos: list = None) -> int:
        if len(ball_bbox) != 4 or not players: return -1
        ball_pos  = ((ball_bbox[0]+ball_bbox[2])/2, (ball_bbox[1]+ball_bbox[3])/2)
        ball_spd  = self._calc_ball_speed(ball_history)
        thr_fly   = 30 * self._res_scale
        thr_cont  = 15 * self._res_scale
        if ball_spd > thr_fly:
            self.ball_state = "flying"
            pid = self._detect_flying(ball_pos, ball_history, players)
        elif ball_spd > thr_cont:
            self.ball_state = "contested"
            pid = self._detect_contested(ball_pos, players)
        else:
            # Slow ball — check if any player is actually close enough
            pid = self._detect_controlled(ball_pos, ball_history, players,
                                          ball_transformed_pos)
            if pid == -1:
                self.ball_state = "loose_ball"  # ball on ground, no one in range
            else:
                self.ball_state = "controlled"

        return self._smooth(pid)

    def get_confidence(self) -> float:
        if self._frames_stable < 3:    return 0.3
        if self.ball_state=="flying":  return 0.5
        if self.ball_state=="controlled": return 0.9
        return 0.6

    def _calc_ball_speed(self, history):
        if len(history)<2 or not history[-1] or not history[-2]: return 0.0
        s = measure_distance(history[-1], history[-2])
        self._speed_history.append(s)
        if len(self._speed_history)>5: self._speed_history.pop(0)
        return float(np.mean(self._speed_history))

    # 真实距离阈值：球距脚底 < 2.5m 才算控球（宽松，避免漏判）
    _CONTROL_DIST_M = 2.5

    def _detect_controlled(self, ball_pos, history, players,
                           ball_transformed_pos=None):
        """
        优先使用单应性变换后的真实米数距离（< 1.2m）判断控球，
        不可用时退回到像素距离（bbox 高度 1.5 倍）。
        """
        best, best_s = -1, 0
        use_real = (ball_transformed_pos is not None
                    and len(ball_transformed_pos) == 2)

        for pid, info in players.items():
            if not info or "bbox" not in info: continue
            b    = info["bbox"]
            foot = ((b[0]+b[2])/2, b[3])

            if use_real:
                pt = info.get("position_transformed")
                if pt and len(pt) == 2:
                    # 真实米数距离
                    dx = ball_transformed_pos[0] - pt[0]
                    dy = ball_transformed_pos[1] - pt[1]
                    d_m = (dx*dx + dy*dy) ** 0.5
                    if d_m > self._CONTROL_DIST_M: continue
                    s = 100 / (d_m + 0.01)
                    if s > best_s: best_s = s; best = pid
                    continue  # 跳过像素回退

            # 像素距离回退：阈值 = bbox 高度的 3 倍（宽松，不漏判）
            bbox_h   = max(b[3] - b[1], 1)
            max_dist = bbox_h * 3.0
            d = measure_distance(ball_pos, foot)
            if d > max_dist: continue
            s = 100 / (d + 1)
            if s > best_s: best_s = s; best = pid
        return best

    def _detect_flying(self, ball_pos, history, players):
        if len(history)<3 or not history[-1] or not history[-3]: return -1
        vel = np.array([history[-1][0]-history[-3][0], history[-1][1]-history[-3][1]])
        max_d = 300 * self._res_scale
        min_v = 5   * self._res_scale
        best, best_s = -1, 0
        for pid, info in players.items():
            if not info or "bbox" not in info: continue
            b    = info["bbox"]
            foot = np.array([(b[0]+b[2])/2, b[3]])
            tp   = foot - np.array(ball_pos)
            d    = float(np.linalg.norm(tp))
            if d > max_d or np.linalg.norm(vel) < min_v: continue
            cos  = float(np.dot(vel,tp)/(np.linalg.norm(vel)*d+1e-6))
            if cos>0.5:
                s = cos*1000/(d+1)
                if s>best_s: best_s=s; best=pid
        return best

    def _detect_contested(self, ball_pos, players):
        best, best_d = -1, int(80 * self._res_scale)
        for pid, info in players.items():
            if not info or "bbox" not in info: continue
            b = info["bbox"]
            d = measure_distance(ball_pos, ((b[0]+b[2])/2, b[3]))
            if d < best_d: best_d=d; best=pid
        return best

    def _smooth(self, new_pid: int) -> int:
        self._possession_history.append(new_pid)
        if len(self._possession_history)>10: self._possession_history.pop(0)
        if new_pid == self._current:
            self._frames_stable += 1
            return self._current
        if (len(self._possession_history)>=3
                and all(p==new_pid for p in self._possession_history[-3:])
                and new_pid != -1):
            self._current        = new_pid
            self._frames_stable  = 0
        return self._current if self._current is not None else new_pid


# ═══════════════════════════════════════════════════════════════════════
# render_minimap_frame  ← 小地图回放专用渲染函数
# ═══════════════════════════════════════════════════════════════════════

def render_minimap_frame(frame_idx: int, tracks: dict,
                         tracked_bboxes: dict, team_control: np.ndarray,
                         config, hex_t1: str, hex_t2: str,
                         ball_trail: list = None) -> np.ndarray:
    """
    渲染单帧小地图图像（不含原始视频画面）。
    输出尺寸约 700×400（由 draw_pitch 决定），可直接写入 MP4。
    """
    if not HAS_SPORTS:
        return np.zeros((400, 700, 3), dtype=np.uint8)

    pitch = draw_pitch(config=config)

    # ── 找被追踪目标 ──────────────────────────────────────────────────
    tracked_pos  = None
    tracked_team = None
    if frame_idx in tracked_bboxes:
        sx, sy, sw, sh = tracked_bboxes[frame_idx]
        center = (sx+sw/2, sy+sh/2)
        min_d  = 150
        for pid, info in tracks["players"][frame_idx].items():
            if not info or "bbox" not in info: continue
            b  = info["bbox"]
            cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            d  = measure_distance(center, (cx, cy))
            if d < min_d:
                min_d = d
                tp = info.get("position_minimap")
                if tp and len(tp)==2 and not any(np.isnan(p) for p in tp):
                    tracked_pos  = tp
                    tracked_team = info.get("team")

    # ── 分组队伍位置 ──────────────────────────────────────────────────
    t1_pos, t2_pos = [], []
    for pid, info in tracks["players"][frame_idx].items():
        if not info: continue
        pos = info.get("position_minimap")
        if not pos or len(pos)!=2 or any(np.isnan(p) for p in pos): continue
        if info.get("team")==1: t1_pos.append(pos)
        else:                   t2_pos.append(pos)

    # ── 绘制球员点 ────────────────────────────────────────────────────
    if t1_pos:
        pitch = draw_points_on_pitch(config=config, xy=np.array(t1_pos),
            face_color=sv.Color.from_hex(hex_t1), edge_color=sv.Color.BLACK,
            radius=12, pitch=pitch)
    if t2_pos:
        pitch = draw_points_on_pitch(config=config, xy=np.array(t2_pos),
            face_color=sv.Color.from_hex(hex_t2), edge_color=sv.Color.BLACK,
            radius=12, pitch=pitch)

    # ── 高亮被追踪球员 ────────────────────────────────────────────────
    if tracked_pos:
        pitch = draw_points_on_pitch(config=config, xy=np.array([tracked_pos]),
            face_color=sv.Color.from_hex("#FFD700"),
            edge_color=sv.Color.from_hex("#FFD700"), radius=22, pitch=pitch)
        inner = hex_t1 if tracked_team==1 else hex_t2
        pitch = draw_points_on_pitch(config=config, xy=np.array([tracked_pos]),
            face_color=sv.Color.from_hex(inner), edge_color=sv.Color.BLACK,
            radius=14, pitch=pitch)

    # ── 球轨迹拖尾（最近30帧）────────────────────────────────────────
    # 用 draw_points_on_pitch 绘制（position_minimap 是 config 坐标系，非像素坐标）
    if ball_trail and len(ball_trail) > 1:
        n = len(ball_trail)
        for i, pos in enumerate(ball_trail):
            alpha = (i + 1) / n          # 0=oldest → 1=newest
            intensity = int(alpha * 200) + 55   # 55→255 绿色渐变
            hex_trail = f"#{0:02x}{intensity:02x}{0:02x}"
            radius = max(3, int(alpha * 7))
            try:
                pitch = draw_points_on_pitch(
                    config=config,
                    xy=np.array([pos]),
                    face_color=sv.Color.from_hex(hex_trail),
                    edge_color=sv.Color.from_hex(hex_trail),
                    radius=radius,
                    pitch=pitch
                )
            except Exception:
                pass  # 坐标越界时跳过

    # ── 足球 ─────────────────────────────────────────────────────────
    ball_pos = tracks["ball"][frame_idx].get(1, {}).get("position_minimap")
    if ball_pos and len(ball_pos)==2 and not any(np.isnan(p) for p in ball_pos):
        pitch = draw_points_on_pitch(config=config, xy=np.array([ball_pos]),
            face_color=sv.Color.from_hex("#FFFFFF"),
            edge_color=sv.Color.BLACK, radius=8, pitch=pitch)

    # ── 控球率角标 ────────────────────────────────────────────────────
    ctrl = team_control[:frame_idx+1]
    t1c  = int(np.sum(ctrl==1)); t2c = int(np.sum(ctrl==2))
    tot  = t1c+t2c or 1
    h, w = pitch.shape[:2]
    overlay = pitch.copy()
    cv2.rectangle(overlay, (w-220, 5), (w-5, 55), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, pitch, 0.4, 0, pitch)
    cv2.putText(pitch, f"T1:{t1c/tot*100:.0f}%  T2:{t2c/tot*100:.0f}%",
                (w-215, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # ── 帧号时间戳 ────────────────────────────────────────────────────
    secs = frame_idx / 24
    cv2.putText(pitch, f"{int(secs//60):02d}:{secs%60:04.1f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return pitch
