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
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None

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
KEYPOINT_STRIDE       = 60   # 每60帧检测一次关键点（提升小地图精度）
MINIMAP_SMOOTH_WINDOW = 25
SPEED_SMOOTH_WINDOW   = 7


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
            results      = self.model.predict(batch_frames, conf=0.1, verbose=False, half=True, imgsz=416)
            for res, fidx in zip(results, batch_idx):
                det_dict[fidx] = res

        for fidx, det in det_dict.items():
            ds = sv.Detections.from_ultralytics(det)
            for k, cid in enumerate(ds.class_id):
                if "goalkeeper" in self.class_names[cid].lower():
                    ds.class_id[k] = self.player_id

            d_tracks = self.tracker.update_with_detections(ds)
            tracks["players"][fidx]  = {}
            tracks["referees"][fidx] = {}
            tracks["ball"][fidx]     = {}

            for d in d_tracks:
                bbox, cid, tid = d[0].tolist(), d[3], d[4]
                if   cid == self.player_id:   tracks["players"][fidx][tid]  = {"bbox": bbox}
                elif cid == self.referee_id:  tracks["referees"][fidx][tid] = {"bbox": bbox}

            for d in ds:
                if d[3] == self.ball_id:
                    tracks["ball"][fidx][1] = {"bbox": d[0].tolist()}

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
        raw = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df  = pd.DataFrame(raw, columns=["x1","y1","x2","y2"])
        df  = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]


    def draw_ellipse(self, frame, bbox, color, track_id, is_tracked=False):
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        width = int(bbox[2] - bbox[0])
        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2)
        cv2.rectangle(frame, (int(x_center - width/2 - 2), int(y2 - 8)),
                      (int(x_center + width/2 + 2), int(y2 + 8)), color, -1)
        if track_id is not None:
            cv2.putText(frame, str(track_id), (int(x_center - width/2), int(y2 + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
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

    def predict(self, frames: list) -> list:
        sampled = {}
        indices = list(range(0, len(frames), KEYPOINT_STRIDE))
        for i in range(0, len(indices), YOLO_BATCH_SIZE):
            batch_idx = indices[i:i+YOLO_BATCH_SIZE]
            results   = self.model.predict([frames[j] for j in batch_idx],
                                           conf=0.1, verbose=False, half=True, imgsz=416)
            for res, fidx in zip(results, batch_idx):
                kps = {}
                if res.keypoints is not None and res.keypoints.xy.shape[1] > 0:
                    xy    = res.keypoints.xy[0].cpu().numpy()
                    confs = (res.keypoints.conf[0].cpu().numpy()
                             if res.keypoints.conf is not None
                             else np.ones(len(xy)))
                    for kid, (x, y) in enumerate(xy):
                        if confs[kid] > 0.3 and (x!=0 or y!=0):
                            kps[kid] = [float(x), float(y)]
                sampled[fidx] = kps

        result = []
        for i in range(len(frames)):
            if i in sampled:
                result.append(sampled[i])
            else:
                nearest = min(sampled.keys(), key=lambda k: abs(k-i), default=0)
                result.append(sampled.get(nearest, {}))
        return result


# ═══════════════════════════════════════════════════════════════════════
# ViewTransformer
# ═══════════════════════════════════════════════════════════════════════

class ViewTransformer:
    def __init__(self):
        if not HAS_SPORTS:
            self.config = None; return
        self.config = SoccerPitchConfiguration()
        mc = max(abs(v[0]) for v in self.config.vertices) + \
             max(abs(v[1]) for v in self.config.vertices)
        if   mc > 500: self.scale_factor, self.minimap_scale = 0.01, 1.0
        elif mc > 50:  self.scale_factor, self.minimap_scale = 0.1,  10.0
        else:          self.scale_factor, self.minimap_scale = 1.0,  100.0


    def add_transformed_position_to_tracks(self, tracks: dict, kps_list: list):
        if not HAS_SPORTS: return
        # === Colab Hardcoded Fallback Config ===
        court_width, court_length = 68.0, 23.0
        pv = np.array([[110, 1035], [2650, 1035], [2250, 275], [800, 275]], dtype=np.float32)
        tv = np.array([[0, court_width], [court_length, court_width], [court_length, 0], [0, 0]], dtype=np.float32)
        fb_trans = cv2.getPerspectiveTransform(pv, tv)

        def fb_transform(point):
            if cv2.pointPolygonTest(pv, (int(point[0]), int(point[1])), False) >= 0:
                res = cv2.perspectiveTransform(np.array(point, dtype=np.float32).reshape(-1, 1, 2), fb_trans)
                return res.reshape(-1, 2)
            return None

        for fnum, kps in enumerate(kps_list):
            src, dst = [], []
            for kid, pos in kps.items():
                if kid < len(self.config.vertices):
                    src.append(pos); dst.append(self.config.vertices[kid])
            
            transformer = SportsViewTransformer(source=np.array(src), target=np.array(dst)) if len(src) >= 4 else None
            
            for obj, otracks in tracks.items():
                if fnum >= len(otracks): continue
                for info in otracks[fnum].values():
                    pos = info.get("position_adjusted") or info.get("position")
                    if not pos: continue
                    t = transformer.transform_points(np.array([pos])) if transformer else None
                    if t is not None and len(t) > 0:
                        info["position_transformed"] = [t[0][0]*self.scale_factor, t[0][1]*self.scale_factor]
                        info["position_minimap"]     = [t[0][0]*self.minimap_scale, t[0][1]*self.minimap_scale]
                    else:
                        fb = fb_transform(pos)
                        if fb is not None:
                            info["position_transformed"] = [fb[0][0]*self.scale_factor*100, fb[0][1]*self.scale_factor*100]
                            info["position_minimap"] = [fb[0][0]*self.minimap_scale*100, fb[0][1]*self.minimap_scale*100]


    def interpolate_2d_positions(self, tracks: dict):
        """平滑 minimap 坐标（大窗口高斯平滑）"""
        for obj, otracks in tracks.items():
            if obj in ("ball","referees"): continue
            all_ids = set()
            for fd in otracks: all_ids.update(fd.keys())
            for tid in all_ids:
                rows = []
                for fd in otracks:
                    info = fd.get(tid)
                    pt = info.get("position_transformed",[np.nan,np.nan]) if info else [np.nan,np.nan]
                    mp = info.get("position_minimap",   [np.nan,np.nan]) if info else [np.nan,np.nan]
                    rows.append({"x":pt[0],"y":pt[1],"mx":mp[0],"my":mp[1]})

                df = pd.DataFrame(rows).interpolate("linear").bfill().ffill()
                if len(df) > MINIMAP_SMOOTH_WINDOW:
                    df["mx"] = df["mx"].rolling(MINIMAP_SMOOTH_WINDOW, min_periods=1,
                                                center=True, win_type="gaussian").mean(std=3)
                    df["my"] = df["my"].rolling(MINIMAP_SMOOTH_WINDOW, min_periods=1,
                                                center=True, win_type="gaussian").mean(std=3)
                    df["x"]  = df["x"].rolling(SPEED_SMOOTH_WINDOW, min_periods=1, center=True).mean()
                    df["y"]  = df["y"].rolling(SPEED_SMOOTH_WINDOW, min_periods=1, center=True).mean()

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
    def __init__(self):
        self.team_colors      = {}
        self.player_team_dict = {}
        self.kmeans           = None

    def _get_player_color(self, frame, bbox) -> np.ndarray:
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if img.shape[0]==0 or img.shape[1]==0: return np.array([0,0,0])
        top = img[:img.shape[0]//2, :]
        km  = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(top.reshape(-1,3))
        lbl = km.labels_.reshape(top.shape[0], top.shape[1])
        corners = [lbl[0,0], lbl[0,-1], lbl[-1,0], lbl[-1,-1]]
        return km.cluster_centers_[1 - max(set(corners), key=corners.count)]

    def assign_team_color(self, frame, player_detections: dict):
        colors = [self._get_player_color(frame, d["bbox"])
                  for d in player_detections.values() if d]
        if not colors: return
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(colors)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

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

        if len(all_colors) < 2:
            return
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42).fit(np.array(all_colors))
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return 1  # fallback：kmeans 未初始化时默认队伍1
        color = self._get_player_color(frame, bbox)
        tid   = self.kmeans.predict(color.reshape(1,-1))[0] + 1
        self.player_team_dict[player_id] = tid
        return tid


# ═══════════════════════════════════════════════════════════════════════
# SmartBallPossessionDetector
# ═══════════════════════════════════════════════════════════════════════

class SmartBallPossessionDetector:
    def __init__(self, fps: int = 24):
        self.fps                  = fps
        self.max_control_distance = 70
        self._possession_history  = []
        self._speed_history       = []
        self._current             = None
        self._frames_stable       = 0
        self.ball_state           = "controlled"

    def detect_possession(self, frame_num: int, players: dict,
                          ball_bbox: list, ball_history: list) -> int:
        if len(ball_bbox) != 4 or not players: return -1
        ball_pos  = ((ball_bbox[0]+ball_bbox[2])/2, (ball_bbox[1]+ball_bbox[3])/2)
        ball_spd  = self._calc_ball_speed(ball_history)
        self.ball_state = ("flying" if ball_spd>30 else
                           "contested" if ball_spd>15 else "controlled")
        pid = (self._detect_flying(ball_pos, ball_history, players)   if self.ball_state=="flying"   else
               self._detect_contested(ball_pos, players)               if self.ball_state=="contested" else
               self._detect_controlled(ball_pos, ball_history, players))
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

    def _detect_controlled(self, ball_pos, history, players):
        best, best_s = -1, 0
        for pid, info in players.items():
            if not info or "bbox" not in info: continue
            b    = info["bbox"]
            foot = ((b[0]+b[2])/2, b[3])
            d    = measure_distance(ball_pos, foot)
            if d > self.max_control_distance: continue
            s = 100/(d+1)
            if s > best_s: best_s=s; best=pid
        return best

    def _detect_flying(self, ball_pos, history, players):
        if len(history)<3 or not history[-1] or not history[-3]: return -1
        vel = np.array([history[-1][0]-history[-3][0], history[-1][1]-history[-3][1]])
        best, best_s = -1, 0
        for pid, info in players.items():
            if not info or "bbox" not in info: continue
            b    = info["bbox"]
            foot = np.array([(b[0]+b[2])/2, b[3]])
            tp   = foot - np.array(ball_pos)
            d    = float(np.linalg.norm(tp))
            if d>300 or np.linalg.norm(vel)<5: continue
            cos  = float(np.dot(vel,tp)/(np.linalg.norm(vel)*d+1e-6))
            if cos>0.5:
                s = cos*1000/(d+1)
                if s>best_s: best_s=s; best=pid
        return best

    def _detect_contested(self, ball_pos, players):
        best, best_d = -1, 40
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
                         config, hex_t1: str, hex_t2: str) -> np.ndarray:
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
