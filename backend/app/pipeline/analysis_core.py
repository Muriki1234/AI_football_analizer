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
YOLO_DETECTION_STRIDE = 3    # 每3帧检测一次（原2）
YOLO_BATCH_SIZE       = 60   # 单批处理帧数（60 = 更好GPU利用率）
KEYPOINT_STRIDE       = 20   # 每20帧检测一次关键点
MINIMAP_SMOOTH_WINDOW = 25
SPEED_SMOOTH_WINDOW   = 7
PLAYER_CONF           = 0.59  # 球员/球检测置信度（同时适用于球）


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
                          x_max: float = 120.0, y_max: float = 70.0):
    """Clamp a transformed pitch coordinate to valid field bounds.
    Aligned with _PITCH_LEN/WID used by _mm_p2px and make_pitch_background (FIFA: 105×68 m)."""
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


def _check_memory_and_gc(threshold_pct: float = 85.0) -> float:
    """检查内存使用率（%），超阈值时强制 GC。返回当前使用率（无 psutil 返回 -1）。"""
    try:
        import psutil, gc
        mem = psutil.virtual_memory()
        used_pct = mem.percent
        used_mb  = mem.used  // 1024 // 1024
        total_mb = mem.total // 1024 // 1024
        print(f"[MEM] RAM {used_pct:.1f}%  {used_mb}MB / {total_mb}MB")
        if used_pct > threshold_pct:
            print(f"[MEM] ⚠ usage > {threshold_pct:.0f}% — forcing gc.collect()")
            gc.collect()
        return used_pct
    except ImportError:
        return -1.0


# ═══════════════════════════════════════════════════════════════════════
# SceneChangeDetector — 基于球员数统计的比赛分段
# ═══════════════════════════════════════════════════════════════════════

class SceneChangeDetector:
    """
    检测比赛中的非比赛段（中场休息、回放、广告等）。

    核心思路：正常比赛画面每帧应有 10+ 球员；中场/回放时球员数会骤降。
    连续 N 秒（window）球员数 < threshold → 标记为 non-match 段。

    用法：
        det = SceneChangeDetector(fps=24)
        segments = det.detect_segments(tracks, total_frames)
        # → [{"type": "first_half", "start_frame": 0, "end_frame": 67200, ...},
        #    {"type": "halftime",   "start_frame": 67200, "end_frame": 89000, ...},
        #    {"type": "second_half","start_frame": 89000, "end_frame": 156000, ...}]
    """

    def __init__(self, fps: float = 24.0,
                 min_players_match: int = 5,
                 min_non_match_seconds: float = 30.0,
                 smooth_window_seconds: float = 3.0):
        """
        Args:
            fps: 视频帧率
            min_players_match: 单帧球员数 ≥ 此值视为"比赛中"
            min_non_match_seconds: 连续多少秒非比赛才判为中场（避免短暂遮挡）
            smooth_window_seconds: 球员数平滑窗口（避免单帧抖动）
        """
        self.fps = max(1.0, float(fps))
        self.min_players_match = min_players_match
        self.min_non_match_seconds = min_non_match_seconds
        self.smooth_window = max(1, int(smooth_window_seconds * self.fps))

    def detect_segments(self, tracks: dict, total_frames: int) -> list:
        """返回比赛分段列表。保证覆盖 [0, total_frames) 且无重叠。"""
        if total_frames <= 0:
            return []

        # ── 1. 每帧球员数 ───────────────────────────────────────────
        counts = np.zeros(total_frames, dtype=np.int32)
        players = tracks.get("players", [])
        for fi in range(min(total_frames, len(players))):
            pd = players[fi]
            if pd:
                counts[fi] = len(pd)

        # ── 2. 滑窗平均平滑（抑制单帧抖动） ─────────────────────────
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            smoothed = np.convolve(counts, kernel, mode="same")
        else:
            smoothed = counts.astype(np.float32)

        # ── 3. 每帧 in-match 布尔序列 ────────────────────────────────
        in_match = smoothed >= self.min_players_match

        # ── 4. 找连续的 non-match 段（长度 ≥ min_non_match_seconds 才算）
        min_gap = int(self.min_non_match_seconds * self.fps)
        non_match_spans = []  # [(start, end)] 左闭右开
        i = 0
        while i < total_frames:
            if not in_match[i]:
                j = i
                while j < total_frames and not in_match[j]:
                    j += 1
                if (j - i) >= min_gap:
                    non_match_spans.append((i, j))
                i = j
            else:
                i += 1

        # ── 5. 合成分段列表 ─────────────────────────────────────────
        segments = []
        cursor = 0
        # 找"主中场"：只从视频中段 [25%, 75%] 的候选里挑最长那个。
        # 这样可以避免把开场/终场的非比赛段（广告、球员入场、赛后采访）误判为中场，
        # 防止统计把开场 10 分钟+真正比赛 60 分钟切成莫名其妙的"上半场 10min/下半场 60min"。
        mid_low  = int(total_frames * 0.25)
        mid_high = int(total_frames * 0.75)
        halftime_candidates = [
            s for s in non_match_spans
            if mid_low <= s[0] <= mid_high or mid_low <= s[1] <= mid_high
        ]
        halftime_span = None
        if halftime_candidates:
            halftime_span = max(halftime_candidates, key=lambda s: s[1] - s[0])

        if halftime_span is None:
            # 无中场 → 整个视频就是一段比赛
            segments.append(self._mk_seg("match", 0, total_frames))
        else:
            hs, he = halftime_span
            if hs > 0:
                segments.append(self._mk_seg("first_half", 0, hs))
            segments.append(self._mk_seg("halftime", hs, he))
            if he < total_frames:
                segments.append(self._mk_seg("second_half", he, total_frames))

        return segments

    def _mk_seg(self, seg_type: str, start: int, end: int) -> dict:
        return {
            "type":        seg_type,
            "start_frame": int(start),
            "end_frame":   int(end),
            "start_sec":   round(start / self.fps, 2),
            "end_sec":     round(end   / self.fps, 2),
            "duration_sec": round((end - start) / self.fps, 2),
        }


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
        self.tracker      = sv.ByteTrack()  # 只用于球员追踪
        self.class_names  = self.model.names
        self.player_id = self.ball_id = self.referee_id = None
        for i, name in self.class_names.items():
            n = name.lower()
            if   "player" in n or "person" in n: self.player_id   = i
            elif "ball"   in n:                   self.ball_id     = i
            elif "referee" in n:                  self.referee_id  = i
        if self.player_id is None: self.player_id = 0

    def _process_detections(self, ds: "sv.Detections", fidx: int, tracks: dict):
        """
        将一帧的 sv.Detections 写入 tracks：
        - 球：直接从原始 YOLO 输出取最高置信度（不经 ByteTrack）
        - 球员：只把球员送入 sv.ByteTrack，裁判和球不混入
        """
        tracks["players"][fidx] = {}
        tracks["referees"][fidx] = {}
        tracks["ball"][fidx]    = {}

        if ds is None or len(ds) == 0:
            return

        # Goalkeeper → player
        for k, cid in enumerate(ds.class_id):
            if "goalkeeper" in self.class_names[cid].lower():
                ds.class_id[k] = self.player_id

        # 球：原始 YOLO 输出，取置信度最高的一个
        ball_best_conf = -1
        for k in range(len(ds)):
            if ds.class_id[k] == self.ball_id:
                conf = float(ds.confidence[k])
                if conf > ball_best_conf:
                    ball_best_conf = conf
                    tracks["ball"][fidx][1] = {"bbox": ds.xyxy[k].tolist()}

        # ByteTrack：只送球员（裁判和球都排除）
        player_ds = ds[ds.class_id == self.player_id]
        d_tracks  = self.tracker.update_with_detections(player_ds)
        for d in d_tracks:
            bbox, tid = d[0].tolist(), d[4]
            tracks["players"][fidx][tid] = {"bbox": bbox}

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
            results      = self.model.predict(batch_frames, conf=PLAYER_CONF, iou=0.45,
                                              verbose=False, half=True, imgsz=1280)
            for res, fidx in zip(results, batch_idx):
                det_dict[fidx] = res

        # ByteTrack 必须按帧序处理以维持追踪状态
        for fidx in sorted(det_dict.keys()):
            ds = sv.Detections.from_ultralytics(det_dict[fidx])
            self._process_detections(ds, fidx, tracks)

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
                                    chunk_size: int = 500,
                                    progress_callback=None) -> dict:
        """流式版本：分块处理视频，ByteTrack 跨块保持连续状态。

        Args:
            progress_callback: 每个 chunk 完成后调用，签名：
                               callback(ratio: float, frames_done: int,
                                        frames_total: int, eta_sec: float)
        """
        import time as _time
        tracks = {"players":  [{} for _ in range(total_frames)],
                  "referees": [{} for _ in range(total_frames)],
                  "ball":     [{} for _ in range(total_frames)]}

        t_start = _time.perf_counter()

        for start_idx, chunk in stream_video_chunks(video_path, chunk_size):
            det_local = list(range(0, len(chunk), YOLO_DETECTION_STRIDE))
            det_dict  = {}
            for i in range(0, len(det_local), YOLO_BATCH_SIZE):
                batch_l = det_local[i:i+YOLO_BATCH_SIZE]
                results = self.model.predict([chunk[j] for j in batch_l],
                                             conf=PLAYER_CONF, iou=0.45,
                                             verbose=False, half=True, imgsz=1280)
                for res, local_idx in zip(results, batch_l):
                    det_dict[local_idx] = res

            # ByteTrack 必须按帧序处理
            for local_idx in sorted(det_dict.keys()):
                global_idx = start_idx + local_idx
                if global_idx >= total_frames:
                    break
                ds = sv.Detections.from_ultralytics(det_dict[local_idx])
                self._process_detections(ds, global_idx, tracks)

            # ── 进度汇报 + ETA ─────────────────────────────────────────
            frames_done = min(start_idx + len(chunk), total_frames)
            ratio = frames_done / total_frames if total_frames > 0 else 1.0
            elapsed = _time.perf_counter() - t_start
            eta = (elapsed / ratio) * (1.0 - ratio) if ratio > 0 else 0.0
            print(f"[YOLO] {frames_done}/{total_frames} frames "
                  f"({ratio*100:.0f}%)  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
            if progress_callback:
                progress_callback(ratio, frames_done, total_frames, eta)

            # ── 内存监控 + 必要时 GC ───────────────────────────────────
            _check_memory_and_gc()

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
        # 比例坐标，适应任意分辨率（之前硬编码 1080p 的 1350/1900，4K 或 720p 会偏）
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.70), int(h * 0.79)
        x2, y2 = int(w * 0.99), int(h * 0.90)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), -1)
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
            # 字体 / 线宽按分辨率缩放，避免 4K 时文字显得迷你
            scale = min(w, h) / 1080.0
            font_scale = 1.0 * scale
            thickness  = max(1, int(3 * scale))
            # 两行文字在 box 内均匀分布
            tx = x1 + int(50 * scale)
            ty1 = y1 + int((y2 - y1) * 0.42)
            ty2 = y1 + int((y2 - y1) * 0.82)
            cv2.putText(frame, f"Team 1: {team_1*100:.0f}%", (tx, ty1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, t1_color, thickness)
            cv2.putText(frame, f"Team 2: {team_2*100:.0f}%", (tx, ty2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, t2_color, thickness)
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
                                           conf=0.1, verbose=False, half=True, imgsz=416)
            for res, fidx in zip(results, batch_idx):
                kps = {}
                if res.keypoints is not None and res.keypoints.xy.shape[1] > 0:
                    xy    = res.keypoints.xy[0].cpu().numpy()
                    confs = (res.keypoints.conf[0].cpu().numpy()
                             if res.keypoints.conf is not None
                             else np.ones(len(xy)))
                    for kid, (x, y) in enumerate(xy):
                        if confs[kid] > 0.65 and (x!=0 or y!=0):
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
                                               conf=0.1, verbose=False, half=True, imgsz=416)
                for res, local_idx in zip(results, batch_l):
                    global_idx = start_idx + local_idx
                    kps = {}
                    if res.keypoints is not None and res.keypoints.xy.shape[1] > 0:
                        xy    = res.keypoints.xy[0].cpu().numpy()
                        confs = (res.keypoints.conf[0].cpu().numpy()
                                 if res.keypoints.conf is not None
                                 else np.ones(len(xy)))
                        for kid, (x, y) in enumerate(xy):
                            if confs[kid] > 0.65 and (x != 0 or y != 0):
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
        1:  (0,     1450),   # big_rect_left_top_pt1
        2:  (2015,  1450),   # big_rect_left_top_pt2
        3:  (0,     5550),   # big_rect_left_bottom_pt1
        4:  (2015,  5550),   # big_rect_left_bottom_pt2
        5:  (0,     2584),   # small_rect_left_top_pt1
        6:  (550,   2584),   # small_rect_left_top_pt2
        7:  (0,     4416),   # small_rect_left_bottom_pt1
        8:  (550,   4416),   # small_rect_left_bottom_pt2
        9:  (0,     7000),   # sideline_bottom_left
        # 10: left_semicircle_right — 跳过，影响homography精度
        11: (6000,  0),      # center_line_top
        12: (6000,  7000),   # center_line_bottom
        13: (6000,  2585),   # center_circle_top
        14: (6000,  4415),   # center_circle_bottom
        # 15: field_center — 跳过，影响homography精度
        16: (12000, 0),      # sideline_top_right
        17: (12000, 1450),   # big_rect_right_top_pt1
        18: (9985,  1450),   # big_rect_right_top_pt2
        19: (12000, 5550),   # big_rect_right_bottom_pt1
        20: (9985,  5550),   # big_rect_right_bottom_pt2
        21: (12000, 2584),   # small_rect_right_top_pt1
        22: (11450, 2584),   # small_rect_right_top_pt2
        23: (12000, 4416),   # small_rect_right_bottom_pt1
        24: (11450, 4416),   # small_rect_right_bottom_pt2
        25: (12000, 7000),   # sideline_bottom_right
        # 26: right_semicircle_left — 跳过，影响homography精度
        27: (5085,  3500),   # center_circle_left
        28: (6915,  3500),   # center_circle_right
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

        # 提前热身：扫描所有帧，找到第一个可靠 homography（≥6 个关键点）
        # 作为初始 fallback，避免开头几秒因 _last_transformer=None 被跳过
        if self._last_transformer is None:
            for kps in kps_list:
                src0, dst0 = [], []
                for kid, pos in kps.items():
                    target = self.SOCCANA_PITCH_COORDS.get(kid)
                    if target is not None:
                        src0.append(pos); dst0.append(target)
                if len(src0) >= 8:
                    d0 = np.array(dst0)
                    if (d0[:,0].max()-d0[:,0].min() > 3000 and
                            d0[:,1].max()-d0[:,1].min() > 1500):
                        self._last_transformer = SportsViewTransformer(
                            source=np.array(src0), target=d0)
                        break

        for fnum, kps in enumerate(kps_list):
            src, dst = [], []
            for kid, pos in kps.items():
                target = self.SOCCANA_PITCH_COORDS.get(kid)
                if target is not None:
                    src.append(pos); dst.append(target)

            if len(src) >= 6:
                dst_arr = np.array(dst, dtype=np.float32)
                src_arr = np.array(src, dtype=np.float32)
                x_span  = dst_arr[:, 0].max() - dst_arr[:, 0].min()
                y_span  = dst_arr[:, 1].max() - dst_arr[:, 1].min()
                # 关键点必须在场地上有足够分布才更新，否则退化矩阵
                # x_span > 3000 (~25% 场长) 且 y_span > 1500 (~21% 场宽)
                if x_span > 3000 and y_span > 1500:
                    # RANSAC inlier 检查：inlier < 50% 说明 keypoint ID 乱了，拒绝更新
                    import cv2 as _cv2
                    _, mask = _cv2.findHomography(src_arr, dst_arr, _cv2.RANSAC, 500.0)
                    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0
                    if inlier_ratio >= 0.5:
                        transformer = SportsViewTransformer(source=src_arr, target=dst_arr)
                        self._last_transformer = transformer
                    elif self._last_transformer is not None:
                        transformer = self._last_transformer
                    else:
                        continue
                elif self._last_transformer is not None:
                    transformer = self._last_transformer
                else:
                    continue
            elif self._last_transformer is not None:
                # 关键点不足 → 用上一个稳定 homography 做 fallback
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
        # Convert to HSV for lighting-robust clustering
        top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        km  = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(top_hsv.reshape(-1,3))
        lbl = km.labels_.reshape(top.shape[0], top.shape[1])
        corners = [lbl[0,0], lbl[0,-1], lbl[-1,0], lbl[-1,-1]]
        bg_cluster = max(set(corners), key=corners.count)
        kit_cluster = 1 - bg_cluster
        # Return the kit cluster center converted back to BGR for downstream use
        kit_hsv = km.cluster_centers_[kit_cluster].astype(np.uint8).reshape(1,1,3)
        return cv2.cvtColor(kit_hsv, cv2.COLOR_HSV2BGR).reshape(3).astype(float)

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

# ── 纯 OpenCV 小地图渲染（替代 sports lib，速度快 10-20x）────────────────

_MM_W      = 840   # 输出宽度（像素）
_MM_H      = 560   # 输出高度（像素）
_MM_MARGIN = 40    # 球场边距（像素）
# SOCCANA_PITCH_COORDS 坐标系：x 0-12000, y 0-7000 (÷100 → 120×70)
# 必须与 clamp_pitch_position 和 make_pitch_background 保持一致，
# 否则球员位置和场地线条坐标系不同，小地图出现~14%偏移。
_PITCH_LEN = 120.0
_PITCH_WID = 70.0


def _mm_p2px(pos, fps_w=_MM_W, fps_h=_MM_H, margin=_MM_MARGIN):
    """将球场坐标（米, 0-105 × 0-68）转换为像素坐标。"""
    pw = fps_w - 2 * margin
    ph = fps_h - 2 * margin
    px = margin + int(float(pos[0]) / _PITCH_LEN * pw)
    py = margin + int(float(pos[1]) / _PITCH_WID * ph)
    return (px, py)


def _hex_to_bgr(hex_str: str) -> tuple:
    """'#RRGGBB' → (B, G, R) for OpenCV."""
    h = hex_str.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def make_pitch_background(width=_MM_W, height=_MM_H, margin=_MM_MARGIN) -> np.ndarray:
    """
    纯 OpenCV 绘制标准足球场底图（只需生成一次，后续每帧 copy）。
    坐标系与 SOCCANA_PITCH_COORDS ÷100 对齐：(0,0)=左上，(120,70)=右下。
    线条坐标从 SOCCANA_PITCH_COORDS/100 导出，确保与 position_transformed 一致。
    """
    bg = np.full((height, width, 3), (34, 120, 34), dtype=np.uint8)  # 绿草地

    def p(x, y):  # SOCCANA 坐标(÷100) → 像素
        return _mm_p2px((x, y), width, height, margin)

    W = (255, 255, 255)
    LT = 2

    # ── 外框 (SOCCANA: 0,0 → 12000,7000 → ÷100 = 120×70) ──
    cv2.rectangle(bg, p(0, 0), p(120, 70), W, LT)

    # ── 中线 (center_line_top/bottom: x=6000 → 60) ──
    cv2.line(bg, p(60, 0), p(60, 70), W, LT)

    # ── 中圆 (center_circle_left=5085 → 50.85, center=60, r=9.15) ──
    cx, cy = p(60, 35)
    r_px = int(9.15 / _PITCH_LEN * (width - 2 * margin))
    cv2.circle(bg, (cx, cy), r_px, W, LT)
    cv2.circle(bg, (cx, cy), 4, W, -1)

    # ── 禁区 (big_rect: x=2015→20.15, y=1450/5550→14.5/55.5) ──
    cv2.rectangle(bg, p(0, 14.5),   p(20.15, 55.5), W, LT)
    cv2.rectangle(bg, p(99.85, 14.5), p(120, 55.5), W, LT)

    # ── 小禁区 (small_rect: x=550→5.5, y=2584/4416→25.84/44.16) ──
    cv2.rectangle(bg, p(0, 25.84),    p(5.5, 44.16), W, LT)
    cv2.rectangle(bg, p(114.5, 25.84), p(120, 44.16), W, LT)

    # ── 点球点 ──
    # FIFA 11m，按 SOCCANA 120m 坐标系等比缩放：11/105*120 = 12.57m
    # 右侧对称：120 - 12.57 = 107.43m
    cv2.circle(bg, p(12.57, 35), 4, W, -1)
    cv2.circle(bg, p(107.43, 35), 4, W, -1)

    # ── 点球弧（只画禁区外的部分）──
    # 禁区线在 x=20.15，点球点在 x=12.57，距离=7.58m，弧半径=9.15m
    # 弧露出禁区外的半角 = arccos(7.58/9.15) ≈ 34°
    cv2.ellipse(bg, p(12.57, 35),  (r_px, r_px), 0, -34, 34,   W, LT)
    cv2.ellipse(bg, p(107.43, 35), (r_px, r_px), 0, 146, 214,  W, LT)

    # ── 角球弧 (radius 1m) ──
    cr = int(1.0 / _PITCH_LEN * (width - 2 * margin))
    cv2.ellipse(bg, p(0,   0),  (cr, cr), 0,   0,  90, W, LT)
    cv2.ellipse(bg, p(120, 0),  (cr, cr), 0,  90, 180, W, LT)
    cv2.ellipse(bg, p(0,  70),  (cr, cr), 0, 270, 360, W, LT)
    cv2.ellipse(bg, p(120, 70), (cr, cr), 0, 180, 270, W, LT)

    return bg


def render_minimap_frame(frame_idx: int, tracks: dict,
                         tracked_bboxes: dict, team_control: np.ndarray,
                         config, hex_t1: str, hex_t2: str,
                         ball_trail: list = None,
                         pitch_bg: np.ndarray = None,
                         fps: float = 24.0) -> np.ndarray:
    """
    纯 OpenCV 单帧小地图渲染（不再调用 draw_pitch/draw_points_on_pitch）。

    pitch_bg: 预渲染的球场底图（make_pitch_background() 结果）。
              若为 None 则临时生成（兼容旧调用，但较慢）。
    fps:      用于时间戳显示（默认 24）。
    """
    # 底图：优先使用传入的预渲染版本
    frame = (pitch_bg.copy() if pitch_bg is not None
             else make_pitch_background())
    h, w = frame.shape[:2]

    bgr_t1 = _hex_to_bgr(hex_t1)
    bgr_t2 = _hex_to_bgr(hex_t2)

    # ── 找被追踪目标 ──────────────────────────────────────────────────
    tracked_pos  = None
    tracked_team = None
    if frame_idx in tracked_bboxes and frame_idx < len(tracks["players"]):
        sx, sy, sw, sh = tracked_bboxes[frame_idx]
        center = (sx + sw / 2, sy + sh / 2)
        min_d  = 150
        for pid, info in tracks["players"][frame_idx].items():
            if not info or "bbox" not in info:
                continue
            b = info["bbox"]
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            d = measure_distance(center, (cx, cy))
            if d < min_d:
                min_d = d
                # position_transformed 已在 0-105m × 0-68m 区间，直接给 _mm_p2px。
                # position_minimap 是遗留 config-scale（~0-10500），仅供 heatmap/sprint 的
                # draw_points_on_pitch 使用；新 OpenCV 渲染器必须用 position_transformed。
                tp = info.get("position_transformed")
                if tp and len(tp) == 2 and not any(np.isnan(p) for p in tp):
                    tracked_pos  = tp
                    tracked_team = info.get("team")

    # ── 球轨迹拖尾（最近 30 帧）──────────────────────────────────────
    if ball_trail and len(ball_trail) > 1:
        n = len(ball_trail)
        for bi, pos in enumerate(ball_trail):
            try:
                alpha = (bi + 1) / n                    # 0=旧 → 1=新
                intensity = int(alpha * 200) + 55       # 55→255
                r_trail   = max(2, int(alpha * 6))
                cv2.circle(frame, _mm_p2px(pos, w, h),
                           r_trail, (0, intensity, 0), -1)
            except Exception:
                pass

    # ── 绘制普通球员点 ────────────────────────────────────────────────
    if frame_idx < len(tracks["players"]):
        for pid, info in tracks["players"][frame_idx].items():
            if not info:
                continue
            pos = info.get("position_transformed")  # meters, matches _mm_p2px
            if not pos or len(pos) != 2 or any(np.isnan(p) for p in pos):
                continue
            px = _mm_p2px(pos, w, h)
            color = bgr_t1 if info.get("team") == 1 else bgr_t2
            cv2.circle(frame, px, 10, (0, 0, 0),  -1)   # 黑边
            cv2.circle(frame, px,  8, color,       -1)   # 队伍色

    # ── 高亮追踪球员（金环 + 队伍色内芯）─────────────────────────────
    if tracked_pos is not None:
        tpx = _mm_p2px(tracked_pos, w, h)
        inner = bgr_t1 if tracked_team == 1 else bgr_t2
        cv2.circle(frame, tpx, 18, (0, 215, 255), -1)   # 金色外环
        cv2.circle(frame, tpx, 12, (0,   0,   0), -1)   # 黑边
        cv2.circle(frame, tpx, 10, inner,          -1)   # 队伍色内芯

    # ── 足球 ─────────────────────────────────────────────────────────
    ball_pos = tracks["ball"][frame_idx].get(1, {}).get("position_transformed") \
               if frame_idx < len(tracks["ball"]) else None
    if ball_pos and len(ball_pos) == 2 and not any(np.isnan(p) for p in ball_pos):
        bpx = _mm_p2px(ball_pos, w, h)
        cv2.circle(frame, bpx, 7, (0, 0, 0),       -1)  # 黑边
        cv2.circle(frame, bpx, 5, (255, 255, 255),  -1)  # 白球

    # ── 控球率角标（右上角半透明黑底）────────────────────────────────
    ctrl = team_control[:frame_idx + 1]
    t1c  = int(np.sum(ctrl == 1))
    t2c  = int(np.sum(ctrl == 2))
    tot  = t1c + t2c or 1
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 225, 5), (w - 5, 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"T1 {t1c/tot*100:.0f}%   T2 {t2c/tot*100:.0f}%",
                (w - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── 时间戳（左上角）──────────────────────────────────────────────
    secs = frame_idx / max(fps, 1.0)
    cv2.putText(frame, f"{int(secs // 60):02d}:{secs % 60:04.1f}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame
