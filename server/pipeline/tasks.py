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
import sys
import subprocess
import pickle
import threading
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

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
    SceneChangeDetector,
    stream_video_chunks,
    read_frames_at_indices,
    read_video,
    _check_memory_and_gc,
    make_pitch_background,
    KeypointDetector,
    ViewTransformer,
    AccurateSpeedEstimator,
    TeamAssigner,
    SmartBallPossessionDetector,
    save_video,
    bgr_to_hex,
    render_minimap_frame,
    measure_distance,
    run_segment_detection,
)

try:
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from sports.configs.soccer import SoccerPitchConfiguration
    import supervision as sv
    HAS_SPORTS = True
except ImportError:
    HAS_SPORTS = False

# server/pipeline/tasks.py → parents[2] is the repo root.
# Env vars YOLO_MODEL_PATH / KEYPOINT_MODEL_PATH are set by
# server.models.weights.ensure_weights() at startup, so REPO_ROOT is only hit
# when running the pipeline outside the FastAPI server (ad-hoc scripts, tests).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_path(env_var: str, preferred_name: str) -> str:
    """Resolve a weight file path lazily — re-read env var EVERY call so we
    pick up paths set by ensure_weights() during lifespan startup. Module-level
    capture would freeze the value before weights are downloaded and break
    every YOLO load downstream."""
    configured = os.environ.get(env_var)
    if configured:
        return configured

    # Prefer the workspace weights dir used by the server; fall back to repo root
    # for legacy local-dev layouts.
    workspace_weights = Path("/workspace/weights") / preferred_name
    if workspace_weights.exists():
        return str(workspace_weights)
    return str(REPO_ROOT / preferred_name)


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ── 模型路径：必须 lazy 解析 ─────────────────────────────────────────────────
# 不要在 module import 时缓存：FastAPI lifespan 启动顺序是
#   1) import server.main → import routes → import pipeline.tasks
#   2) lifespan 运行 ensure_weights() 才设 YOLO_MODEL_PATH 等 env var
# 如果在 import 时就把 MODEL_PATH 算出来，env var 还是空的 → 拿到 /app/soccana_best.pt
# (不存在) → 单帧检测、全局分析全部 FileNotFoundError。这是 RunPod 部署后所有
# YOLO 推理失败的根本原因。
def get_yolo_model_path() -> str:
    return _resolve_model_path("YOLO_MODEL_PATH", "soccana_best.pt")


def get_keypoint_model_path() -> str:
    return _resolve_model_path("KEYPOINT_MODEL_PATH", "soccana_kpts_best.pt")


def get_samurai_script() -> str:
    return os.environ.get("SAMURAI_SCRIPT", "/app/samurai/scripts/demo.py")


# SAMURAI 插值：缺口 ≤ 30 帧线性插，超过则保留 NaN 由下游跳过
MAX_INTERP_GAP_FRAMES = 30


@contextmanager
def _video_capture(path: str):
    """cv2.VideoCapture 上下文管理器，保证异常也释放句柄"""
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()


def _atomic_pickle_dump(obj, target_path: Path) -> None:
    """原子写 pickle：先写 .tmp → fsync → os.replace。
    顺手把同路径的 in-memory cache 清掉 —— 否则 re-analyze 重写了 pickle，
    但 _TRACKS_CACHE 里还是旧对象，下游 feature 任务读到 stale tracks。
    """
    target_path = Path(target_path)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    print(f"[INFO] Writing pickle to {target_path} (tmp: {tmp_path})")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    # Invalidate in-memory cache for this path (no-op if not cached yet)
    try:
        with _TRACKS_CACHE_LOCK:
            _TRACKS_CACHE.pop(str(target_path), None)
    except NameError:
        # _TRACKS_CACHE defined later in module; first call during import-time
        # benchmark setup may hit this — safe to skip, nothing is cached yet.
        pass
    try:
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Pickle write complete: {target_path.name} ({size_mb:.1f} MB)")
    except Exception:
        print(f"[INFO] Pickle write complete: {target_path.name}")


def _probe_fps(path: str):
    """用 ffprobe 获取 fps（VideoCapture 返回 0 时的 fallback）"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
        if "/" in out:
            num, den = out.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else None
    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════
# 阶段 0：首帧球员检测（轻量同步任务，给前端选择球员用）
# ═══════════════════════════════════════════════════════════════════════

_cached_yolo_model = None

def _get_yolo_model():
    """Return a cached YOLO instance — loads once, reused across all calls.

    Resolves the weight path *now* (not at module import) so ensure_weights()
    has time to set YOLO_MODEL_PATH during lifespan startup. Without this the
    cache key would be the wrong /app fallback even after weights download."""
    global _cached_yolo_model
    if _cached_yolo_model is None:
        from ultralytics import YOLO as _YOLO
        path = get_yolo_model_path()
        _require_file(path, "YOLO model")
        _cached_yolo_model = _YOLO(path)
    return _cached_yolo_model


def detect_frame_players(session_id: str, session: dict, frame_idx: int,
                         sm: SessionManager) -> dict:
    """
    在指定帧上运行 YOLO，返回球员 bbox 列表 + 标注后的帧图。

    Returns:
        {
          "players": [{"id": 1, "bbox": [x1,y1,x2,y2]}, ...],
          "annotated_frame_path": "first_frame.jpg",  # relative to session dir
          "image_dimensions": {"width": W, "height": H},
        }
    """
    from ultralytics import YOLO

    video_path = session.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Session {session_id} has no video")

    cap = cv2.VideoCapture(video_path)
    try:
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    finally:
        cap.release()

    h, w = frame.shape[:2]
    model = _get_yolo_model()
    results = model.predict([frame], conf=0.30, iou=0.45, verbose=False)[0]

    # YOLO returns class names dict; pick the player class id
    class_names = model.names
    player_class_id = None
    for cid, name in class_names.items():
        n = name.lower()
        if "player" in n or "person" in n:
            player_class_id = cid
            break
    if player_class_id is None:
        player_class_id = 0

    players = []
    annotated = frame.copy()
    if results.boxes is not None and len(results.boxes) > 0:
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        # Map YOLO classes: goalkeeper → player
        for k, cid in enumerate(cls_ids):
            if "goalkeeper" in class_names[int(cid)].lower():
                cls_ids[k] = player_class_id

        # Sort by confidence descending so the best players come first
        order = np.argsort(-confs)
        for player_no, k in enumerate(order, start=1):
            if int(cls_ids[k]) != player_class_id:
                continue
            x1, y1, x2, y2 = boxes_xyxy[k].tolist()
            players.append({
                "id": player_no,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confs[k]),
                "name": f"Player {player_no}",
                "number": "?",
                "avatar": "👤",
            })
            # Draw box on annotated frame
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 215, 255), 2)
            cv2.putText(annotated, f"#{player_no}", (int(x1), max(int(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    out_dir = sm.session_output_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_path = "first_frame.jpg"
    cv2.imwrite(str(out_dir / rel_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        "players": players,
        "annotated_frame_path": rel_path,
        "image_dimensions": {"width": int(w), "height": int(h)},
    }


# ═══════════════════════════════════════════════════════════════════════
# 阶段 1：SAMURAI 追踪
# ═══════════════════════════════════════════════════════════════════════

def _run_samurai_segment(seg_idx: int, session_id: str, output_dir: Path,
                          video_path: str, segment: dict,
                          orig_w: int, orig_h: int, total_orig_frames: int,
                          resize_factor: float, skip_step: int,
                          samurai_script: str, sam2_model_path: str,
                          samurai_root: str) -> dict:
    """
    Run SAMURAI on ONE segment as a subprocess. Designed to be called from a
    thread pool so multiple segments can run concurrently — each subprocess
    has its own CUDA context but they share the GPU happily up to 4× on a
    16 GB+ card (SAM2 base_plus ≈ 500 MB GPU each).

    Returns {"start": int, "end": int, "sparse_bboxes": {orig_fid: [x,y,w,h]}}.
    Raises RuntimeError on failure so the orchestrator can decide whether to
    fail the whole job or just skip that segment.
    """
    start_frame = int(segment["start_frame"])
    end_frame   = int(segment["end_frame"])
    bbox        = segment["bbox"]

    new_w = int(orig_w * resize_factor)
    new_h = int(orig_h * resize_factor)

    # Per-segment frame extraction dir + bbox seed file
    frames_dir = output_dir / f"samurai_frames_{seg_idx}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    init_txt = output_dir / f"input_bbox_{seg_idx}.txt"
    output_video = output_dir / f"samurai_temp_{seg_idx}.mp4"
    output_txt = output_dir / f"samurai_temp_{seg_idx}_bboxes.txt"

    # Extract this segment's frames at stride. Use ffmpeg between(n,a,b)
    # so we don't decode frames outside the segment range.
    vf = (
        f"select='gte(n\\,{start_frame})*lte(n\\,{end_frame - 1})"
        f"*not(mod(n-{start_frame}\\,{skip_step}))',"
        f"scale={new_w}:{new_h}"
    )
    cmd_ext = [
        "ffmpeg", "-i", video_path, "-y",
        "-vf", vf, "-vsync", "0", "-q:v", "2",
        str(frames_dir / "%05d.jpg"),
    ]
    res = subprocess.run(cmd_ext, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"[seg {seg_idx}] ffmpeg extract failed: {res.stderr[-1000:]}"
        )
    if not list(frames_dir.glob("*.jpg")):
        raise RuntimeError(f"[seg {seg_idx}] no frames extracted")

    # Write seed bbox (scaled to resize_factor; clamped to new dims)
    bx = max(0.0, min(bbox["x"] * resize_factor, new_w - 2.0))
    by = max(0.0, min(bbox["y"] * resize_factor, new_h - 2.0))
    bw = max(1.0, min(bbox["w"] * resize_factor, new_w - bx))
    bh = max(1.0, min(bbox["h"] * resize_factor, new_h - by))
    if bw < 8 or bh < 8:
        raise ValueError(
            f"[seg {seg_idx}] bbox too small after scaling: {bw:.1f}x{bh:.1f}"
        )
    with open(init_txt, "w") as f:
        f.write(f"{bx},{by},{bw},{bh}\n")

    # Spawn SAMURAI subprocess
    cmd = [
        "python", samurai_script,
        "--video_path", str(frames_dir),
        "--txt_path", str(init_txt),
        "--video_output_path", str(output_video),
        "--model_path", sam2_model_path,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{samurai_root}:{samurai_root}/sam2:" + env.get("PYTHONPATH", "")
    )
    env["HYDRA_FULL_ERROR"] = "1"
    # subprocess 超时：基于 segment 长度估算，最少 10 分钟。SAMURAI ≈ 0.5×
    # 视频时长，给 5× 余量。timeout=None 之前能让一个挂死的子进程吃掉整个
    # Serverless worker 时窗 + GPU 显存。
    _seg_sec = max(1.0, (end_frame - start_frame) / 25.0)
    _samurai_timeout = max(600.0, _seg_sec * 5.0)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_samurai_timeout,
            env=env, cwd=samurai_root,
        )
    except subprocess.TimeoutExpired as _te:
        raise RuntimeError(
            f"[seg {seg_idx}] SAMURAI timed out after {_samurai_timeout/60:.1f} min "
            f"(segment {start_frame}..{end_frame})"
        ) from _te
    if proc.returncode != 0:
        tail = proc.stderr[-2000:] if proc.stderr else ""
        raise RuntimeError(
            f"[seg {seg_idx}] SAMURAI exited {proc.returncode}: {tail}"
        )

    if not output_txt.exists():
        raise FileNotFoundError(
            f"[seg {seg_idx}] SAMURAI produced no bbox output at {output_txt}"
        )

    # Parse + scale back to original coords. Output file format:
    # frame_id, [class], x, y, w, h
    scale_back = 1.0 / resize_factor
    sparse_bboxes: dict[int, list[float]] = {}
    with open(output_txt) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            original_fid = start_frame + (fid * skip_step)
            x = float(parts[2]); y = float(parts[3])
            w = float(parts[4]); h = float(parts[5])
            if w > 0 and h > 0 and original_fid < total_orig_frames:
                sparse_bboxes[original_fid] = [
                    x * scale_back, y * scale_back,
                    w * scale_back, h * scale_back,
                ]

    # Clean up extracted frames immediately to bound disk usage
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    return {
        "seg_idx": seg_idx,
        "start": start_frame,
        "end": end_frame,
        "sparse_bboxes": sparse_bboxes,
        "frames_emitted": len(sparse_bboxes),
    }


# ── Parallel YOLO detection workers ─────────────────────────────────────────
# Module-level so they are picklable for ProcessPoolExecutor(mp_context='spawn').

def _yolo_parallel_worker(args: dict) -> str:
    """
    Subprocess worker: run detection on one video segment, write partial
    tracks to a pkl file, return its path.
    Called via ProcessPoolExecutor(mp_context='spawn').
    """
    import pickle
    from pathlib import Path
    from .analysis_core import (
        run_segment_detection, Tracker, CameraMovementEstimator, ViewTransformer,
    )

    seg_idx    = args["seg_idx"]
    video_path = args["video_path"]
    start      = args["start_frame"]
    end        = args["end_frame"]
    yolo_path  = args["yolo_path"]
    kpt_path   = args["kpt_path"]
    output_dir = Path(args["output_dir"])
    batch_size = args.get("batch_size", 15)

    raw_tracks, cam_mov, kps, seg_start, seg_end = run_segment_detection(
        video_path, start, end, yolo_path, kpt_path, batch_size=batch_size,
    )

    # Post-process (everything except add_speed_and_distance):
    tracker_obj = Tracker(yolo_path)
    tracker_obj.add_position_to_tracks(raw_tracks)
    cam_obj = CameraMovementEstimator.from_video_path(video_path)
    cam_obj.add_adjust_positions_to_tracks(raw_tracks, cam_mov)
    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(raw_tracks, kps)
    # Interpolate ball within segment (cross-segment gaps handled after merge)
    raw_tracks["ball"] = tracker_obj.interpolate_ball_positions(raw_tracks["ball"])

    out_path = output_dir / f"yolo_partial_{seg_idx}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "seg_idx": seg_idx,
            "start_frame": seg_start,
            "end_frame": seg_end,
            "tracks": raw_tracks,
            "cam_mov": cam_mov,
        }, f, protocol=4)

    print(f"[YOLO-PAR] seg {seg_idx} done: frames {seg_start}–{seg_end}, "
          f"pkl={out_path.name}")
    return str(out_path)


def _run_yolo_parallel(video_path: str, total_frames: int, n_segs: int,
                        yolo_path: str, kpt_path: str, output_dir: Path,
                        fps: float,
                        progress_callback=None) -> tuple:
    """
    Split video into n_segs equal segments, run YOLO+KPT+flow on each in
    parallel subprocesses (spawn context to avoid CUDA fork issues).
    Returns (merged_tracks, merged_cam_mov) ready for speed/team/possession.
    """
    import multiprocessing as mp
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from .analysis_core import YOLO_BATCH_SIZE

    seg_size = total_frames // n_segs
    # batch_size per subprocess — reduce to avoid OOM when running n_segs parallel GPU procs
    batch_size = max(8, YOLO_BATCH_SIZE // n_segs)
    seg_args = []
    for i in range(n_segs):
        start = i * seg_size
        end   = total_frames if i == n_segs - 1 else (i + 1) * seg_size
        seg_args.append({
            "seg_idx":     i,
            "video_path":  video_path,
            "start_frame": start,
            "end_frame":   end,
            "yolo_path":   yolo_path,
            "kpt_path":    kpt_path,
            "output_dir":  str(output_dir),
            "batch_size":  batch_size,
        })

    print(f"[YOLO-PAR] {n_segs} segments, "
          f"batch_size={batch_size}, "
          f"~{seg_size / max(fps, 1):.0f}s each")

    ctx = mp.get_context("spawn")
    pkl_paths: dict = {}

    with ProcessPoolExecutor(max_workers=n_segs, mp_context=ctx) as pool:
        futures = {pool.submit(_yolo_parallel_worker, a): a["seg_idx"]
                   for a in seg_args}
        done = 0
        for fut in as_completed(futures):
            seg_idx = futures[fut]
            try:
                pkl_paths[seg_idx] = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"[YOLO-PAR] segment {seg_idx} failed: {exc}") from exc
            done += 1
            if progress_callback:
                progress_callback(done / n_segs, done * seg_size, total_frames, 0)
            print(f"[YOLO-PAR] {done}/{n_segs} segments done")

    # Merge: concatenate by original frame index
    merged_tracks = {
        "players":  [{} for _ in range(total_frames)],
        "referees": [{} for _ in range(total_frames)],
        "ball":     [{} for _ in range(total_frames)],
    }
    merged_cam_mov = [[0.0, 0.0]] * total_frames

    for seg_idx in sorted(pkl_paths):
        pkl_path = pkl_paths[seg_idx]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        start = data["start_frame"]
        seg_tracks = data["tracks"]
        seg_cam    = data["cam_mov"]
        for key in ("players", "referees", "ball"):
            for i, fd in enumerate(seg_tracks[key]):
                if start + i < total_frames:
                    merged_tracks[key][start + i] = fd
        for i, cm in enumerate(seg_cam):
            if start + i < total_frames:
                merged_cam_mov[start + i] = cm
        try:
            os.remove(pkl_path)
        except OSError:
            pass

    return merged_tracks, merged_cam_mov


def run_samurai_tracking(session_id: str, session: dict,
                         player_bbox: dict, sm: SessionManager):
    """Thin wrapper kept for the legacy single-bbox call site in
    routes/analysis.py. The real implementation now lives in
    run_samurai_tracking_multi() — this just packs the one bbox
    into a single-element segments list and forwards.

    player_bbox shape: {"x", "y", "w", "h", "frame": int (optional)}
    """
    start_frame = int(player_bbox.get("frame", 0))
    # The multi path uses [start, end) ranges. We default the end to a
    # big sentinel — it gets clamped to total_orig_frames inside multi.
    segments = [{
        "start_frame": start_frame,
        "end_frame":   10 ** 9,  # clamped to video length inside
        "bbox": {
            "x": float(player_bbox["x"]),
            "y": float(player_bbox["y"]),
            "w": float(player_bbox["w"]),
            "h": float(player_bbox["h"]),
        },
    }]
    return run_samurai_tracking_multi(session_id, session, segments, sm)


def run_samurai_tracking_multi(session_id: str, session: dict,
                                segments: list, sm: SessionManager):
    """
    Multi-segment SAMURAI: each segment runs in its own subprocess in parallel.
    The user picks a bbox at the start of each segment, so we can recover from
    ID switches / substitutions / camera cuts cleanly at segment boundaries.

    Args:
        segments: list of {"start_frame": int, "end_frame": int,
                           "bbox": {"x", "y", "w", "h"}}.
                  end_frame is exclusive. Frames not covered by any segment
                  get linearly interpolated from neighboring segments (or
                  dropped if the gap exceeds MAX_INTERP_GAP_FRAMES).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd   # ← the legacy single-segment fn imports this locally;
                          #   matching that here so the gap-interpolation works.
    try:
        video_path  = session["video_path"]
        output_dir  = sm.session_output_dir(session_id)
        cache_path  = output_dir / "samurai_tracking.pkl"

        if not segments:
            raise ValueError("At least one segment is required")

        sm.update_status(session_id, "tracking", progress=5, stage="samurai_multi_start")

        # Probe video once (cheap)
        if cv2 is None:
            raise ImportError("cv2 is required for frame extraction")
        with _video_capture(video_path) as cap:
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_orig_frames <= 0:
            total_orig_frames = 1500

        # Validate + sort segments by start_frame
        segs_sorted = sorted(
            [dict(s) for s in segments], key=lambda s: int(s["start_frame"])
        )
        for s_ in segs_sorted:
            s_["start_frame"] = max(0, int(s_["start_frame"]))
            s_["end_frame"]   = min(total_orig_frames, int(s_["end_frame"]))
        # Drop empty / inverted ranges before dispatch — an empty ffmpeg
        # select=... command silently produces zero frames, which then makes
        # SAMURAI exit with a confusing "no frames" error.
        segs_sorted = [s_ for s_ in segs_sorted
                       if s_["end_frame"] > s_["start_frame"] + 1]
        if not segs_sorted:
            raise ValueError(
                "All segments collapsed after validation. Check start/end "
                "frames against video length."
            )

        # Resolve sam paths + samurai_root once
        samurai_script = get_samurai_script()
        _require_file(samurai_script, "SAMURAI demo script")
        sam2_model_path = os.environ.get(
            "SAM2_MODEL_PATH",
            "/workspace/weights/sam2.1_hiera_base_plus.pt",
        )
        _require_file(sam2_model_path, "SAM2 checkpoint")
        samurai_root = str(Path(samurai_script).resolve().parents[1])

        RESIZE_FACTOR = 0.5
        SKIP_STEP = 10

        n_segments = len(segs_sorted)

        # 24GB GPU: run ALL segments across ALL periods in parallel.
        # Previously periods were serialised (first half → second half) to
        # stay within 16GB (4 procs × ~2.5GB = ~10GB per period).  On 24GB
        # all 8 segments run simultaneously (~14-16GB total), cutting
        # SAMURAI wall-time roughly in half (e.g. 30 min → 15 min).
        # Hard cap at 8 concurrent workers: each SAMURAI proc uses ~2.5GB,
        # so 8 × 2.5 = ~20GB which fits comfortably in 24GB.  Extra segments
        # (e.g. 3 periods × 4 players = 12) queue behind the first 8 and run
        # as slots free up — still faster than the old serial-by-period path.
        MAX_PARALLEL = int(os.environ.get("SAMURAI_MAX_PARALLEL", "8"))
        max_workers  = min(n_segments, MAX_PARALLEL)
        all_segs = list(enumerate(segs_sorted))
        print(f"[SAMURAI-MULTI] {n_segments} segments — "
              f"{max_workers} running in parallel (cap={MAX_PARALLEL})")

        def _submit_segment(pool, i, seg):
            return pool.submit(
                _run_samurai_segment,
                i, session_id, output_dir, video_path, seg,
                orig_w, orig_h, total_orig_frames,
                RESIZE_FACTOR, SKIP_STEP,
                samurai_script, sam2_model_path, samurai_root,
            )

        results = []
        done = 0
        seg_attempts: dict[int, int] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                _submit_segment(pool, i, seg): i
                for i, seg in all_segs
            }
            for i in futures.values():
                seg_attempts[i] = 1
            pending = set(futures.keys())
            while pending:
                for fut in list(as_completed(pending)):
                    pending.discard(fut)
                    seg_idx = futures[fut]
                    try:
                        r = fut.result()
                        results.append(r)
                        done += 1
                        pct = 5 + int(80 * done / n_segments)
                        sm.update_status(
                            session_id, "tracking", progress=pct,
                            stage=f"samurai_multi ({done}/{n_segments} segments)"
                        )
                        print(f"[SAMURAI-MULTI] seg {r['seg_idx']} done "
                              f"({r['frames_emitted']} sparse frames, "
                              f"range {r['start']}..{r['end']})")
                    except Exception as seg_exc:
                        attempt = seg_attempts[seg_idx]
                        if attempt < 2:
                            seg_attempts[seg_idx] += 1
                            print(f"[SAMURAI-MULTI] seg {seg_idx} attempt "
                                  f"{attempt} failed: {seg_exc} — retrying")
                            retry_seg = next(s for (i, s) in all_segs if i == seg_idx)
                            retry_fut = _submit_segment(pool, seg_idx, retry_seg)
                            futures[retry_fut] = seg_idx
                            pending.add(retry_fut)
                        else:
                            print(f"[SAMURAI-MULTI] seg {seg_idx} failed "
                                  f"after {attempt} attempts: {seg_exc} — "
                                  "interpolation will cover the gap")

        if not results:
            raise RuntimeError(
                "All SAMURAI segments failed. Check stderr for details."
            )

        # Merge: union of all segments' sparse bboxes
        sparse_bboxes: dict[int, list[float]] = {}
        for r in results:
            for fid, box in r["sparse_bboxes"].items():
                sparse_bboxes[fid] = box

        if not sparse_bboxes:
            raise RuntimeError("No SAMURAI bboxes produced across all segments")

        # Linear-interpolate gaps (same logic as single-segment path).
        start_index = segs_sorted[0]["start_frame"]
        end_index = max(sparse_bboxes.keys())
        df = pd.DataFrame(
            index=range(start_index, end_index + 1),
            columns=["x", "y", "w", "h"],
        )
        for fid, box in sparse_bboxes.items():
            if fid <= end_index:
                df.loc[fid] = box
        df = df.astype(float).interpolate(
            method="linear",
            limit=MAX_INTERP_GAP_FRAMES,
            limit_direction="both",
        )

        bboxes_dict: dict[int, tuple] = {}
        for fid, row in df.iterrows():
            if pd.isna(row["x"]) or pd.isna(row["y"]):
                continue
            bboxes_dict[fid] = (row["x"], row["y"], row["w"], row["h"])

        _atomic_pickle_dump({"bboxes": bboxes_dict}, cache_path)

        sm.update_status(
            session_id, "tracking_done",
            progress=100, stage="samurai_done",
            samurai_cache_path=str(cache_path),
            samurai_tracked_frames=len(bboxes_dict),
        )
        print(f"[SAMURAI-MULTI] ✓ merged {len(sparse_bboxes)} sparse → "
              f"{len(bboxes_dict)} interpolated frames")

    except Exception as exc:
        sm.update_status(session_id, "tracking_failed",
                         error=str(exc), stage="samurai_error")
        _log_error("SAMURAI tracking (multi)", session_id, exc)
        raise


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
        _bench_log = []
        def _bench(label, t0):
            elapsed = _time.perf_counter() - t0
            _bench_log.append(f"{label:<25}: {elapsed:.1f}s")
            # Also write to file so it's readable from Colab cell
            try:
                log_path = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark.log")
                with open(os.path.abspath(log_path), "a") as _f:
                    _f.write(f"[BENCH] {label:<25}: {elapsed:.1f}s\n")
            except Exception:
                pass
            return elapsed

        video_path   = session["video_path"]
        # samurai_cache_path may not be set yet on the concurrent path
        # (handler pre-writes it but a refresh race could miss it). The
        # deferred-load block below polls for the file with a 10-min cap,
        # so we just need to not crash here.
        samurai_pkl  = session.get("samurai_cache_path")
        output_dir   = sm.session_output_dir(session_id)
        tracks_cache = output_dir / "tracks.pkl"

        # ── 1. 读取视频元数据（不加载帧到内存）────────────────────────
        sm.update_status(session_id, "analyzing", progress=5, stage="loading_video")
        with _video_capture(video_path) as cap_meta:
            raw_fps = cap_meta.get(cv2.CAP_PROP_FPS)
            total   = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
            _vid_w  = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
            _vid_h  = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        # FPS fallback：容器/编解码器有时不报 fps（返回 0 或异常值），用 ffprobe 兜底
        if not raw_fps or raw_fps <= 1 or raw_fps > 240:
            raw_fps = _probe_fps(video_path) or 25.0
        fps = float(raw_fps)
        if total <= 0:
            total = 1500
        # 把真实 fps / 尺寸写入 session，下游按需任务不再 cap.get(FPS) 重开视频
        sm.update_status(
            session_id, "analyzing", progress=5, stage="loading_video",
            video_fps=fps, video_width=_vid_w, video_height=_vid_h,
        )
        print(f"[INFO] Video: {total} frames @ {fps:.1f}fps, {_vid_w}x{_vid_h}")

        # SAMURAI bboxes only become a hard dependency at _compute_player_summary
        # (end of pipeline). To allow run_samurai_tracking_multi to run in
        # parallel with this analysis, we defer the load until step 8 below.
        # If samurai_pkl isn't ready by then, we poll briefly and continue.
        tracked_bboxes: dict = {}
        if samurai_pkl and Path(samurai_pkl).exists():
            with open(samurai_pkl, "rb") as f:
                samurai_data = pickle.load(f)
            tracked_bboxes = samurai_data["bboxes"]
            tb_keys = sorted(tracked_bboxes.keys())
            print(f"[INFO] SAMURAI bboxes (preloaded): {len(tracked_bboxes)} frames, "
                  f"range [{tb_keys[0] if tb_keys else 'N/A'} – "
                  f"{tb_keys[-1] if tb_keys else 'N/A'}]")
        else:
            print(f"[INFO] SAMURAI cache not ready yet — will load before summary")

        # ── 2-4. 检测 + 光流 + 关键点（一律流式，避免 18 GB 全帧 read_video()）────
        # 之前 ≤3000 帧走 read_video() 一次性加载所有帧，3000×1080p×3 ≈ 18 GB RAM，
        # 在 RunPod 上必爆 OOM。流式路径单 chunk = 500 帧 × 6MB ≈ 3GB，长短视频
        # 都用同一条路径，行为一致、不会 OOM。
        sm.update_status(session_id, "analyzing", progress=10, stage="yolo_detection")
        yolo_path = get_yolo_model_path()
        kpt_path  = get_keypoint_model_path()
        _require_file(yolo_path, "YOLO model")
        _require_file(kpt_path, "Keypoint model")
        if not HAS_SPORTS:
            raise ImportError("sports library is required for keypoint and minimap analysis")
        tracker = Tracker(yolo_path)

        team_init_start = max(0, int(total * 0.05))
        team_init_end = min(total - 1, int(total * 0.95))
        team_init_indices = (
            np.linspace(team_init_start, team_init_end, 8, dtype=int).tolist()
            if total > 0 else []
        )
        team_vote_step = max(1, total // 20)
        team_vote_indices = list(range(0, total, team_vote_step))
        team_sample_indices = sorted(set(team_init_indices + team_vote_indices))
        team_sample_frames = {}

        print(f"[INFO] Streaming pipeline ({total} frames) — RAM-bounded mode")
        _t = _time.perf_counter()

        # 进度回调：合并 pass 占 10%→55%（45 个百分点，因为现在涵盖 YOLO+光流+关键点）
        _merge_p0, _merge_range = 10, 45

        def _merge_progress(ratio, frames_done, frames_total, eta_sec):
            pct = int(_merge_p0 + ratio * _merge_range)
            eta_str = (f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"
                       if eta_sec > 0 else "?")
            sm.update_status(
                session_id, "analyzing", progress=pct,
                stage=f"streaming_analysis ({frames_done}/{frames_total} frames, ETA {eta_str})"
            )

        # ── Parallel YOLO across N segments ─────────────────────────────
        # PARALLEL_YOLO_SEGS=4 (default) → 4 subprocesses each handle 1/4
        # of the video. Each subprocess runs YOLO+KPT+optical flow, then
        # results are merged. Speeds up detection from ~10min → ~3min on a
        # 16GB+ GPU.  Set PARALLEL_YOLO_SEGS=1 to disable and fall through
        # to the merged-streaming or legacy paths.
        #
        # 24GB GPU GPU contention guard：
        #   - 单期视频 (1 period × 4 players) → 4 SAMURAI 段，跟 4 YOLO 段同时 = 8 进程，~18GB ✅
        #   - 多期视频 (2+ periods × 4 players) → 8 SAMURAI 段，加 4 YOLO 段 = 12 进程，~24-28GB ❌ 可能 OOM
        #
        # 所以当 SAMURAI 在并发运行时 (_samurai_done_event 在 session 里) 把 YOLO 限到 3 段：
        #   3 YOLO (~6GB) + 8 SAMURAI (~12-20GB) = 18-26GB，绝大多数 24GB 卡撑得住。
        # 没 SAMURAI 并发时（比如 re-analyze 已有 tracks.pkl）保持 4 段最快。
        _samurai_concurrent = "_samurai_done_event" in session
        _default_segs = 3 if _samurai_concurrent else 4
        n_yolo_segs = int(os.environ.get("PARALLEL_YOLO_SEGS", str(_default_segs)))
        parallel_ok = False
        if n_yolo_segs > 1:
            print(f"[INFO] Parallel YOLO: {n_yolo_segs} segments")
            sm.update_status(session_id, "analyzing", progress=12,
                             stage=f"parallel_yolo ({n_yolo_segs} segments)")
            try:
                tracks, cam_mov = _run_yolo_parallel(
                    video_path, total, n_yolo_segs,
                    yolo_path, kpt_path, output_dir, fps,
                    progress_callback=_merge_progress,
                )
                # Ball interpolation across segment boundaries
                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
                # 2D position interpolation across segment boundaries
                vt_main = ViewTransformer()
                vt_main.interpolate_2d_positions(tracks)
                # Capture team-sample frames from merged tracks for team-color init
                if team_sample_indices:
                    missing = [i for i in team_sample_indices
                               if i not in team_sample_frames]
                    if missing:
                        team_sample_frames.update(
                            read_frames_at_indices(video_path, missing))
                _bench("parallel_yolo", _t)
                print(f"[INFO] Parallel YOLO done. "
                      f"Captured {len(team_sample_frames)}/{len(team_sample_indices)} "
                      "team sample frames")
                parallel_ok = True
            except Exception as par_exc:
                import sys
                print(f"[YOLO-PAR] parallel pipeline crashed: {par_exc!r}", flush=True)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                print("[YOLO-PAR] falling back to merged/legacy path", flush=True)
                team_sample_frames.clear()
                _t = _time.perf_counter()

        # Merged streaming pipeline: 1 video read instead of 3, CUDA streams
        # parallelize YOLO/keypoint, optical flow runs concurrently on CPU.
        # Set MERGED_STREAMING=0 to fall back to the legacy 3-pass path if a
        # regression turns up.
        use_merged = os.environ.get("MERGED_STREAMING", "1") != "0"

        # Merged path may fail (e.g. ultralytics+CUDA-streams compat). On any
        # exception, log the traceback so we can debug, then fall back to the
        # legacy 3-pass path so the user's analysis still completes.
        merged_ok = False
        if not parallel_ok and use_merged:
            from .analysis_core import run_merged_streaming_pipeline
            try:
                kp = KeypointDetector(kpt_path)
                cam = CameraMovementEstimator.from_video_path(video_path)

                tracks, cam_mov, kps = run_merged_streaming_pipeline(
                    video_path, total,
                    tracker=tracker, kpt_detector=kp, cam_estimator=cam,
                    progress_callback=_merge_progress,
                    sample_frame_indices=team_sample_indices,
                    sampled_frames_out=team_sample_frames,
                )
                tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
                tracker.add_position_to_tracks(tracks)
                cam.add_adjust_positions_to_tracks(tracks, cam_mov)
                vt = ViewTransformer()
                vt.add_transformed_position_to_tracks(tracks, kps)
                vt.interpolate_2d_positions(tracks)
                _bench("merged_streaming", _t)
                print(f"[INFO] Captured {len(team_sample_frames)}/{len(team_sample_indices)} "
                      "team sample frames during merged pipeline")
                merged_ok = True
            except Exception as merged_exc:
                import sys
                print(f"[MERGED] pipeline crashed: {merged_exc!r}", flush=True)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                print("[MERGED] falling back to legacy 3-pass path", flush=True)
                # Reset team-color sample dict — the merged path may have
                # partially filled it before crashing
                team_sample_frames.clear()
                _t = _time.perf_counter()

        if not parallel_ok and (not use_merged or not merged_ok):
            reason = "MERGED_STREAMING=0" if not use_merged else "merged path failed"
            print(f"[INFO] Using legacy 3-pass pipeline ({reason})")
            tracks = tracker.get_object_tracks_streamed(
                video_path, total, progress_callback=_merge_progress,
                sample_frame_indices=team_sample_indices,
                sampled_frames_out=team_sample_frames)
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            tracker.add_position_to_tracks(tracks)
            _bench("yolo_detection", _t)
            print(f"[INFO] Captured {len(team_sample_frames)}/{len(team_sample_indices)} "
                  "team sample frames during YOLO")
            _check_memory_and_gc()

            sm.update_status(session_id, "analyzing", progress=35, stage="camera_motion")
            _t = _time.perf_counter()
            cam     = CameraMovementEstimator.from_video_path(video_path)
            cam_mov = cam.get_camera_movement_streamed(video_path, total)
            cam.add_adjust_positions_to_tracks(tracks, cam_mov)
            _bench("camera_motion", _t)
            _check_memory_and_gc()

            sm.update_status(session_id, "analyzing", progress=50, stage="keypoint_detection")
            _t = _time.perf_counter()
            kp  = KeypointDetector(kpt_path)
            vt  = ViewTransformer()
            kps = kp.predict_streamed(video_path, total, cam_movement=cam_mov)
            vt.add_transformed_position_to_tracks(tracks, kps)
            vt.interpolate_2d_positions(tracks)
            _bench("keypoint_detection", _t)
        _check_memory_and_gc()

        # ── 5-7. 三路并行：速度计算 / 队伍颜色+投票 / 场景分割 ──────────
        # YOLO 结束后 GPU 已释放（SAMURAI 仍在后台跑），三个任务互不依赖：
        #   Thread A — 速度/距离（CPU，读 position_transformed，写 speed/distance）
        #   Thread B — 队伍颜色初始化 + 多帧投票（CPU，读 bbox+sample_frames）
        #   Thread C — 场景分割（CPU，读 tracks player 数 或 user_periods）
        # 三路 join 后再做球权检测（需要 B 的 player_final_team），避免写冲突。

        _speed_exc:  list = [None]
        _team_exc:   list = [None]
        _scene_exc:  list = [None]
        _team_result: dict = {}   # team_assigner, team_color_initialized, player_final_team
        _scene_result: dict = {}  # segments

        # 进度条单调推进：主线程在 fan-out 前后各设一次 progress，
        # 子线程只更新 stage（文本标签），避免多线程争抢导致进度倒退。
        sm.update_status(session_id, "analyzing", progress=70,
                         stage="parallel_post_yolo (speed + team + scene)")

        def _capture_exc(slot: list):
            """Print traceback in the crashing thread, then store exc for main."""
            slot[0] = sys.exc_info()[1]
            traceback.print_exc()

        # ── Thread A: 速度 & 距离 ─────────────────────────────────────
        def _thread_speed():
            try:
                sm.update_status(session_id, "analyzing",
                                 stage="speed_calculation")
                _t = _time.perf_counter()
                speed_est = AccurateSpeedEstimator(fps=fps)
                speed_est.add_speed_and_distance_to_tracks(tracks)
                _bench("speed_calculation", _t)
            except Exception:
                _capture_exc(_speed_exc)

        # ── Thread B: 队伍颜色初始化 + 多帧投票 ──────────────────────
        def _thread_team():
            try:
                from collections import Counter as _Counter
                sm.update_status(session_id, "analyzing",
                                 stage="team_assignment")
                _ta = TeamAssigner()

                sm.update_status(session_id, "analyzing",
                                 stage="team_color_init")
                _t = _time.perf_counter()
                _init_colors = _ta.assign_team_color_from_frame_dict(
                    team_sample_frames, tracks["players"], team_init_indices)
                if _ta.kmeans is None:
                    print("[WARN] Missing streamed team samples; falling back to video seek")
                    _ta.assign_team_color_from_video(video_path, tracks["players"], n_samples=8)
                _tci = bool(_ta.kmeans is not None)
                _bench("team_color_init", _t)
                if _tci:
                    print(f"[INFO] Team colors initialized from {_init_colors} color samples")
                else:
                    print("[WARN] Team color initialization failed — all players assigned to team 1")

                _pfd: dict = {}  # player_final_team
                if _tci:
                    sm.update_status(session_id, "analyzing",
                                     stage="team_voting")
                    _t = _time.perf_counter()
                    _t_read_total = 0.0
                    _t_color_total = 0.0
                    _vote_samples = 0
                    _vote_color_ops = 0
                    _player_vote_dict: dict = {}

                    _missing = [idx for idx in team_vote_indices
                                if idx not in team_sample_frames]
                    if _missing:
                        print(f"[WARN] Missing {len(_missing)} vote sample frames; "
                              "falling back to video seek for those")
                        _tr = _time.perf_counter()
                        team_sample_frames.update(
                            read_frames_at_indices(video_path, _missing))
                        _t_read_total += _time.perf_counter() - _tr

                    for _idx in team_vote_indices:
                        _frame = team_sample_frames.get(_idx)
                        if _frame is None or _idx >= len(tracks["players"]):
                            continue
                        _vote_samples += 1
                        for _pid, _info in _ta._iter_color_sample_players(
                                tracks["players"][_idx]):
                            try:
                                _tc2 = _time.perf_counter()
                                _color = _ta._get_player_color(_frame, _info['bbox'])
                                _t_color_total += _time.perf_counter() - _tc2
                                if _color is not None and not np.all(_color == 0):
                                    _cid = int(_ta.kmeans.predict(_color.reshape(1, -1))[0])
                                    _pred = _ta._cluster_to_team.get(_cid)
                                    if _pred is None:
                                        _d1 = np.linalg.norm(
                                            _color - _ta.team_colors.get(1, np.zeros(3)))
                                        _d2 = np.linalg.norm(
                                            _color - _ta.team_colors.get(2, np.zeros(3)))
                                        _pred = 1 if _d1 <= _d2 else 2
                                    _player_vote_dict.setdefault(_pid, []).append(_pred)
                                    _vote_color_ops += 1
                            except Exception:
                                pass

                    _pfd = {
                        _pid: _Counter(_votes).most_common(1)[0][0]
                        for _pid, _votes in _player_vote_dict.items() if _votes
                    }
                    _bench("team_voting", _t)
                    print(
                        f"[INFO] Multi-frame voting done: {len(_pfd)} players assigned "
                        f"from {_vote_samples} sampled frames / {_vote_color_ops} color ops "
                        f"(read={_t_read_total:.1f}s, color={_t_color_total:.1f}s)"
                    )

                _team_result.update({
                    "team_assigner":         _ta,
                    "team_color_initialized": _tci,
                    "player_final_team":      _pfd,
                })
            except Exception:
                _capture_exc(_team_exc)

        # ── Thread C: 场景分割 ────────────────────────────────────────
        def _thread_scene():
            try:
                sm.update_status(session_id, "analyzing",
                                 stage="scene_segmentation")
                _t = _time.perf_counter()
                _user_p = session.get("match_periods_frames")

                def _period_segment(seg_type: str, start: int, end: int) -> dict:
                    return {
                        "type": seg_type,
                        "start_frame": start,
                        "end_frame": end,
                        "start_sec": round(start / max(fps, 1.0), 2),
                        "end_sec": round(end / max(fps, 1.0), 2),
                        "duration_sec": round((end - start) / max(fps, 1.0), 2),
                    }

                if _user_p and len(_user_p) > 0:
                    _segs = []
                    _cursor = 0
                    for _i, (_ps, _pe) in enumerate(_user_p):
                        if _ps > _cursor:
                            _segs.append(_period_segment(
                                "halftime" if _i > 0 else "pre_match", _cursor, _ps))
                        _stype = ("first_half" if _i == 0 and len(_user_p) > 1
                                  else "second_half" if _i == 1 and len(_user_p) > 1
                                  else "match")
                        _segs.append(_period_segment(_stype, _ps, _pe))
                        _cursor = _pe
                    if _cursor < total:
                        _segs.append(_period_segment("post_match", _cursor, total))
                    print(f"[INFO] Scene segments (from user periods): "
                          f"{[s['type'] for s in _segs]}")
                else:
                    _scene_det = SceneChangeDetector(fps=fps)
                    _segs = _scene_det.detect_segments(tracks, total)
                    print(f"[INFO] Scene segments (auto-detected): "
                          f"{[s['type'] for s in _segs]} "
                          f"(durations: {[round(s['duration_sec']) for s in _segs]}s)")

                _scene_result["segments"] = _segs
                _bench("scene_segmentation", _t)
            except Exception:
                _capture_exc(_scene_exc)

        # ── 启动三路并行，等待全部完成 ────────────────────────────────
        _t_parallel = _time.perf_counter()
        _th_speed = threading.Thread(target=_thread_speed, daemon=True, name="speed")
        _th_team  = threading.Thread(target=_thread_team,  daemon=True, name="team")
        _th_scene = threading.Thread(target=_thread_scene, daemon=True, name="scene")
        _th_speed.start(); _th_team.start(); _th_scene.start()
        _th_speed.join();  _th_team.join();  _th_scene.join()
        _bench("parallel_post_yolo", _t_parallel)

        # 单调推进：所有并行阶段都完成后再推到 87
        sm.update_status(session_id, "analyzing", progress=87,
                         stage="parallel_post_yolo_done")

        # 传播线程异常（traceback 已在子线程内 print，这里只需要重新 raise）
        for _exc in (_speed_exc[0], _team_exc[0], _scene_exc[0]):
            if _exc is not None:
                raise _exc

        # 解包结果
        team_assigner          = _team_result["team_assigner"]
        team_color_initialized = _team_result["team_color_initialized"]
        player_final_team      = _team_result["player_final_team"]
        segments               = _scene_result["segments"]

        # ── 8. 球权检测（串行，写入 tracks；需要 B 的 player_final_team）──
        team_control  = []
        poss_detector = SmartBallPossessionDetector(fps=fps, video_w=_vid_w, video_h=_vid_h)
        ball_history  = []

        if team_color_initialized:
            sm.update_status(session_id, "analyzing", progress=88,
                             stage="possession_detection")
            _t = _time.perf_counter()
            for i, p_tracks in enumerate(tracks["players"]):
                for pid, info in p_tracks.items():
                    if not info:
                        continue
                    tid = player_final_team.get(pid, 1)
                    info["team"]       = tid
                    info["team_color"] = team_assigner.team_colors.get(tid, np.array([0,0,0]))

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
                    p_tracks[pid_has_ball]["has_ball"]              = True
                    p_tracks[pid_has_ball]["possession_confidence"] = conf
                    team_control.append(p_tracks[pid_has_ball].get("team", 0))
                else:
                    team_control.append(0)

            _bench("possession_detection", _t)

        # ── 9. 摘要 & 缓存 ────────────────────────────────────────────
        _bench("TOTAL", _t_total)
        try:
            log_path = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark.log")
            with open(os.path.abspath(log_path), "a") as _f:
                _f.write("-" * 40 + "\n")
        except Exception:
            pass

        # ── Wait for SAMURAI here if it was running concurrently. ─────────
        # Timeout scales with video length: SAMURAI walltime is roughly
        # 0.5× video duration for the multi-segment-parallel path (4 chunks
        # share the GPU). We use 2× video duration as the cap, floored at
        # 10 min so a 30s clip still gets a sane cushion. This prevents
        # the 10-min hardcoded timeout from firing on 1h+ videos.
        video_duration_sec = total / max(fps, 1.0)
        samurai_timeout_sec = max(600.0, 2.0 * video_duration_sec)

        if not tracked_bboxes:
            samurai_event = session.get("_samurai_done_event")
            samurai_pkl = (sm.get_session(session_id) or {}).get("samurai_cache_path")
            sm.update_status(session_id, "analyzing", progress=91,
                             stage="waiting_for_samurai")
            print(f"[ANALYSIS] waiting for SAMURAI (timeout = "
                  f"{samurai_timeout_sec/60:.1f} min for a "
                  f"{video_duration_sec/60:.1f} min clip)")

            if samurai_event is not None:
                # Path 1: Event-based — block here until SAMURAI thread sets it.
                got = samurai_event.wait(timeout=samurai_timeout_sec)
                if not got:
                    print(f"[WARN] SAMURAI event not set after "
                          f"{samurai_timeout_sec/60:.1f} min — "
                          f"continuing without tracked_bboxes")
            elif samurai_pkl:
                # Path 2: cross-worker fallback — file-existence poll.
                wait_deadline = _time.perf_counter() + samurai_timeout_sec
                while not Path(samurai_pkl).exists():
                    if _time.perf_counter() > wait_deadline:
                        print(f"[WARN] SAMURAI pkl not present after "
                              f"{samurai_timeout_sec/60:.1f} min — "
                              f"continuing without tracked_bboxes")
                        break
                    _time.sleep(5.0)

            # Either path: try to load the bboxes if the file is now there
            if samurai_pkl and Path(samurai_pkl).exists():
                with open(samurai_pkl, "rb") as f:
                    samurai_data = pickle.load(f)
                tracked_bboxes = samurai_data.get("bboxes", {})
                print(f"[INFO] SAMURAI bboxes loaded (post-analysis): "
                      f"{len(tracked_bboxes)} frames")

        sm.update_status(session_id, "analyzing", progress=92, stage="computing_summary")
        print(f"[INFO] Computing summary for session {session_id}…")
        _t = _time.perf_counter()
        player_summary = _compute_player_summary(
            tracks, tracked_bboxes, team_control, fps=fps, segments=segments)
        _bench("compute_summary", _t)

        cache_payload = {
            "tracks":              tracks,
            "tracked_bboxes":      tracked_bboxes,
            "team_control":        team_control,
            "possession_switches": player_summary.get("possession_switches", 0),
            "segments":            segments,
            # 颜色序列化（numpy array → list）
            "team_colors": {
                k: v.tolist() for k, v in team_assigner.team_colors.items()
            },
            "team_colors_hex": {
                k: bgr_to_hex(v) for k, v in team_assigner.team_colors.items()
            },
        }
        print(f"[INFO] Persisting analysis cache for session {session_id}…")
        _t = _time.perf_counter()
        _atomic_pickle_dump(cache_payload, tracks_cache)
        _bench("persist_tracks_cache", _t)

        # Upload tracks.pkl to R2 so feature tasks on other workers can access it
        try:
            from ..storage.r2 import upload_to_r2
            r2_url = upload_to_r2(tracks_cache, f"{session_id}/tracks.pkl")
            if r2_url:
                print(f"[INFO] tracks.pkl uploaded to R2 for session {session_id}")
        except Exception as r2_exc:
            print(f"[WARN] tracks.pkl R2 upload failed (non-fatal): {r2_exc}")

        # ── Export compact JSON for frontend canvas minimap + heatmap ────
        # Lets the frontend draw the minimap as an overlay on the main video
        # (perfectly synced via currentTime) instead of rendering a separate
        # MP4 server-side. ~300-700KB per session, well worth the small CPU.
        minimap_url, heatmap_url = None, None
        try:
            # Pass match_periods so the JSON embeds them — the frontend uses
            # them to map rendered-video currentTime back to original-frame
            # indices (the mp4 has halftime stripped so timecodes diverge).
            _user_periods = session.get("match_periods_frames")
            _periods_for_export = (
                [tuple(p) for p in _user_periods] if _user_periods else None
            )
            minimap_url, heatmap_url = _export_position_jsons(
                session_id, tracks, tracked_bboxes, team_control,
                team_assigner.team_colors, fps, total, output_dir,
                match_periods=_periods_for_export,
            )
        except Exception as exc:
            print(f"[WARN] position JSON export failed (non-fatal): {exc}")

        sm.update_status(
            session_id, "analysis_done",
            progress=100, stage="done",
            tracks_cache_path=str(tracks_cache),
            player_summary=player_summary,
            total_frames=total,
            segments=segments,
            minimap_data_url=minimap_url,
            heatmap_data_url=heatmap_url,
        )

    except Exception as exc:
        # 失败时删除可能半成品的 tracks.pkl，避免下游按需任务读到损坏数据
        try:
            if tracks_cache.exists():
                tracks_cache.unlink()
        except Exception:
            pass
        sm.update_status(session_id, "analysis_failed",
                         error=str(exc), stage="analysis_error",
                         tracks_cache_path=None)
        _log_error("Global analysis", session_id, exc)


def _summary_for_range(tracks: dict, tracked_bboxes: dict, team_control: list,
                        start: int, end: int, fps: int) -> dict:
    """对 [start, end) 帧区间计算一份 summary（不含全局字段）"""
    speeds, distances, has_ball_count = [], [], 0

    end = min(end, len(tracks["players"]))
    for i in range(start, end):
        if i not in tracked_bboxes:
            continue
        sx, sy, sw, sh = tracked_bboxes[i]
        matched = _find_matched_player(tracks["players"][i], (sx+sw/2, sy+sh/2))
        if matched:
            speeds.append(matched.get("speed", 0))
            distances.append(matched.get("distance", 0))
            if matched.get("has_ball"):
                has_ball_count += 1

    sub_ctrl = team_control[start:min(end, len(team_control))]
    arr = np.array(sub_ctrl) if sub_ctrl else np.array([])
    t1 = int(np.sum(arr == 1)) if arr.size else 0
    t2 = int(np.sum(arr == 2)) if arr.size else 0
    total_ctrl = t1 + t2 or 1

    possession_switches = 0
    prev_team = 0
    for t in sub_ctrl:
        if t != 0 and t != prev_team and prev_team != 0:
            possession_switches += 1
        if t != 0:
            prev_team = t

    # distance 是累计值，用区间首末的差 更准（而非 max）
    dist_delta = 0.0
    if distances:
        dist_delta = float(max(distances) - min(distances))

    return {
        "max_speed_kmh":        round(float(max(speeds)),        1) if speeds else 0,
        "avg_speed_kmh":        round(float(np.mean(speeds)),    1) if speeds else 0,
        "total_distance_m":     round(dist_delta,                0),
        "possession_seconds":   round(has_ball_count / fps,      1),
        "team1_possession_pct": round(t1 / total_ctrl * 100,     1),
        "team2_possession_pct": round(t2 / total_ctrl * 100,     1),
        "possession_switches":  possession_switches,
    }


def _compute_player_summary(tracks: dict, tracked_bboxes: dict,
                            team_control: list, fps: int = 24,
                            segments: list = None) -> dict:
    """从 tracks 中提取被追踪球员的关键数字（存入 session，立即可用）

    若提供 segments，overall 只统计实际比赛帧（排除 halftime），
    并额外返回 by_segment 字段（上下半场分别统计）。
    """
    total_frames = len(tracks["players"])

    # ── Overall：若有中场则只统计比赛帧，避免中场零速度稀释均值 ──
    has_halftime = segments and any(s["type"] == "halftime" for s in segments)
    if has_halftime:
        match_ranges = [(s["start_frame"], s["end_frame"])
                        for s in segments if s["type"] != "halftime"]
    else:
        match_ranges = [(0, total_frames)]

    speeds, distances, has_ball_count = [], [], 0
    all_sub_ctrl = []
    for rng_start, rng_end in match_ranges:
        rng_end = min(rng_end, total_frames)
        for i in range(rng_start, rng_end):
            if i not in tracked_bboxes:
                continue
            sx, sy, sw, sh = tracked_bboxes[i]
            matched = _find_matched_player(tracks["players"][i], (sx + sw / 2, sy + sh / 2))
            if matched:
                speeds.append(matched.get("speed", 0))
                distances.append(matched.get("distance", 0))
                if matched.get("has_ball"):
                    has_ball_count += 1
        sub = team_control[rng_start:min(rng_end, len(team_control))]
        all_sub_ctrl.extend(sub)

    arr = np.array(all_sub_ctrl) if all_sub_ctrl else np.array([])
    t1  = int(np.sum(arr == 1)) if arr.size else 0
    t2  = int(np.sum(arr == 2)) if arr.size else 0
    total_ctrl = t1 + t2 or 1

    possession_switches = 0
    prev_team = 0
    for t in all_sub_ctrl:
        if t != 0 and t != prev_team and prev_team != 0:
            possession_switches += 1
        if t != 0:
            prev_team = t

    dist_delta = float(max(distances) - min(distances)) if distances else 0.0

    overall = {
        "max_speed_kmh":        round(float(max(speeds)),        1) if speeds else 0,
        "avg_speed_kmh":        round(float(np.mean(speeds)),    1) if speeds else 0,
        "total_distance_m":     round(dist_delta,                0),
        "possession_seconds":   round(has_ball_count / fps,      1),
        "team1_possession_pct": round(t1 / total_ctrl * 100,     1),
        "team2_possession_pct": round(t2 / total_ctrl * 100,     1),
        "possession_switches":  possession_switches,
    }

    # ── 分段统计（跳过 halftime）──────────────────────────────────
    if segments:
        by_segment = []
        for seg in segments:
            if seg["type"] == "halftime":
                continue
            seg_stats = _summary_for_range(
                tracks, tracked_bboxes, team_control,
                seg["start_frame"], seg["end_frame"], fps)
            seg_stats["segment_type"] = seg["type"]
            seg_stats["start_sec"]    = seg["start_sec"]
            seg_stats["end_sec"]      = seg["end_sec"]
            by_segment.append(seg_stats)
        overall["by_segment"] = by_segment

    return overall


# ═══════════════════════════════════════════════════════════════════════
# 阶段 3：按需生成任务
# ═══════════════════════════════════════════════════════════════════════

# ── 3a. 热力图 ───────────────────────────────────────────────────────────────

def _export_position_jsons(session_id: str, tracks: dict, tracked_bboxes: dict,
                            team_control: list, team_colors: dict, fps: float,
                            total: int, output_dir: Path,
                            match_periods: list | None = None) -> tuple[str | None, str | None]:
    """
    Export two compact JSON files for the frontend canvas overlays:

      1. minimap_positions.json — pitch coords of all players + ball,
         sampled at MINIMAP_SAMPLE_FPS (≈ every Nth video frame). The
         frontend interpolates / nearest-frame-picks between samples.
         For a 90-min match this caps the JSON at ~2 MB instead of 15+
         MB if we kept every frame.

      2. heatmap_positions.json — sampled flat list of (x, y) for the
         tracked player. Same downsampling — 1 Hz is plenty for a KDE.

    Both upload to R2 and return their public URLs (or None on failure).
    """
    import json

    pitch_length = 12000  # cm — matches SoccerPitchConfiguration default
    pitch_width  = 7000

    # Sample at ~5 Hz for the minimap (smooth enough for canvas overlay),
    # ~1 Hz for the heatmap (KDE doesn't need sub-second precision).
    # These caps keep JSON size linear in video duration, not in frame
    # count, so a 4K-60fps 1-hour clip is no worse than a 720p-25fps one.
    MINIMAP_SAMPLE_HZ = 5.0
    HEATMAP_SAMPLE_HZ = 1.0
    minimap_stride = max(1, int(round(fps / MINIMAP_SAMPLE_HZ)))
    heatmap_stride = max(1, int(round(fps / HEATMAP_SAMPLE_HZ)))

    minimap_frames: list[list[dict]] = []
    ball_path: list = []
    minimap_sampled_indices: list[int] = []   # original frame index for each sample
    tracked_positions: list[list[float]] = []

    for i in range(total):
        # Skip frames that don't land on our sample boundaries — the
        # frontend interpolates between the kept ones.
        emit_minimap = (i % minimap_stride == 0) or (i == total - 1)
        emit_heatmap = (i % heatmap_stride == 0)
        if not (emit_minimap or emit_heatmap):
            continue

        # Players for this frame
        players_this_frame = []
        frame_players = tracks["players"][i] if i < len(tracks["players"]) else {}
        for pid, info in (frame_players or {}).items():
            mm_pos = info.get("position_minimap")
            tr_pos = info.get("position_transformed")
            if mm_pos is not None:
                # position_minimap is already in SoccerPitchConfiguration scale
                # (0-12000 × 0-7000 cm) — use directly.
                pos = mm_pos
            elif tr_pos is not None:
                # position_transformed is in METRES (0-105 × 0-68).
                # Convert to the same cm scale so toPx() on the frontend works:
                #   x_cm = x_m * 100,  y_cm = y_m * 100
                pos = [float(tr_pos[0]) * 100.0, float(tr_pos[1]) * 100.0]
            else:
                continue
            try:
                x = float(pos[0]); y = float(pos[1])
            except (TypeError, IndexError):
                continue
            team = int(info.get("team", 0)) if info.get("team") is not None else 0
            entry = {
                "id": int(pid),
                "x": round(x, 1),
                "y": round(y, 1),
                "t": team,
            }
            if info.get("has_ball"):
                entry["b"] = 1
            players_this_frame.append(entry)

        # Tracked player position (for heatmap) — only on heatmap-stride frames
        if emit_heatmap and i in tracked_bboxes:
            sx, sy, sw, sh = tracked_bboxes[i]
            matched = _find_matched_player(frame_players or {}, (sx + sw / 2, sy + sh / 2))
            if matched:
                mm = matched.get("position_minimap")
                tr = matched.get("position_transformed")
                if mm is not None:
                    tpos = [float(mm[0]), float(mm[1])]
                elif tr is not None:
                    tpos = [float(tr[0]) * 100.0, float(tr[1]) * 100.0]
                else:
                    tpos = None
                if tpos is not None:
                    try:
                        tracked_positions.append([round(tpos[0], 1), round(tpos[1], 1)])
                    except (TypeError, IndexError):
                        pass

        # Per-minimap-sample bookkeeping
        if emit_minimap:
            # Mark ONLY the tracked player so the frontend can highlight it.
            # BUG (fixed): the old check `p["id"] in (frame_players or {})`
            # was true for EVERY player (all ids are keys of frame_players),
            # so all dots got tr=1 and the frontend's "if p.tr → continue"
            # loop skipped all of them, leaving only the last one visible.
            if i in tracked_bboxes:
                sx, sy, sw, sh = tracked_bboxes[i]
                matched = _find_matched_player(
                    frame_players or {}, (sx + sw / 2, sy + sh / 2)
                )
                if matched:
                    # Identify the pid whose info object is the matched one
                    matched_pid = None
                    for pid, info in (frame_players or {}).items():
                        if info is matched:
                            matched_pid = int(pid)
                            break
                    if matched_pid is not None:
                        for p in players_this_frame:
                            if p["id"] == matched_pid:
                                p["tr"] = 1
                                break

            minimap_frames.append(players_this_frame)
            minimap_sampled_indices.append(i)

            # Ball — same key-lookup order as players, same unit normalisation
            bx_data = None
            for _, b in (tracks["ball"][i] if i < len(tracks["ball"]) else {}).items():
                if not b:
                    continue
                b_mm = b.get("position_minimap")
                b_tr = b.get("position_transformed")
                if b_mm is not None:
                    bpos = [float(b_mm[0]), float(b_mm[1])]
                elif b_tr is not None:
                    bpos = [float(b_tr[0]) * 100.0, float(b_tr[1]) * 100.0]
                else:
                    continue
                try:
                    bx_data = [round(bpos[0], 1), round(bpos[1], 1)]
                except (TypeError, IndexError):
                    pass
                break
            ball_path.append(bx_data)

    team_colors_hex = {
        str(k): bgr_to_hex(v) for k, v in team_colors.items()
    } if team_colors else {}

    # Downsample team_control to match the minimap sample rate. The full
    # array is one int per frame — for a 3h match that's 270K ints, ~800KB
    # of JSON the frontend never actually indexes per-frame anyway.
    team_control_sampled = [
        int((team_control or [0] * (i + 1))[i]) if i < len(team_control or [])
        else 0
        for i in minimap_sampled_indices
    ] if team_control else []

    minimap_payload = {
        "pitch":         {"length": pitch_length, "width": pitch_width, "unit": "cm"},
        "fps":           round(float(fps), 3),
        "total_frames":  int(total),
        # Frames are sampled — frontend should pick nearest-by-time, not
        # treat array index as frame index.
        "sample_stride": minimap_stride,
        "sample_indices": minimap_sampled_indices,
        # match_periods enables rendered-mp4-currentTime → original-frame
        # mapping on the frontend. Without it, after halftime is stripped
        # from the mp4 the minimap would lag the video by the halftime
        # length. Format: [[start_frame, end_frame], ...] in ORIGINAL frames.
        "match_periods": [list(p) for p in (match_periods or [])],
        "team_colors":   team_colors_hex,
        "frames":        minimap_frames,
        "ball":          ball_path,
        # team_control is now aligned 1:1 with `frames` / `sample_indices`,
        # not with the original frame range.
        "team_control":  team_control_sampled,
    }

    heatmap_payload = {
        "pitch":     {"length": pitch_length, "width": pitch_width, "unit": "cm"},
        "positions": tracked_positions,
    }

    # Write locally first
    mm_local = output_dir / "minimap_positions.json"
    hm_local = output_dir / "heatmap_positions.json"
    mm_local.write_text(json.dumps(minimap_payload, separators=(",", ":")))
    hm_local.write_text(json.dumps(heatmap_payload, separators=(",", ":")))

    mm_size = mm_local.stat().st_size / 1024
    hm_size = hm_local.stat().st_size / 1024
    print(f"[INFO] Exported minimap_positions.json ({mm_size:.0f} KB), "
          f"heatmap_positions.json ({hm_size:.0f} KB)")

    # Upload to R2
    mm_url = hm_url = None
    try:
        from ..storage.r2 import upload_to_r2
        mm_url = upload_to_r2(mm_local, f"{session_id}/minimap_positions.json")
        hm_url = upload_to_r2(hm_local, f"{session_id}/heatmap_positions.json")
    except Exception as exc:
        print(f"[WARN] position JSON R2 upload failed: {exc}")

    return mm_url, hm_url


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
        fps = float(session.get("video_fps") or 25.0)

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
        fps = float(session.get("video_fps") or 25.0)
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

        fps = float(session.get("video_fps") or 25.0)

        # 预渲染球场底图（只算一次，后续每帧 copy — 核心加速点）
        pitch_bg = make_pitch_background()
        mh, mw  = pitch_bg.shape[:2]

        # 第一帧
        # 轨迹坐标用 position_transformed（米制，匹配 _mm_p2px）。
        # position_minimap 是遗留 config-scale（~0-10500），画到 840px 画布外。
        ball_trail: list = []
        ball_info_0 = tracks["ball"][0].get(1, {}) if len(tracks["ball"]) > 0 else {}
        ball_mp_0 = ball_info_0.get("position_transformed") if ball_info_0 else None
        if ball_mp_0:
            ball_trail.append((ball_mp_0[0], ball_mp_0[1]))
        first_frame = render_minimap_frame(0, tracks, tracked_bboxes, team_control,
                                            config, hex_t1, hex_t2,
                                            ball_trail=list(ball_trail),
                                            pitch_bg=pitch_bg, fps=fps)

        output_path = sm.session_output_dir(session_id) / "minimap_replay.mp4"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{mw}x{mh}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "pipe:",
            *_h264_encode_args(),
            # +faststart: moov atom 搬到文件头 → 浏览器 <video> 秒开，
            # 不用等整个文件下载完才能寻址。对 tunnel 场景尤其重要。
            "-movflags", "+faststart",
            str(output_path)
        ]
        # with 语法：确保异常时也 wait()+kill 子进程（避免 zombie ffmpeg）
        # stderr 保留到临时文件，编码失败时可回读诊断（DEVNULL 会吞掉 libx264 报错）
        ff_log = sm.session_output_dir(session_id) / "minimap_ffmpeg.log"
        with open(ff_log, "wb") as _errf, \
             sp.Popen(ffmpeg_cmd, stdin=sp.PIPE,
                      stdout=sp.DEVNULL, stderr=_errf) as proc:
            try:
                # 写第一帧
                proc.stdin.write(first_frame.tobytes())

                # 流式渲染后续帧
                for i in range(1, total_frames):
                    ball_info = tracks["ball"][i].get(1, {}) if i < len(tracks["ball"]) else {}
                    ball_mp = ball_info.get("position_transformed") if ball_info else None
                    if ball_mp:
                        ball_trail.append((ball_mp[0], ball_mp[1]))
                        if len(ball_trail) > 30:
                            ball_trail.pop(0)
                    try:
                        frame = render_minimap_frame(i, tracks, tracked_bboxes, team_control,
                                                      config, hex_t1, hex_t2,
                                                      ball_trail=list(ball_trail),
                                                      pitch_bg=pitch_bg, fps=fps)
                    except Exception:
                        frame = np.zeros((mh, mw, 3), dtype=np.uint8)
                    proc.stdin.write(frame.tobytes())

                    if i % 120 == 0:
                        pct = int(10 + (i / total_frames) * 85)
                        sm.update_task(session_id, task_id, progress=pct)
            finally:
                if proc.stdin:
                    try: proc.stdin.close()
                    except Exception: pass

        # 校验编码结果：returncode 非零或文件 <1KB 说明 libx264 静默失败
        # (常见原因：pitch_bg 奇数尺寸被 yuv420p 拒绝，或 Colab 磁盘满)
        if proc.returncode != 0 or (not output_path.exists()) or output_path.stat().st_size < 1024:
            tail = ""
            try:
                with open(ff_log, "rb") as _rf:
                    tail = _rf.read()[-2000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"FFmpeg minimap encode failed (rc={proc.returncode}, "
                f"size={output_path.stat().st_size if output_path.exists() else 0}B). "
                f"stderr tail:\n{tail}"
            )

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

# Replay rendering tunables. Adjust if RunPod gives you a different CPU count.
REPLAY_WORKERS = min(4, max(1, (os.cpu_count() or 4)))
# Downscale anything taller than this before encoding (Plan C). 0 disables.
REPLAY_MAX_HEIGHT = 1080
# Below this frame count we skip the multi-process overhead and use the
# old single-pass path (faster for short clips).
REPLAY_PARALLEL_THRESHOLD = 240


# Plan F: GPU-accelerated h264 encoding via NVENC. Detected once at module
# import. If the ffmpeg binary wasn't built with nvenc, or there's no NVIDIA
# GPU visible, we silently fall back to libx264 (CPU).
def _probe_nvenc() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
        )
        return b"h264_nvenc" in (r.stdout or b"")
    except Exception:
        return False


_NVENC_AVAILABLE = _probe_nvenc()
print(f"[ENCODER] NVENC available: {_NVENC_AVAILABLE} "
      f"({'h264_nvenc (GPU)' if _NVENC_AVAILABLE else 'libx264 (CPU fallback)'})",
      flush=True)


def _h264_encode_args() -> list[str]:
    """ffmpeg encoder flags. GPU (NVENC) if available, else CPU (libx264)."""
    if _NVENC_AVAILABLE:
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",   # p1..p7 = fastest..slowest; p4 = balanced
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "23",       # constant quality target (≈ libx264 crf 23)
            "-b:v", "0",       # let -cq drive bitrate
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
    ]


print(f"[REPLAY] h264 encoder: "
      f"{'h264_nvenc (GPU)' if _NVENC_AVAILABLE else 'libx264 (CPU)'}")


def _render_chunk_to_mp4(args):
    """
    Render a contiguous frame range to its own temp mp4. Runs in a worker
    process so several chunks can be encoded in parallel.

    All arguments are pickle-safe (paths + plain dicts/strings/numbers).
    The worker reloads tracks.pkl itself to avoid serialising hundreds of MB
    across the process boundary on every call.
    """
    import subprocess as sp_local
    import pickle as pkl_local

    # match_periods is a list of (start_frame, end_frame) tuples — frames
    # falling outside ANY period are dropped from the rendered mp4 (e.g.
    # the user's chosen halftime gap). None means "render every frame".
    (start, end, video_path, tracks_cache_path,
     hex_t1, hex_t2, fps, w, h,
     yolo_path, output_path, max_height, match_periods) = args

    # Pre-compute a fast in-period check
    if match_periods:
        # Sorted, non-overlapping — clamp once to the chunk's range
        period_set = [(max(ps, start), min(pe, end))
                       for ps, pe in match_periods
                       if pe > start and ps < end]
        def _in_period(i: int) -> bool:
            for ps, pe in period_set:
                if ps <= i < pe:
                    return True
            return False
    else:
        def _in_period(_: int) -> bool:
            return True

    # Reload analysis cache in this worker (workers run in parallel, so the
    # 5-10s of pickle-load happens concurrently across all of them).
    with open(tracks_cache_path, "rb") as f:
        data = pkl_local.load(f)
    tracks = data["tracks"]
    tracked_bboxes = data["tracked_bboxes"]
    team_control = data["team_control"]

    tracker_obj = Tracker(yolo_path)
    team_assigner = TeamAssigner()
    team_assigner.team_colors = {
        1: np.array([int(hex_t1[5:7], 16), int(hex_t1[3:5], 16), int(hex_t1[1:3], 16)]),
        2: np.array([int(hex_t2[5:7], 16), int(hex_t2[3:5], 16), int(hex_t2[1:3], 16)]),
    }
    config = SoccerPitchConfiguration() if HAS_SPORTS else None

    # Plan C: downscale if input is taller than max_height. We resize AFTER
    # rendering at native res because the YOLO bboxes are stored in original
    # coordinates; rescaling them everywhere would touch every drawing call.
    # Encoding fewer pixels still helps (~20-30%).
    out_w, out_h = w, h
    if max_height and h > max_height:
        scale = max_height / h
        out_w = (int(w * scale) // 2) * 2   # libx264 needs even dims
        out_h = (int(max_height) // 2) * 2
    do_resize = (out_w, out_h) != (w, h)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:",
        *_h264_encode_args(),
        str(output_path),
    ]

    cap = cv2.VideoCapture(video_path)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    proc = sp_local.Popen(ffmpeg_cmd, stdin=sp_local.PIPE,
                           stdout=sp_local.DEVNULL, stderr=sp_local.DEVNULL)
    rendered_frames = 0
    skipped_frames = 0
    try:
        for i in range(start, end):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Skip non-match frames cheaply: still decode (cv2 doesn't expose
            # frame skipping that's faster than read), but don't render/write.
            # Output mp4 ends up containing only valid-period frames.
            if not _in_period(i):
                skipped_frames += 1
                continue
            rargs = (i, frame, tracks, tracked_bboxes, team_control,
                     team_assigner, tracker_obj, config, hex_t1, hex_t2)
            _, rendered = _render_single_frame_worker_full(rargs)
            if do_resize:
                rendered = cv2.resize(rendered, (out_w, out_h), interpolation=cv2.INTER_AREA)
            try:
                proc.stdin.write(rendered.tobytes())
                rendered_frames += 1
            except BrokenPipeError:
                break
    finally:
        cap.release()
        if proc.stdin:
            try: proc.stdin.close()
            except Exception: pass
        proc.wait()

    return {
        "start": start,
        "end": end,
        "rendered": rendered_frames,
        "path": str(output_path),
        "rc": proc.returncode,
        "out_w": out_w,
        "out_h": out_h,
    }


def run_full_replay(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    生成带有 YOLO 检测框、球员 ID、球权三角的全景视频 MP4。

    多进程并行渲染（Plan A）+ 编码前 1080p 上限降采样（Plan C）。
    用 REPLAY_WORKERS 个 worker 把帧均分成块，每块单独跑 ffmpeg → 临时 mp4，
    最后 ffmpeg concat demuxer 拼成最终文件。短视频（< REPLAY_PARALLEL_THRESHOLD
    帧）走老的单进程路径，省掉 worker 启动开销。
    """
    try:
        import math
        import subprocess as sp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        sm.update_task(session_id, task_id, status="running", progress=5)

        data = _load_cache(session)
        tracks, tracked_bboxes = data["tracks"], data["tracked_bboxes"]
        hex_t1 = data["team_colors_hex"].get(1, "#3498db")
        hex_t2 = data["team_colors_hex"].get(2, "#e74c3c")

        total_frames = len(tracks["players"])
        print(f"[REPLAY] total_frames={total_frames}, "
              f"tracked_bboxes={len(tracked_bboxes)}")

        fps = float(session.get("video_fps") or 25.0)
        _vw = int(session.get("video_width") or 0)
        _vh = int(session.get("video_height") or 0)

        with _video_capture(session["video_path"]) as cap:
            w = _vw or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = _vh or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_dir = sm.session_output_dir(session_id)
        output_path = output_dir / "full_replay.mp4"

        # Decide between single-process and chunked-parallel paths
        n_workers = REPLAY_WORKERS
        if total_frames < REPLAY_PARALLEL_THRESHOLD or n_workers <= 1:
            print(f"[REPLAY] single-process path "
                  f"(frames={total_frames}, workers_avail={n_workers})")
            return _render_full_replay_single(
                session_id, session, task_id, sm,
                tracks, tracked_bboxes, data["team_control"],
                hex_t1, hex_t2, fps, w, h, total_frames, output_path,
            )

        print(f"[REPLAY] parallel rendering: {n_workers} workers × "
              f"~{math.ceil(total_frames / n_workers)} frames each "
              f"(downscale: {h}->{REPLAY_MAX_HEIGHT if h > REPLAY_MAX_HEIGHT else h})")
        sm.update_task(session_id, task_id, progress=10)

        # Build chunk arg tuples
        tracks_cache_path = session.get("tracks_cache_path") or str(output_dir / "tracks.pkl")
        yolo_path = get_yolo_model_path()
        chunk_size = math.ceil(total_frames / n_workers)
        # User-supplied match periods drive frame skipping at render time.
        # Frames outside every period are silently dropped, producing a
        # shorter mp4 that's pure match footage stitched back-to-back.
        match_periods_raw = session.get("match_periods_frames")
        match_periods = (
            [tuple(p) for p in match_periods_raw]
            if match_periods_raw else None
        )
        chunk_args = []
        for ci in range(n_workers):
            start = ci * chunk_size
            end = min(start + chunk_size, total_frames)
            if start >= end:
                continue
            chunk_path = output_dir / f"full_replay_chunk_{ci:03d}.mp4"
            chunk_args.append((
                start, end, session["video_path"], tracks_cache_path,
                hex_t1, hex_t2, fps, w, h,
                yolo_path, str(chunk_path), REPLAY_MAX_HEIGHT,
                match_periods,
            ))

        # Dispatch to worker pool
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_render_chunk_to_mp4, c): c for c in chunk_args}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                if r.get("rc", 0) != 0:
                    raise RuntimeError(
                        f"Chunk render failed (rc={r['rc']}, range={r['start']}..{r['end']})"
                    )
                results.append(r)
                done += 1
                pct = 10 + int(80 * done / len(chunk_args))
                sm.update_task(session_id, task_id, progress=pct)

        # Sort by start frame so concat order is correct
        results.sort(key=lambda r: r["start"])

        # Concat with the demuxer (-c copy is a stream copy, basically instant)
        sm.update_task(session_id, task_id, progress=92, stage="concat")
        concat_list = output_dir / "full_replay_concat.txt"
        with open(concat_list, "w") as cf:
            for r in results:
                cf.write(f"file '{Path(r['path']).name}'\n")

        concat_log = output_dir / "full_replay_concat.log"
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy",
            "-movflags", "+faststart", str(output_path),
        ]
        with open(concat_log, "wb") as _errf:
            crc = sp.run(concat_cmd, cwd=str(output_dir),
                         stdout=sp.DEVNULL, stderr=_errf).returncode
        if crc != 0 or not output_path.exists() or output_path.stat().st_size < 1024:
            tail = ""
            try:
                tail = concat_log.read_bytes()[-2000:].decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"ffmpeg concat failed (rc={crc}, "
                f"size={output_path.stat().st_size if output_path.exists() else 0}B). "
                f"stderr tail:\n{tail}"
            )

        # Clean up intermediate chunk files
        for r in results:
            try: Path(r["path"]).unlink()
            except Exception: pass
        for p in (concat_list, concat_log):
            try: p.unlink()
            except Exception: pass

        rendered_total = sum(r["rendered"] for r in results)
        print(f"[REPLAY] done — {rendered_total}/{total_frames} frames across "
              f"{len(results)} chunks, output={output_path.stat().st_size // 1024} KB")

        sm.update_task(session_id, task_id, progress=100)
        _finish_task(sm, session_id, task_id, output_path)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("full_replay", session_id, exc)


def _render_full_replay_single(session_id, session, task_id, sm,
                                tracks, tracked_bboxes, team_control,
                                hex_t1, hex_t2, fps, w, h, total_frames, output_path):
    """Original single-process path. Used for short clips."""
    import subprocess as sp

    tracker_obj = Tracker(get_yolo_model_path())
    team_assigner = TeamAssigner()
    team_assigner.team_colors = {
        1: np.array([int(hex_t1[5:7], 16), int(hex_t1[3:5], 16), int(hex_t1[1:3], 16)]),
        2: np.array([int(hex_t2[5:7], 16), int(hex_t2[3:5], 16), int(hex_t2[1:3], 16)]),
    }
    config = SoccerPitchConfiguration() if HAS_SPORTS else None

    # Downscale path (Plan C) — even on single-process we benefit from smaller encode
    out_w, out_h = w, h
    if REPLAY_MAX_HEIGHT and h > REPLAY_MAX_HEIGHT:
        scale = REPLAY_MAX_HEIGHT / h
        out_w = (int(w * scale) // 2) * 2
        out_h = (int(REPLAY_MAX_HEIGHT) // 2) * 2
    do_resize = (out_w, out_h) != (w, h)

    # Single-process path also honours user-supplied match_periods
    match_periods_raw = session.get("match_periods_frames")
    match_periods = (
        [tuple(p) for p in match_periods_raw]
        if match_periods_raw else None
    )
    def _in_period(i: int) -> bool:
        if not match_periods:
            return True
        for ps, pe in match_periods:
            if ps <= i < pe:
                return True
        return False

    with _video_capture(session["video_path"]) as cap:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "pipe:",
            *_h264_encode_args(),
            "-movflags", "+faststart",
            str(output_path),
        ]
        ff_log = sm.session_output_dir(session_id) / "full_replay_ffmpeg.log"
        with open(ff_log, "wb") as _errf, \
             sp.Popen(ffmpeg_cmd, stdin=sp.PIPE,
                      stdout=sp.DEVNULL, stderr=_errf) as proc:
            try:
                sm.update_task(session_id, task_id, progress=15)
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                    if not _in_period(i):
                        continue   # drop halftime / pre-game / post-game frames
                    args = (i, frame, tracks, tracked_bboxes, team_control,
                            team_assigner, tracker_obj, config, hex_t1, hex_t2)
                    _, rendered = _render_single_frame_worker_full(args)
                    if do_resize:
                        rendered = cv2.resize(rendered, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    try:
                        proc.stdin.write(rendered.tobytes())
                    except BrokenPipeError:
                        break
                    if i % 60 == 0:
                        pct = int(15 + (i / total_frames) * 80)
                        sm.update_task(session_id, task_id, progress=pct)
            finally:
                if proc.stdin:
                    try: proc.stdin.close()
                    except Exception: pass

    if proc.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1024:
        tail = ""
        try:
            with open(ff_log, "rb") as _rf:
                tail = _rf.read()[-2000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"FFmpeg full_replay encode failed (rc={proc.returncode}, "
            f"size={output_path.stat().st_size if output_path.exists() else 0}B). "
            f"stderr tail:\n{tail}"
        )

    _finish_task(sm, session_id, task_id, output_path)


def _draw_ball_marker(frame, ball_cx: int, ball_top_y: int, color):
    """Draw a small downward-pointing triangle floating above the ball.

    Replaces the old yellow rectangle + 'Ball 0.83' label with a clean
    broadcast-style indicator: a green ▼ ~14 px above the ball, easier to
    read on busy pitches and consistent with the possession triangle.
    """
    # Triangle apex sits 6 px above the ball, body 14 px tall × 16 px wide.
    apex_y = max(0, ball_top_y - 6)
    base_y = max(0, ball_top_y - 22)
    half_w = 9
    pts = np.array([
        [ball_cx, apex_y],
        [ball_cx - half_w, base_y],
        [ball_cx + half_w, base_y],
    ], dtype=np.int32)
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)


def _draw_speed_badge(frame, center_x: int, bottom_y: int, speed_kmh: float):
    """Render a broadcast-style speed pill BELOW the tracked player.

    Layout: gold rounded pill, drop shadow, white "##.# km/h" reading.
    Mimics the Bundesliga / Premier League TV overlay. Sits 12 px under the
    foot ellipse, centered horizontally on the player.
    """
    if cv2 is None or np is None:
        return frame
    h, w = frame.shape[:2]
    text = f"{speed_kmh:.1f} km/h"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad_x, pad_y = 10, 6
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2 + baseline
    # Pill rectangle, centered under the player's feet.
    x0 = max(2, center_x - box_w // 2)
    y0 = min(h - box_h - 2, bottom_y + 12)
    x1 = min(w - 2, x0 + box_w)
    y1 = y0 + box_h

    # Drop shadow (slightly offset, dark, semi-transparent).
    shadow = frame.copy()
    cv2.rectangle(shadow, (x0 + 2, y0 + 3), (x1 + 2, y1 + 3), (0, 0, 0), -1)
    cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)

    # Gold pill background. BGR (50, 200, 245) = soft amber, matches the
    # tracked-player gold ellipse used elsewhere.
    cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 200, 245), -1)
    # Thin black outline so the pill never blends into yellow ad boards.
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 1)

    # White text, slight shadow for legibility.
    text_x = x0 + pad_x
    text_y = y1 - pad_y - baseline + th
    cv2.putText(frame, text, (text_x + 1, text_y + 1), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


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

        # Possession marker: BLUE ▼ above the player who has the ball.
        # Was red — switched to blue per user spec so it visually distinguishes
        # from "danger" reds elsewhere.
        POSSESSION_BGR = (255, 80, 0)   # vivid blue (BGR)

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
                    frame = tracker.draw_triangle(frame, bbox, POSSESSION_BGR)

        # Draw target player
        if samurai_bbox_xyxy is not None:
            y2 = int(samurai_bbox_xyxy[3])
            x_c = int((samurai_bbox_xyxy[0] + samurai_bbox_xyxy[2]) / 2)
            width = samurai_bbox_xyxy[2] - samurai_bbox_xyxy[0]

            if width <= 0:
                return i, frame

            # Vivid magenta outer ring — pops on green pitch and against any jersey
            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width), int(0.35*width)),
                       angle=0.0, startAngle=-45, endAngle=235, color=(255, 0, 255), thickness=10)

            # Bright cyan inner ring — high-contrast complement to the magenta
            cv2.ellipse(frame, center=(x_c, y2), axes=(int(width*0.9), int(0.35*width*0.9)),
                       angle=0.0, startAngle=-45, endAngle=235, color=(255, 255, 0), thickness=5)

            if current_yolo_info and current_yolo_info.get('has_ball', False):
                confidence = current_yolo_info.get('possession_confidence', 1.0)
                if confidence > 0.6:
                    frame = tracker.draw_triangle(frame, samurai_bbox_xyxy, POSSESSION_BGR)

            # ── Speed badge under the tracked player (mimics broadcast HUDs).
            # User asked for "图二的样子但更好看" — gold pill, drop shadow,
            # bold reading, units suffix, only when speed is meaningful.
            speed = float(current_yolo_info.get("speed", 0.0)) if current_yolo_info else 0.0
            if speed >= 0.5:  # hide noisy 0-2 km/h floats so the badge isn't always on
                _draw_speed_badge(
                    frame,
                    center_x=x_c,
                    bottom_y=y2,
                    speed_kmh=speed,
                )

        # Draw ball — small green ▼ floating ABOVE the ball (no rectangle).
        # User specifically asked for the box to go away in favor of a clean
        # broadcast-style indicator.
        BALL_GREEN = (0, 255, 80)
        for _, info in tracks['ball'][i].items():
            if not info or 'bbox' not in info:
                continue
            bbox = info['bbox']
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            cx_ball = int((bbox[0] + bbox[2]) / 2)
            top_ball = int(bbox[1])
            _draw_ball_marker(frame, cx_ball, top_ball, BALL_GREEN)

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
        fps = float(session.get("video_fps") or 25.0)

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


# ── 3h. AI 总结（Gemini 多模态）──────────────────────────────────────────────

def _render_gemini_video(video_path: str, bboxes_dict: dict,
                          output_path: Path, stride: int = 5,
                          progress_cb=None) -> bool:
    """
    为 Gemini 生成全分辨率带框视频（不重跑检测，直接复用 bboxes_dict）。

    原理：
      FFmpeg 解码原始视频（优先 CUDA 加速） → Python 每 stride 帧叠一次 bbox
      → FFmpeg 编码 H264 输出。
      纯绘图，零 AI 推理。stride=5 ≈ 4.8fps，比 samurai_temp.mp4 的 2.4fps 好 2×，
      分辨率完全保留原始（1080p）。

    Args:
        progress_cb: (frames_written, total_to_write) → None，用于外层更新进度
    Returns:
        True 成功，False 失败（调用方 fallback 到 samurai_temp.mp4）
    """
    try:
        import subprocess as _sp

        with _video_capture(video_path) as cap:
            fps  = cap.get(cv2.CAP_PROP_FPS) or 24
            w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_fps = max(1.0, fps / stride)
        total_out = (total_frames + stride - 1) // stride  # 预估输出帧数

        # ── FFmpeg 解码端（优先 CUDA，失败自动降级 CPU）──
        def _make_decode_cmd(hwaccel: bool) -> list:
            base = ["ffmpeg"]
            if hwaccel:
                base += ["-hwaccel", "cuda"]
            base += [
                "-i", video_path,
                "-vf", f"select='not(mod(n\\,{stride}))',setpts=N/FRAME_RATE/TB",
                "-vsync", "vfr",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ]
            return base

        # ── FFmpeg 编码端 ──
        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(out_fps), "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
            "-movflags", "+faststart",   # moov -> 文件头，Gemini 分析和浏览器预览都更快
            str(output_path),
        ]

        frame_size = w * h * 3
        frames_written = 0

        # CUDA/CPU 重试必须把 encoder 也重建——否则 CUDA 半途失败时，
        # 已写入 encoder stdin 的 N 帧会和 CPU 重试的帧 0..M 拼成错位视频
        for use_cuda in (True, False):
            if output_path.exists():
                try: output_path.unlink()
                except Exception: pass

            with _sp.Popen(encode_cmd, stdin=_sp.PIPE,
                           stdout=_sp.DEVNULL, stderr=_sp.DEVNULL) as encode_proc:
                try:
                    with _sp.Popen(_make_decode_cmd(use_cuda),
                                   stdout=_sp.PIPE, stderr=_sp.DEVNULL) as decode_proc:
                        try:
                            frames_written = 0
                            orig_idx       = 0   # 对应原始视频帧号
                            ok = True
                            while True:
                                raw = decode_proc.stdout.read(frame_size)
                                if len(raw) < frame_size:
                                    break
                                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

                                # 叠加追踪框（bboxes_dict 已插值到每帧原始分辨率）
                                if orig_idx in bboxes_dict:
                                    bx, by, bw_, bh_ = bboxes_dict[orig_idx]
                                    x1, y1 = int(bx), int(by)
                                    x2, y2 = int(bx + bw_), int(by + bh_)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                                    cv2.putText(frame, "TRACKED",
                                                (x1, max(y1 - 10, 20)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                # 时间戳左上角
                                secs = orig_idx / fps
                                cv2.putText(frame,
                                            f"{int(secs // 60):02d}:{secs % 60:04.1f}",
                                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                                try:
                                    encode_proc.stdin.write(frame.tobytes())
                                except BrokenPipeError:
                                    ok = False
                                    break

                                frames_written += 1
                                orig_idx       += stride
                                if progress_cb and frames_written % 300 == 0:
                                    progress_cb(frames_written, total_out)
                        finally:
                            if decode_proc.stdout:
                                try: decode_proc.stdout.close()
                                except Exception: pass
                finally:
                    if encode_proc.stdin:
                        try: encode_proc.stdin.close()
                        except Exception: pass

            if ok and frames_written > 0:
                break   # 成功，不需要重试 CPU
            if use_cuda:
                print("[AI_SUMMARY] CUDA decode failed, retrying with CPU (rebuilding encoder)...")

        success = output_path.exists() and output_path.stat().st_size > 0
        print(f"[AI_SUMMARY] gemini_video: {frames_written} frames written, "
              f"size={output_path.stat().st_size // 1024 // 1024}MB" if success else
              f"[AI_SUMMARY] gemini_video render failed")
        return success

    except Exception as exc:
        print(f"[AI_SUMMARY] _render_gemini_video error: {exc}")
        return False


def run_ai_summary(session_id: str, session: dict, task_id: str, sm: SessionManager):
    """
    多模态 AI 战术分析：把全分辨率标注视频 + 统计 JSON 送给 Qwen-VL 生成报告。

    工作流：
      1. 加载 tracks.pkl（统计数据 + 分段 + 颜色）
      2. 拼装统计 JSON（控球率 / 速度 / 分段 / 球员 summary）
      3. 渲染 ai_video.mp4：原始分辨率 + stride=5 ≈ 5fps + 追踪框（CUDA 加速）
         fallback → samurai_temp.mp4（低质但已存在）
      4. ffmpeg 抽帧（1fps，≤60 帧）→ image_url × N（每帧 ~100KB，无大小限制）
      5. 调用 Qwen-VL（DashScope 国际版 OpenAI-compat API）生成 markdown 报告
      6. 写入 task.result + 落盘 ai_summary.md
    """
    try:
        import json
        sm.update_task(session_id, task_id, status="running", progress=5,
                       stage="loading_data")

        # ── 1. 依赖检查 & 配置 ──
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set in environment")
        model_name = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-flash")

        try:
            from openai import OpenAI as _OpenAI
        except ImportError as ie:
            raise RuntimeError(
                "openai not installed. "
                "Run: pip install openai>=1.0.0") from ie

        _qwen_client = _OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        # ── 2. 加载统计数据 ──
        data           = _load_cache(session)
        team_control   = np.array(data.get("team_control", []))
        hex_colors     = data.get("team_colors_hex", {})
        segments       = data.get("segments", [])
        player_summary = session.get("player_summary", {})
        total_frames   = int(session.get("total_frames", 0))

        fps = float(session.get("video_fps") or 25.0)

        t1  = int(np.sum(team_control == 1)) if team_control.size else 0
        t2  = int(np.sum(team_control == 2)) if team_control.size else 0
        neu = int(np.sum(team_control == 0)) if team_control.size else 0
        total_ctrl = t1 + t2 + neu or 1

        # ── 3. 组织给 Qwen 的 JSON ──
        stats_payload = {
            "video": {
                "total_frames":   total_frames,
                "duration_sec":   round(total_frames / fps, 1) if fps else 0,
                "fps":            round(fps, 1),
            },
            "possession": {
                "team1_pct":      round(t1  / total_ctrl * 100, 1),
                "team2_pct":      round(t2  / total_ctrl * 100, 1),
                "neutral_pct":    round(neu / total_ctrl * 100, 1),
                "team1_color":    hex_colors.get(1, "#3498db"),
                "team2_color":    hex_colors.get(2, "#e74c3c"),
                "switches":       int(data.get("possession_switches", 0)),
            },
            "segments":       segments,
            "tracked_player": player_summary,
        }
        stats_json = json.dumps(stats_payload, ensure_ascii=False, indent=2)

        # ── 4. 渲染全分辨率带框视频（CUDA 加速，不重跑检测）──────────────────
        output_dir       = sm.session_output_dir(session_id)
        tracked_bboxes   = data.get("tracked_bboxes", {})
        gemini_vid_path  = output_dir / "gemini_video.mp4"

        sm.update_task(session_id, task_id, progress=10, stage="rendering_ai_video")

        def _render_progress(done, total_est):
            pct = int(10 + min(14, done / max(total_est, 1) * 14))
            sm.update_task(session_id, task_id, progress=pct,
                           stage="rendering_ai_video")

        rendered = _render_gemini_video(
            session["video_path"], tracked_bboxes, gemini_vid_path,
            stride=5, progress_cb=_render_progress,
        )

        if rendered:
            video_for_ai  = str(gemini_vid_path)
            video_source  = "full_res_annotated"
        else:
            # fallback：samurai_temp.mp4（低质量但已存在）
            samurai_annotated = output_dir / "samurai_temp.mp4"
            if samurai_annotated.exists() and samurai_annotated.stat().st_size > 0:
                video_for_ai = str(samurai_annotated)
                video_source = "samurai_annotated"
            else:
                video_for_ai = session["video_path"]
                video_source = "raw_video"
        print(f"[AI_SUMMARY] Video source: {video_source} → {video_for_ai}")

        # ── 5. ffmpeg 抽帧 → 多张 image_url（OpenAI-compat 兼容方式）────────────
        # video_url base64 有 10MB 单项限制，整段视频超标。
        # 改为每秒抽 1 帧（最多 60 帧），每张 JPEG ~50-150KB，完全在限制内。
        import time as _time, base64 as _b64, subprocess as _sp
        import tempfile as _tf, shutil as _sh
        sm.update_task(session_id, task_id, progress=28, stage="extracting_frames")

        _frame_dir = Path(_tf.mkdtemp())
        try:
            _sp.run([
                "ffmpeg", "-y", "-i", video_for_ai,
                "-vf", "fps=1,scale=640:-2",
                "-q:v", "4", "-frames:v", "60",
                str(_frame_dir / "f%04d.jpg"),
            ], check=True, capture_output=True)
            _frame_files = sorted(_frame_dir.glob("f*.jpg"))[:60]
            if not _frame_files:
                raise RuntimeError("ffmpeg produced no frames from AI video")
            _img_parts = []
            for _fp in _frame_files:
                _b64_frame = _b64.b64encode(_fp.read_bytes()).decode()
                _img_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64_frame}"},
                })
            print(f"[AI_SUMMARY] Extracted {len(_frame_files)} frames for Qwen-VL")
        finally:
            _sh.rmtree(_frame_dir, ignore_errors=True)

        # ── 6. 组 prompt + 调用 Qwen（附进度模拟线程，避免进度条冻结）────────────
        sm.update_task(session_id, task_id, progress=40, stage="qwen_reasoning")

        import threading as _threading
        _stop_sim = _threading.Event()
        def _sim_progress():
            t0 = _time.perf_counter()
            while not _stop_sim.is_set():
                elapsed = _time.perf_counter() - t0
                # 40 → 93 linearly over 120 s，之后停在 93
                sim_pct = min(93, int(40 + elapsed / 120 * 53))
                sm.update_task(session_id, task_id,
                               progress=sim_pct, stage="qwen_reasoning")
                _time.sleep(4)
        _sim_thread = _threading.Thread(target=_sim_progress, daemon=True)
        _sim_thread.start()

        # 根据实际视频源调整 prompt
        if video_source == "raw_video":
            _video_desc = (
                "视频为原始比赛画面（无标注）。"
                "由于渲染标注视频失败，请主要依据下方统计数据进行分析，"
                "视频画面仅作辅助参考（无法识别具体追踪球员）。"
            )
            _player_section = (
                "## 被追踪球员分析\n"
                "本节请完全依据下方统计数据（max_speed / distance / possession_seconds 等）分析，"
                "不要根据视频画面猜测球员身份。\n\n"
            )
        else:
            _video_desc = (
                "视频为带标注的比赛画面（约 5fps），"
                + ("绿色高亮框（TRACKED）标注的是正在被追踪的核心球员。"
                   if video_source == "full_res_annotated"
                   else "SAM2 mask 叠加视频，绿框为被追踪球员。")
            )
            _player_section = (
                "## 被追踪球员分析\n"
                "重点分析视频中绿色框标注的那名球员：跑位、速度爆发、控球、影响力。\n\n"
            )

        system_text = (
            f"你是一个专业足球战术分析师。{_video_desc}"
            "结合以下统计数据和视频画面，生成一份**中文 Markdown 报告**，"
            "包含以下章节（严格按顺序、标题用 ## 二级标题）：\n\n"
            "## 比赛概览\n"
            "简述双方控球对比、关键跑动数据、比赛节奏。\n\n"
            + _player_section +
            "## 战术观察\n"
            "阵型特征、进攻模式、防守组织，基于画面实际观察。\n\n"
            "## 改进建议\n"
            "3 条具体可执行的训练/战术建议。\n\n"
            "要求：语言简练，数据引用具体，不要泛泛而谈。"
        )
        _prompt_text = (
            f"{system_text}\n\n## 统计数据\n```json\n{stats_json}\n```\n\n请生成报告："
        )

        # 构建消息内容：帧图片列表 + 文字 prompt
        _user_content = _img_parts + [{"type": "text", "text": _prompt_text}]

        result_data = None
        report_path = None
        try:
            _resp = _qwen_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": _user_content}],
                temperature=0.4,
                top_p=0.9,
                max_tokens=4096,
            )
        finally:
            _stop_sim.set()
            _sim_thread.join(timeout=2)

        report_md = (_resp.choices[0].message.content or "").strip()
        if not report_md:
            raise RuntimeError("Qwen returned empty response")
        print(f"[AI_SUMMARY] Qwen response: {len(report_md)} chars")

        # ── 7. 落盘 + 完成任务 ──
        sm.update_task(session_id, task_id, progress=95, stage="saving_report")
        report_path = output_dir / "ai_summary.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        result_data = {
            "report_markdown": report_md,
            "video_source":    video_source,
            "model":           model_name,
            "char_count":      len(report_md),
        }
        _finish_task(sm, session_id, task_id, report_path, result=result_data)

    except Exception as exc:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))
        _log_error("ai_summary", session_id, exc)


# ── 模块级 tracks.pkl 缓存（避免并发任务多次读 500MB）──────────────────────
# 多个按需任务（热力图/速度/小地图...）并发时会同时命中这里，需要锁防止重复加载。
_TRACKS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_TRACKS_CACHE_LOCK = threading.Lock()
_TRACKS_CACHE_MAX = 2      # 最多缓存 2 个 session（Colab 单用户足够）


def _load_cache(session: dict) -> dict:
    cache_path = session.get("tracks_cache_path")
    if not cache_path:
        raise FileNotFoundError("tracks.pkl not found — global analysis may have failed")

    # If the local file doesn't exist, try downloading from R2
    # (happens when a feature task runs on a different RunPod worker)
    if not Path(cache_path).exists():
        try:
            from ..storage.r2 import download_from_r2
            session_id = session.get("session_id") or session.get("id", "")
            remote_key = f"{session_id}/tracks.pkl"
            local_path = Path(cache_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if download_from_r2(remote_key, local_path):
                print(f"[INFO] Downloaded tracks.pkl from R2 for session {session_id}")
            else:
                raise FileNotFoundError(
                    f"tracks.pkl not found locally ({cache_path}) and R2 download failed"
                )
        except ImportError:
            raise FileNotFoundError(
                f"tracks.pkl not found at {cache_path} and R2 module unavailable"
            )

    with _TRACKS_CACHE_LOCK:
        if cache_path not in _TRACKS_CACHE:
            with open(cache_path, "rb") as f:
                _TRACKS_CACHE[cache_path] = pickle.load(f)
        # LRU：命中后移到末尾，超限时逐出最旧
        _TRACKS_CACHE.move_to_end(cache_path)
        while len(_TRACKS_CACHE) > _TRACKS_CACHE_MAX:
            _TRACKS_CACHE.popitem(last=False)
        return _TRACKS_CACHE[cache_path]


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
    """任务成功完成时统一写入状态，如果配置了 R2 则上传到云端"""
    from ..storage.r2 import upload_to_r2
    
    file_path = Path(file_path)
    session_dir = sm.session_output_dir(session_id)
    try:
        rel_path = str(file_path.resolve().relative_to(session_dir.resolve()))
    except ValueError:
        rel_path = file_path.name
        
    # Attempt to upload to R2
    remote_key = f"{session_id}/{rel_path}"
    r2_url = upload_to_r2(file_path, remote_key)
    
    # Fallback to local URL if R2 fails or is not configured
    final_url = r2_url if r2_url else rel_path
    
    sm.update_task(
        session_id, task_id,
        status="done",
        progress=100,
        # Store the R2 public URL or the local relative path
        url=final_url,
        result=result,
    )


def _log_error(name: str, session_id: str, exc: Exception):
    print(f"[ERROR] {name} | session={session_id}")
    print(traceback.format_exc())
