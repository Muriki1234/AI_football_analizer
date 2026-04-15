"""
benchmark.py — 管道各阶段耗时分析

用法（Colab）:
    !python backend/benchmark.py --video /path/to/video.mp4 --samurai /path/to/samurai.pkl

输出示例:
    [BENCH] loading_video       :   0.12s
    [BENCH] yolo_detection      :  45.30s  ← 主要瓶颈
    [BENCH] camera_motion       :   8.20s
    [BENCH] keypoint_detection  :  12.50s
    [BENCH] speed_calculation   :   0.80s
    [BENCH] team_assignment     :  15.40s
    [BENCH] possession_detection:   2.10s
    [BENCH] TOTAL               :  84.42s
"""

import argparse
import pickle
import time
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 强制设置 matplotlib 后端（避免 GUI 报错）─────────────────────────────────
import matplotlib
matplotlib.use("Agg")

from app.pipeline.analysis_core import (
    Tracker, CameraMovementEstimator, KeypointDetector,
    ViewTransformer, AccurateSpeedEstimator, TeamAssigner,
    SmartBallPossessionDetector, read_video, read_frames_at_indices,
    SHORT_VIDEO_FRAMES, MODEL_PATH, KEYPOINT_MODEL_PATH,
)

import numpy as np

_timings = {}

class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t
        _timings[self.label] = elapsed
        print(f"[BENCH] {self.label:<25}: {elapsed:7.2f}s")


def run_benchmark(video_path: str, samurai_pkl: str):
    print(f"\n{'='*50}")
    print(f"Benchmarking: {video_path}")
    print(f"{'='*50}\n")

    total_start = time.perf_counter()

    # ── 1. 视频元数据 ──────────────────────────────────────────────────
    with Timer("loading_video"):
        cap = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
        _vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap.release()
        with open(samurai_pkl, "rb") as f:
            samurai_data = pickle.load(f)
        tracked_bboxes = samurai_data["bboxes"]
        print(f"  → {total} frames @ {fps:.1f}fps, {_vid_w}x{_vid_h}, "
              f"{len(tracked_bboxes)} SAMURAI bboxes")

    # ── 2. YOLO 检测 ───────────────────────────────────────────────────
    tracker = Tracker(MODEL_PATH)
    is_short = (total <= SHORT_VIDEO_FRAMES)

    with Timer("yolo_detection"):
        if is_short:
            frames = read_video(video_path)
            tracks = tracker.get_object_tracks(frames)
        else:
            tracks = tracker.get_object_tracks_streamed(video_path, total)
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        tracker.add_position_to_tracks(tracks)

    n_players = sum(len(p) for p in tracks["players"])
    n_ball    = sum(1 for b in tracks["ball"] if b.get(1))
    print(f"  → {n_players} player detections, {n_ball} ball detections")

    # ── 3. 摄像机补偿 ─────────────────────────────────────────────────
    with Timer("camera_motion"):
        if is_short:
            cam = CameraMovementEstimator(frames[0])
            cam_mov = cam.get_camera_movement(frames)
        else:
            cam = CameraMovementEstimator.from_video_path(video_path)
            cam_mov = cam.get_camera_movement_streamed(video_path, total)
        cam.add_adjust_positions_to_tracks(tracks, cam_mov)

    # ── 4. 关键点 + 透视变换 ───────────────────────────────────────────
    with Timer("keypoint_detection"):
        kp = KeypointDetector(KEYPOINT_MODEL_PATH)
        vt = ViewTransformer()
        if is_short:
            kps = kp.predict(frames, cam_movement=cam_mov)
        else:
            kps = kp.predict_streamed(video_path, total, cam_movement=cam_mov)
        vt.add_transformed_position_to_tracks(tracks, kps)
        vt.interpolate_2d_positions(tracks)

    if is_short:
        del frames
        import gc; gc.collect()

    # ── 5. 速度 & 距离 ────────────────────────────────────────────────
    with Timer("speed_calculation"):
        speed_est = AccurateSpeedEstimator()
        speed_est.add_speed_and_distance_to_tracks(tracks)

    # ── 6. 队伍颜色初始化 ─────────────────────────────────────────────
    team_assigner = TeamAssigner()
    with Timer("team_color_init"):
        team_assigner.assign_team_color_from_video(video_path, tracks["players"], n_samples=8)

    # ── 7. 多帧投票 ───────────────────────────────────────────────────
    with Timer("team_voting"):
        from collections import Counter
        player_vote_dict = {}
        SAMPLE_STEP = max(1, total // 20)
        vote_indices = list(range(0, total, SAMPLE_STEP))
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
                            cluster_id = int(team_assigner.kmeans.predict(color.reshape(1,-1))[0])
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

    # ── 8. 球权检测 ───────────────────────────────────────────────────
    with Timer("possession_detection"):
        poss_detector = SmartBallPossessionDetector(fps=fps, video_w=_vid_w, video_h=_vid_h)
        team_control  = []
        ball_history  = []
        for i, p_tracks in enumerate(tracks["players"]):
            for pid, info in p_tracks.items():
                if not info:
                    continue
                tid = player_final_team.get(pid, 1)
                info["team"] = tid
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
                team_control.append(p_tracks[pid_has_ball].get("team", 0))
            else:
                team_control.append(0)

    # ── 汇总 ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*50}")
    print(f"[BENCH] {'TOTAL':<25}: {total_elapsed:7.2f}s")
    print(f"[BENCH] per_frame_avg           : {total_elapsed/total*1000:7.1f}ms/frame")
    print(f"\n排序（最慢 → 最快）:")
    for label, t in sorted(_timings.items(), key=lambda x: -x[1]):
        pct = t / total_elapsed * 100
        print(f"  {label:<25}: {t:7.2f}s  ({pct:.1f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True, help="视频文件路径")
    parser.add_argument("--samurai", required=True, help="SAMURAI pkl 缓存路径")
    args = parser.parse_args()
    run_benchmark(args.video, args.samurai)
