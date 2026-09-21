"""
e2e_video_detection_profiler.py — End-to-End Real Video Detection Profiler

Performs stage-by-stage wall-clock latency profiling on actual 1080p broadcast video:
- Video decode (cv2 / AsyncFramePrefetcher)
- Resize / Preprocessing (Letterbox, RGB, normalization)
- Model Inference (PyTorch forward pass)
- Post-processing / NMS
- Tracking Association (ByteTrack)
- Pitch Keypoint & Camera Processing (Keypoint detector + Homography)
- Interpolation & Sync
- Total Wall-Clock, Effective E2E FPS, Realtime Factor (RTF), CPU/RAM telemetry
"""

import os
import sys
import time
import math
import psutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

from server.pipeline.detection_accelerator import (
    TacticalViewGater,
    AdaptiveTemporalStrideController,
    AsyncFramePrefetcher,
    stream_video_chunks_safe,
)


class StageTimer:
    """Accumulates high-precision wall-clock time for pipeline stages."""
    def __init__(self):
        self.timings: Dict[str, float] = {
            "video_decode": 0.0,
            "resize_preprocessing": 0.0,
            "model_inference": 0.0,
            "postprocessing_nms": 0.0,
            "tracking_association": 0.0,
            "pitch_camera_processing": 0.0,
            "track_interpolation": 0.0,
            "sync_and_overhead": 0.0,
        }
        self.counts: Dict[str, int] = {k: 0 for k in self.timings}

    def add(self, stage: str, duration: float):
        if stage in self.timings:
            self.timings[stage] += duration
            self.counts[stage] += 1
        else:
            self.timings[stage] = duration
            self.counts[stage] = 1


class E2EVideoDetectionProfiler:
    def __init__(
        self,
        video_path: str = "backend/uploads/fe7f8619b7ea_test_17.mp4",
        detector_weights: str = "backend/weights/football/best.pt",
        keypoint_weights: str = "backend/weights/keypoints/best.pt",
        device: str = "cpu",
    ):
        self.video_path = video_path
        self.detector_weights = detector_weights
        self.keypoint_weights = keypoint_weights
        self.device = device

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not Path(detector_weights).exists():
            raise FileNotFoundError(f"Detector weights not found: {detector_weights}")

        # Probe video
        cap = cv2.VideoCapture(video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_video_frames / self.native_fps
        cap.release()

        # Load models
        print(f"[PROFILER] Loading YOLO detector from {detector_weights} (device={device})...")
        self.detector = YOLO(detector_weights)
        self.keypoint_detector = None
        if Path(keypoint_weights).exists():
            print(f"[PROFILER] Loading Keypoint detector from {keypoint_weights}...")
            self.keypoint_detector = YOLO(keypoint_weights)

        # Warmup models
        _warmup = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.detector.predict([_warmup], imgsz=640, device=self.device, verbose=False)
        if self.keypoint_detector:
            self.keypoint_detector.predict([_warmup], imgsz=640, device=self.device, verbose=False)
        print("[PROFILER] Warmup completed.")

    def run_profile(
        self,
        max_frames: int = 150,
        imgsz: int = 1280,
        stride: int = 3,
        use_adaptive_stride: bool = False,
        use_tactical_gater: bool = False,
        use_async_prefetch: bool = False,
        keypoint_stride: int = 20,
        enable_keypoints: bool = True,
        conf_thresh: float = 0.59,
        iou_thresh: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline with stage timers on the real video.
        """
        timer = StageTimer()
        process = psutil.Process()
        cpu_samples = []
        rss_start = process.memory_info().rss / (1024 * 1024)

        n_frames = min(max_frames, self.total_video_frames)
        clip_duration = n_frames / self.native_fps

        gater = TacticalViewGater(min_grass_ratio=0.20) if use_tactical_gater else None
        controller = AdaptiveTemporalStrideController(base_stride=stride, min_stride=max(1, stride-1), max_stride=stride+2) if use_adaptive_stride else None

        # Setup frame reader
        cap = cv2.VideoCapture(self.video_path)
        prefetcher = None
        if use_async_prefetch:
            def _provider():
                ret, frame = cap.read()
                return frame if ret else None
            prefetcher = AsyncFramePrefetcher(_provider, max_buffered_frames=64)
            prefetcher.start()

        try:
            from server.pipeline.bytetrack_adaptive_compensator import AdaptiveByteTracker
            tracker = AdaptiveByteTracker(
                base_fps=fps,
                base_stride=stride,
                track_activation_threshold=conf_thresh,
                minimum_matching_threshold=0.80,
                base_lost_buffer_frames=15,
            )
        except Exception:
            tracker = sv.ByteTrack()
        tracks = {"players": [{} for _ in range(n_frames)], "ball": [{} for _ in range(n_frames)]}
        detected_frames_count = 0
        keypoint_frames_count = 0

        last_detection_frame = -999
        last_thumb = None

        t_wall_start = time.perf_counter()

        try:
            for fidx in range(n_frames):
                # 1. Video Decode
                t0 = time.perf_counter()
                if prefetcher:
                    # Async prefetch queue get
                    try:
                        item, _ = prefetcher.queue.get(timeout=5.0)
                        if item is None or isinstance(item, Exception):
                            break
                        frame = item
                    except Exception:
                        break
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                timer.add("video_decode", time.perf_counter() - t0)

                # Tactical View & Stride Decision
                should_detect = False
                if use_adaptive_stride and controller:
                    is_tactical = True
                    grass_ratio = 1.0
                    if gater:
                        is_tactical, grass_ratio = gater.is_tactical_view(frame)
                    
                    # Compute thumbnail optical movement proxy if last_thumb available
                    motion_mag = 0.0
                    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
                    if last_thumb is not None:
                        diff = cv2.absdiff(small, last_thumb)
                        motion_mag = float(np.mean(diff))
                    last_thumb = small

                    should_detect = controller.evaluate_frame(fidx, is_tactical, motion_mag)
                else:
                    if fidx % stride == 0:
                        should_detect = True

                # 2. YOLO Player/Ball Detection
                if should_detect:
                    detected_frames_count += 1
                    if use_adaptive_stride and controller is not None and hasattr(tracker, "update_stride"):
                        tracker.update_stride(getattr(controller, "current_stride", stride))

                    t_det_start = time.perf_counter()
                    res = self.detector.predict(
                        [frame],
                        imgsz=imgsz,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        device=self.device,
                        verbose=False
                    )[0]
                    t_det_end = time.perf_counter()

                    # Ultralytics internal speed dict breakdown
                    speed_dict = getattr(res, "speed", {})
                    t_prep_s = speed_dict.get("preprocess", 0.0) / 1000.0
                    t_infer_s = speed_dict.get("inference", 0.0) / 1000.0
                    t_nms_s = speed_dict.get("postprocess", 0.0) / 1000.0

                    timer.add("resize_preprocessing", t_prep_s)
                    timer.add("model_inference", t_infer_s)
                    timer.add("postprocessing_nms", t_nms_s)

                    # 3. Tracking Association
                    t_trk_start = time.perf_counter()
                    boxes = res.boxes
                    if len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cids = boxes.cls.cpu().numpy().astype(int)

                        # Player class (0)
                        player_mask = (cids == 0)
                        if np.any(player_mask):
                            p_xyxy = xyxy[player_mask]
                            p_conf = confs[player_mask]
                            p_cids = cids[player_mask]
                            p_ds = sv.Detections(xyxy=p_xyxy, confidence=p_conf, class_id=p_cids)
                            tracked = tracker.update_with_detections(p_ds)
                            for d in tracked:
                                tid = int(d[4]) if len(d) > 4 else int(d[1])
                                tracks["players"][fidx][tid] = {"bbox": d[0].tolist()}

                        # Ball class (1)
                        ball_mask = (cids == 1)
                        if np.any(ball_mask):
                            b_confs = confs[ball_mask]
                            best_idx = np.argmax(b_confs)
                            tracks["ball"][fidx][1] = {"bbox": xyxy[ball_mask][best_idx].tolist()}

                    timer.add("tracking_association", time.perf_counter() - t_trk_start)

                # 4. Pitch Keypoint Processing
                if enable_keypoints and self.keypoint_detector and (fidx % keypoint_stride == 0):
                    keypoint_frames_count += 1
                    t_kp_start = time.perf_counter()
                    kp_res = self.keypoint_detector.predict(
                        [frame],
                        imgsz=640,
                        conf=0.10,
                        device=self.device,
                        verbose=False
                    )[0]
                    timer.add("pitch_camera_processing", time.perf_counter() - t_kp_start)

                # Periodic CPU / RAM sampling
                if fidx % 25 == 0:
                    cpu_samples.append(process.cpu_percent())

        finally:
            if prefetcher:
                prefetcher.stop()
            cap.release()

        # 5. Track Interpolation (Vectorized linear interpolation across un-detected frames)
        t_interp_start = time.perf_counter()
        self._interpolate_player_tracks(tracks["players"], n_frames)
        timer.add("track_interpolation", time.perf_counter() - t_interp_start)

        t_wall_end = time.perf_counter()
        total_wall_clock = t_wall_end - t_wall_start
        effective_e2e_fps = n_frames / max(total_wall_clock, 1e-5)
        realtime_factor = total_wall_clock / max(clip_duration, 1e-5)

        rss_end = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = rss_end

        # Stage breakdown calculation
        sum_stage_time = sum(timer.timings.values())
        timer.timings["sync_and_overhead"] = max(0.0, total_wall_clock - sum_stage_time)

        stage_breakdown = {}
        for stage, t_val in timer.timings.items():
            pct = (t_val / total_wall_clock) * 100.0 if total_wall_clock > 0 else 0.0
            avg_ms = (t_val / n_frames) * 1000.0 if n_frames > 0 else 0.0
            stage_breakdown[stage] = {
                "total_ms": round(t_val * 1000.0, 2),
                "avg_ms_per_frame": round(avg_ms, 2),
                "percentage": round(pct, 2),
            }

        # Identify primary bottleneck
        primary_bottleneck = max(stage_breakdown.items(), key=lambda x: x[1]["total_ms"])

        return {
            "total_video_duration_s": round(clip_duration, 2),
            "total_frames": n_frames,
            "detected_frames": detected_frames_count,
            "keypoint_frames": keypoint_frames_count,
            "detection_ratio_pct": round((detected_frames_count / n_frames) * 100.0, 1),
            "total_wall_clock_s": round(total_wall_clock, 3),
            "effective_e2e_fps": round(effective_e2e_fps, 2),
            "realtime_factor_rtf": round(realtime_factor, 3),
            "avg_cpu_percent": round(np.mean(cpu_samples) if cpu_samples else 0.0, 1),
            "ram_rss_mb": round(peak_ram_mb, 1),
            "stage_breakdown": stage_breakdown,
            "primary_bottleneck": {
                "stage": primary_bottleneck[0],
                "percentage": primary_bottleneck[1]["percentage"],
                "avg_ms_per_frame": primary_bottleneck[1]["avg_ms_per_frame"],
            },
            "tracks_summary": {
                "unique_players_tracked": len(set(
                    tid for f in tracks["players"] for tid in f.keys()
                )),
                "frames_with_ball": sum(1 for f in tracks["ball"] if f),
            },
            "tracks": tracks,
        }

    def _interpolate_player_tracks(self, player_tracks: List[Dict[int, Any]], total_frames: int):
        """Vectorized interpolation for player track bounding boxes."""
        id_obs: Dict[int, Dict[str, List]] = {}
        for fi, fd in enumerate(player_tracks):
            if not fd:
                continue
            for tid, item in fd.items():
                b = item.get("bbox")
                if b and len(b) == 4:
                    if tid not in id_obs:
                        id_obs[tid] = {"fidxs": [], "bboxes": []}
                    id_obs[tid]["fidxs"].append(fi)
                    id_obs[tid]["bboxes"].append(b)

        for tid, data in id_obs.items():
            fidxs = np.array(data["fidxs"])
            if len(fidxs) < 2:
                continue
            bboxes = np.array(data["bboxes"])
            all_frames = np.arange(fidxs[0], fidxs[-1] + 1)
            interp_x1 = np.interp(all_frames, fidxs, bboxes[:, 0])
            interp_y1 = np.interp(all_frames, fidxs, bboxes[:, 1])
            interp_x2 = np.interp(all_frames, fidxs, bboxes[:, 2])
            interp_y2 = np.interp(all_frames, fidxs, bboxes[:, 3])

            for idx_in_all, f in enumerate(all_frames):
                if tid not in player_tracks[f]:
                    player_tracks[f][tid] = {
                        "bbox": [
                            float(interp_x1[idx_in_all]),
                            float(interp_y1[idx_in_all]),
                            float(interp_x2[idx_in_all]),
                            float(interp_y2[idx_in_all]),
                        ],
                        "interpolated": True,
                    }
