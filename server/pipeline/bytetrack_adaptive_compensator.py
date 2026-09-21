"""
bytetrack_adaptive_compensator.py - Stride-Adaptive ByteTrack Engine

Mathematical & Empirical Principle:
1. In ByteTrack, Kalman state transitions advance by dt = 1 step per call.
2. In supervision's ByteTrack implementation:
   max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
   This determines how many call steps an unobserved track persists before being purged.
3. In sports broadcast video with Stride = s (e.g. s=3 or s=4):
   If lost_track_buffer is kept at standard 30 (with frame_rate=25), max_time_lost is 25 steps.
   At stride 3, 25 steps = 75 video frames = 3.0 seconds!
   When a player runs off-screen or is occluded, a phantom Kalman track drifts blindly
   across the pitch for 3 full seconds at constant velocity. When another player arrives in
   that spatial neighborhood, the phantom track matches them, causing severe identity
   switches (IDSW) and track bleeding.
4. Conversely, setting an appropriate scale-adjusted buffer:
   step_buffer = max(4, int(round(base_buffer_frames / stride)))
   purges phantom tracks within 0.4~0.8s, eliminating ghost drift while preserving
   continuity during short occlusions.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class AdaptiveByteTracker:
    """
    Stride-aware wrapper around supervision.ByteTrack.
    Dynamically tunes track lifetime buffer and matching thresholds
    to maintain high tracking fidelity (HOTA / IDF1) and prevent ID switches
    under strided video detection.
    """

    def __init__(
        self,
        base_fps: float = 25.0,
        base_stride: int = 3,
        track_activation_threshold: float = 0.45,
        minimum_matching_threshold: float = 0.80,
        base_lost_buffer_frames: int = 15,
        minimum_consecutive_frames: int = 1,
    ):
        self.base_fps = float(base_fps)
        self.current_stride = max(1, int(base_stride))
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.base_lost_buffer_frames = base_lost_buffer_frames
        self.minimum_consecutive_frames = minimum_consecutive_frames

        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initializes ByteTrack with stride-scaled buffer parameters."""
        # Calculate appropriate step buffer based on stride
        # We want max_time_lost to be roughly (base_lost_buffer_frames / current_stride) steps
        target_steps = max(4, int(round(self.base_lost_buffer_frames / self.current_stride)))
        
        # Supervision computes: max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        # To make max_time_lost == target_steps with frame_rate=30:
        # lost_track_buffer = target_steps
        self.step_buffer = target_steps

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.step_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=30,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

    def update_stride(self, new_stride: int) -> None:
        """Updates stride on the fly if dynamic temporal stride modulation is active."""
        new_stride = max(1, int(new_stride))
        if new_stride != self.current_stride:
            self.current_stride = new_stride
            target_steps = max(4, int(round(self.base_lost_buffer_frames / self.current_stride)))
            self.step_buffer = target_steps
            self.tracker.max_time_lost = target_steps

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """Proxies detection updates to the tuned ByteTrack instance."""
        if detections is None or len(detections) == 0:
            empty_ds = sv.Detections.empty()
            return self.tracker.update_with_detections(empty_ds)
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Resets tracker state."""
        self._init_tracker()


def interpolate_sparse_tracks(
    tracks_by_frame: List[Dict[int, Dict[str, Any]]],
    total_frames: int,
    max_gap_frames: int = 30,
) -> List[Dict[int, Dict[str, Any]]]:
    """
    Vectorized NumPy interpolation for bounding boxes between sampled detection frames.
    Fills in intermediate bounding box coordinates for each track ID.
    """
    id_observations: Dict[int, Dict[str, List]] = {}
    for fi, frame_dict in enumerate(tracks_by_frame):
        if not frame_dict:
            continue
        for tid, item in frame_dict.items():
            bbox = item.get("bbox")
            if bbox and len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                if tid not in id_observations:
                    id_observations[tid] = {"frames": [], "bboxes": [], "meta": {}}
                id_observations[tid]["frames"].append(fi)
                id_observations[tid]["bboxes"].append(bbox)
                for k, v in item.items():
                    if k != "bbox":
                        id_observations[tid]["meta"][k] = v

    output_tracks: List[Dict[int, Dict[str, Any]]] = [{} for _ in range(total_frames)]

    for tid, obs in id_observations.items():
        f_arr = np.array(obs["frames"], dtype=int)
        if len(f_arr) == 0:
            continue
        if len(f_arr) == 1:
            fi = f_arr[0]
            if 0 <= fi < total_frames:
                entry = {"bbox": obs["bboxes"][0]}
                entry.update(obs["meta"])
                output_tracks[fi][tid] = entry
            continue

        b_arr = np.array(obs["bboxes"], dtype=np.float32)

        for seg_idx in range(len(f_arr) - 1):
            f_start = f_arr[seg_idx]
            f_end = f_arr[seg_idx + 1]
            gap = f_end - f_start

            if gap <= 0:
                continue

            if gap > max_gap_frames:
                if 0 <= f_start < total_frames:
                    entry = {"bbox": b_arr[seg_idx].tolist()}
                    entry.update(obs["meta"])
                    output_tracks[f_start][tid] = entry
                continue

            target_f = np.arange(f_start, f_end + 1)
            valid_mask = (target_f >= 0) & (target_f < total_frames)
            target_f = target_f[valid_mask]

            if len(target_f) == 0:
                continue

            alpha = (target_f - f_start) / float(gap)
            b_start = b_arr[seg_idx]
            b_end = b_arr[seg_idx + 1]

            interp_boxes = (1.0 - alpha[:, None]) * b_start[None, :] + alpha[:, None] * b_end[None, :]

            for fi, box in zip(target_f, interp_boxes):
                entry = {"bbox": [float(round(coord, 2)) for coord in box]}
                entry.update(obs["meta"])
                output_tracks[fi][tid] = entry

    return output_tracks
