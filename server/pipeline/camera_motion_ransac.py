"""
camera_motion_ransac.py — Robust RANSAC & Median Flow Camera Movement Estimator

Replaces the naive max-displacement heuristic in CameraMovementEstimator to eliminate:
1. Foreground athlete / referee motion bleed-through:
   Naive code took max(displacement), causing running players in the centre circle to register as 15-50px camera shake.
2. False pan triggers in KeypointDetector:
   Spurious camera spikes exceeded _PAN_TRIGGER_PX (15px), triggering redundant YOLO keypoint inferences.
3. Coordinate distortion:
   add_adjust_positions_to_tracks shifted all player coordinates in the wrong direction during athlete sprints.

Key technical specifications:
- cv2.estimateAffinePartial2D with RANSAC (threshold=3.0px) for robust 2D camera pan/tilt/zoom estimation.
- Median Flow + MAD (Median Absolute Deviation) outlier rejection fallback for low-texture or degenerate matches.
- Motion deadband (< 1.2 px) to preserve static camera shots with zero jitter.
- Drop-in interface compatible with CameraMovementEstimator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False


def estimate_motion_ransac(
    old_pts: np.ndarray,
    new_pts: np.ndarray,
    reproj_threshold: float = 3.0,
    min_inlier_ratio: float = 0.35,
    min_distance: float = 1.2,
) -> Tuple[float, float, float, int]:
    """
    Estimates camera background translation (dx, dy) using RANSAC affine consensus.

    Args:
        old_pts: (N, 1, 2) or (N, 2) array of source points in frame t-1.
        new_pts: (N, 1, 2) or (N, 2) array of tracked points in frame t.
        reproj_threshold: Maximum reprojection error (px) for inliers.
        min_inlier_ratio: Minimum fraction of points required for valid RANSAC.
        min_distance: Threshold below which movement is clamped to 0.0 (static camera).

    Returns:
        (dx, dy, inlier_ratio, inlier_count)
        dx, dy: camera displacement in pixels (frame t-1 to frame t).
    """
    if old_pts is None or new_pts is None:
        return 0.0, 0.0, 0.0, 0

    pts1 = np.asarray(old_pts, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(new_pts, dtype=np.float32).reshape(-1, 2)

    n_pts = len(pts1)
    if n_pts < 3 or len(pts2) != n_pts:
        return 0.0, 0.0, 0.0, 0

    # 1. Primary path: RANSAC Affine Partial (translation + rotation + uniform scale)
    if _HAS_CV2 and n_pts >= 5:
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=reproj_threshold,
                maxIters=500,
                confidence=0.99,
            )
            if matrix is not None and inliers is not None:
                inlier_cnt = int(np.sum(inliers))
                inlier_ratio = inlier_cnt / float(n_pts)

                # Decompose matrix: scale and translation
                scale = math.hypot(matrix[0, 0], matrix[0, 1])
                tx = float(matrix[0, 2])
                ty = float(matrix[1, 2])

                # Sanity check scale: sports camera zoom is smooth, scale should be near 1.0
                if inlier_ratio >= min_inlier_ratio and 0.80 <= scale <= 1.25:
                    # In camera coordinates, if features move by (tx, ty), camera moved by (-tx, -ty)
                    # Matching legacy measure_xy_distance(old, new) convention: cx = old - new = -tx
                    cx = -tx
                    cy = -ty
                    mag = math.hypot(cx, cy)
                    if mag < min_distance:
                        return 0.0, 0.0, inlier_ratio, inlier_cnt
                    return cx, cy, inlier_ratio, inlier_cnt
        except Exception:
            pass

    # 2. Robust fallback: Median Flow with MAD Outlier Rejection
    # Displacements of static background should cluster around true camera motion
    displacements = pts1 - pts2  # old - new convention
    dx_all = displacements[:, 0]
    dy_all = displacements[:, 1]

    med_x = float(np.median(dx_all))
    med_y = float(np.median(dy_all))

    # Median Absolute Deviation (MAD)
    mad_x = float(np.median(np.abs(dx_all - med_x)))
    mad_y = float(np.median(np.abs(dy_all - med_y)))

    # Inliers within 2.5 * 1.4826 * MAD (or 2.0 px floor)
    threshold_x = max(2.0, 3.0 * mad_x)
    threshold_y = max(2.0, 3.0 * mad_y)

    inliers_mask = (np.abs(dx_all - med_x) <= threshold_x) & (np.abs(dy_all - med_y) <= threshold_y)
    inlier_cnt = int(np.sum(inliers_mask))
    inlier_ratio = inlier_cnt / float(n_pts)

    if inlier_cnt >= 2:
        robust_cx = float(np.mean(dx_all[inliers_mask]))
        robust_cy = float(np.mean(dy_all[inliers_mask]))
    else:
        robust_cx = med_x
        robust_cy = med_y

    mag = math.hypot(robust_cx, robust_cy)
    if mag < min_distance:
        return 0.0, 0.0, inlier_ratio, inlier_cnt

    return robust_cx, robust_cy, inlier_ratio, inlier_cnt


class RobustCameraMovementEstimator:
    """
    RANSAC-hardened camera motion estimator for broadcast sports footage.
    Robust against moving athletes, referee traffic, and optical noise.
    """

    def __init__(self, frame: np.ndarray, min_distance: float = 1.2):
        self.minimum_distance = float(min_distance)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) if _HAS_CV2 else None,
        )
        if _HAS_CV2 and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mask = np.zeros_like(gray)
            # Left edge: first 3% of width
            mask[:, 0:max(10, int(w * 0.03))] = 1
            # Right edge: last 3% of width
            mask[:, max(0, int(w * 0.97)):w] = 1
            # Center band: center 10%
            c_lo = int(w * 0.45)
            c_hi = int(w * 0.55)
            mask[:, c_lo:c_hi] = 1
            self.features = dict(
                maxCorners=150,
                qualityLevel=0.3,
                minDistance=4,
                blockSize=7,
                mask=mask,
            )
        else:
            self.features = {}

    @classmethod
    def from_video_path(cls, video_path: str, min_distance: float = 1.2):
        """Constructs estimator from the first frame of a video file."""
        if not _HAS_CV2:
            return cls(None, min_distance)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Cannot read first frame from: {video_path}")
        return cls(frame, min_distance)

    def estimate_step(
        self,
        old_gray: np.ndarray,
        gray: np.ndarray,
        old_pts: Optional[np.ndarray],
    ) -> Tuple[List[float], Optional[np.ndarray]]:
        """Tracks optical flow points and estimates background movement."""
        if not _HAS_CV2 or old_gray is None or gray is None:
            return [0.0, 0.0], old_pts

        if old_pts is None or len(old_pts) < 10:
            old_pts = cv2.goodFeaturesToTrack(old_gray, **self.features)
            if old_pts is None:
                return [0.0, 0.0], None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, gray, old_pts, None, **self.lk_params
        )

        if new_pts is None or status is None:
            return [0.0, 0.0], None

        good_old = old_pts[status == 1]
        good_new = new_pts[status == 1]

        if len(good_old) < 4:
            new_features = cv2.goodFeaturesToTrack(gray, **self.features)
            return [0.0, 0.0], new_features

        cx, cy, inlier_ratio, inlier_cnt = estimate_motion_ransac(
            good_old, good_new,
            reproj_threshold=3.0,
            min_inlier_ratio=0.30,
            min_distance=self.minimum_distance,
        )

        # Refresh points if tracking degrades
        if inlier_cnt < 20 or inlier_ratio < 0.40:
            next_pts = cv2.goodFeaturesToTrack(gray, **self.features)
        else:
            next_pts = good_new.reshape(-1, 1, 2)

        return [cx, cy], next_pts

    def get_camera_movement(self, frames: list, stride: int = 3) -> List[List[float]]:
        """Estimates smoothed camera movement across in-memory frames."""
        total = len(frames)
        if total == 0:
            return []

        sampled_movement: Dict[int, List[float]] = {0: [0.0, 0.0]}
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if _HAS_CV2 else None
        old_pts = cv2.goodFeaturesToTrack(old_gray, **self.features) if _HAS_CV2 and old_gray is not None else None

        for i in range(1, total, stride):
            if not _HAS_CV2:
                sampled_movement[i] = [0.0, 0.0]
                continue

            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            mv, old_pts = self.estimate_step(old_gray, gray, old_pts)
            sampled_movement[i] = mv
            old_gray = gray

        return self._interpolate_movement(sampled_movement, total)

    def get_camera_movement_streamed(
        self, video_path: str, total_frames: int, chunk_size: int = 500, stride: int = 3
    ) -> List[List[float]]:
        """Streamed estimation processing video chunks without loading all frames to RAM."""
        from .analysis_core import stream_video_chunks

        sampled_movement: Dict[int, List[float]] = {0: [0.0, 0.0]}
        old_gray = None
        old_pts = None

        for start_idx, chunk in stream_video_chunks(video_path, chunk_size):
            for local_idx, frame in enumerate(chunk):
                fi = start_idx + local_idx
                if fi >= total_frames:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if _HAS_CV2 else None

                if old_gray is None:
                    old_gray = gray
                    if _HAS_CV2 and old_gray is not None:
                        old_pts = cv2.goodFeaturesToTrack(old_gray, **self.features)
                    continue

                if fi % stride != 0:
                    continue

                mv, old_pts = self.estimate_step(old_gray, gray, old_pts)
                sampled_movement[fi] = mv
                old_gray = gray

        return self._interpolate_movement(sampled_movement, total_frames)

    def _interpolate_movement(
        self, sampled_movement: Dict[int, List[float]], total_frames: int
    ) -> List[List[float]]:
        """Linearly interpolates sparse sampled camera steps and applies rolling smoothing."""
        if total_frames <= 0:
            return []

        all_idx = np.arange(total_frames, dtype=float)
        known_idx = sorted(sampled_movement.keys())
        if len(known_idx) < 2:
            return [[0.0, 0.0] for _ in range(total_frames)]

        kx = np.array([sampled_movement[i][0] for i in known_idx], dtype=float)
        ky = np.array([sampled_movement[i][1] for i in known_idx], dtype=float)
        ki = np.array(known_idx, dtype=float)

        interp_x = np.interp(all_idx, ki, kx)
        interp_y = np.interp(all_idx, ki, ky)

        # 5-frame rolling boxcar filter for subpixel smooth panning
        w = 5
        half = w // 2
        pad_x = np.pad(interp_x, half, mode="edge")
        pad_y = np.pad(interp_y, half, mode="edge")
        smooth_x = np.convolve(pad_x, np.ones(w) / w, mode="valid")
        smooth_y = np.convolve(pad_y, np.ones(w) / w, mode="valid")

        return [[float(round(smooth_x[i], 2)), float(round(smooth_y[i], 2))] for i in range(total_frames)]

    def add_adjust_positions_to_tracks(self, tracks: dict, cam_movement: list):
        """Compensates object pixel positions by camera pan displacement."""
        for obj, otracks in tracks.items():
            for fnum, track in enumerate(otracks):
                if fnum >= len(cam_movement) or not track:
                    continue
                mv = cam_movement[fnum]
                for info in track.values():
                    pos = info.get("position")
                    if pos:
                        info["position_adjusted"] = (pos[0] - mv[0], pos[1] - mv[1])
