"""
pitch_roi_crop_detector.py - Dynamic Pitch ROI Cropping & Non-Pitch Elimination

Key Computer Vision Principle:
In standard broadcast football footage (1080p, 1920x1080):
- The upper 15%~25% of the frame captures spectator stands, roof trusses, advertising scoreboards,
  and sky, where zero actual gameplay occurs.
- Forwarding these non-pitch pixels to the object detector wastes ~20-25% of neural network FLOPs
  and generates false-positive player detections among the crowd.
- By computing the active pitch envelope from detected keypoints or green grass segmentation:
  1. The frame is dynamically sliced to [y_min - pad, y_max + pad], aligned to standard YOLO 32px strides.
  2. Bounding boxes detected in the sliced crop are shifted back to full-frame space via lossless offset addition.
  3. Spurious detections in the crowd stands are automatically eliminated.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PitchROICropDetector:
    """
    Computes active pitch bounding envelope and orchestrates cropped YOLO inference.
    """

    def __init__(
        self,
        min_crop_height: int = 480,
        pad_px: int = 48,
        stride_alignment: int = 32,
    ):
        self.min_crop_height = min_crop_height
        self.pad_px = pad_px
        self.stride_alignment = stride_alignment
        self.cached_roi: Optional[Tuple[int, int]] = None

    def estimate_pitch_vertical_bounds(
        self,
        frame: np.ndarray,
        keypoints: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[int, int]:
        """
        Estimates [y_min, y_max] bounding lines of the pitch.
        Prioritizes verified field keypoints; falls back to green grass color thresholding.
        """
        h, w = frame.shape[:2]

        # 1. Keypoint-based envelope if available
        if keypoints and len(keypoints) >= 4:
            valid_ys = [pt[1] for pt in keypoints.values() if 0 <= pt[1] < h]
            if len(valid_ys) >= 4:
                y_min = max(0, int(min(valid_ys)) - self.pad_px)
                y_max = min(h, int(max(valid_ys)) + self.pad_px)
                return self._align_bounds(y_min, y_max, h)

        # 2. Fast HSV grass segmentation fallback
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([28, 25, 25]), np.array([88, 255, 255]))
        row_density = np.mean(mask > 0, axis=1) # shape: (90,)

        # Find rows where grass covers at least 15% of the horizontal span
        grass_rows = np.where(row_density >= 0.15)[0]
        if len(grass_rows) > 0:
            scale_y = h / 90.0
            y_min = max(0, int(grass_rows[0] * scale_y) - self.pad_px)
            y_max = min(h, int(grass_rows[-1] * scale_y) + self.pad_px)
            return self._align_bounds(y_min, y_max, h)

        # 3. Default safe envelope (skip top 15% stands)
        return self._align_bounds(int(h * 0.12), h, h)

    def _align_bounds(self, y_min: int, y_max: int, total_h: int) -> Tuple[int, int]:
        """Ensures height meets min_crop_height and is aligned to stride_alignment."""
        crop_h = y_max - y_min
        if crop_h < self.min_crop_height:
            needed = self.min_crop_height - crop_h
            y_min = max(0, y_min - needed // 2)
            y_max = min(total_h, y_min + self.min_crop_height)

        # Align height to stride
        actual_h = y_max - y_min
        rem = actual_h % self.stride_alignment
        if rem != 0:
            pad_add = self.stride_alignment - rem
            if y_max + pad_add <= total_h:
                y_max += pad_add
            elif y_min - pad_add >= 0:
                y_min -= pad_add

        return (y_min, y_max)

    def crop_frame(self, frame: np.ndarray, y_min: int, y_max: int) -> np.ndarray:
        """Crops frame along the vertical pitch bounds."""
        return frame[y_min:y_max, :]

    def restore_boxes(
        self,
        boxes_xyxy: np.ndarray,
        y_offset: int,
    ) -> np.ndarray:
        """
        Shifts bounding boxes detected in the cropped strip back to full-frame coordinates.
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.empty((0, 4), dtype=float)

        restored = boxes_xyxy.copy()
        restored[:, 1] += float(y_offset)
        restored[:, 3] += float(y_offset)
        return restored
