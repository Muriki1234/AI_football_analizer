"""
camera_motion_compensator.py
============================
Lightweight Background-Gated Affine Camera Motion Compensator (GMC)

Methodology:
1. Feature Extraction: Fast Shi-Tomasi corner detector on background regions.
2. Player Masking: Excludes all bounding boxes of detected players to ensure camera motion
   reflects true pan/tilt/zoom rather than foreground player motion.
3. Lucas-Kanade Optical Flow + RANSAC:
   Tracks background keypoints from frame t-1 to frame t, fitting an affine transformation matrix:
   A = [[cos(theta)*s, -sin(theta)*s, tx],
        [sin(theta)*s,  cos(theta)*s, ty]]
4. Spatial Compensation:
   - Warps predicted ball coordinates: p_comp = A @ [px, py, 1]
   - Propagates stale homography matrices: H_t = H_{t-1} @ inv(A_3x3)
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple


class CameraMotionCompensator:
    """
    Computes inter-frame camera motion affine transformation with foreground masking.
    """

    def __init__(
        self,
        max_corners: int = 150,
        quality_level: float = 0.01,
        min_distance: float = 15.0,
        ransac_thresh: float = 3.0,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_thresh = ransac_thresh

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None

    def estimate_camera_motion(
        self,
        curr_frame: np.ndarray,
        player_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Estimates the 2x3 affine transformation matrix from previous frame to current frame.
        Returns:
            (affine_matrix_2x3, is_valid)
            If invalid or first frame, returns identity matrix [[1, 0, 0], [0, 1, 0]] and False.
        """
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
        identity_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_pts = self._find_background_features(curr_gray, player_boxes)
            return identity_affine, False

        # If previous points were lost, re-extract
        if self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_pts = self._find_background_features(self.prev_gray, player_boxes)
            if self.prev_pts is None or len(self.prev_pts) < 8:
                self.prev_gray = curr_gray
                return identity_affine, False

        # Track features using Lucas-Kanade optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        good_prev = self.prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 8:
            self.prev_gray = curr_gray
            self.prev_pts = self._find_background_features(curr_gray, player_boxes)
            return identity_affine, False

        # Robust RANSAC Partial Affine fit (Translation + Rotation + Uniform Scale)
        affine_mat, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=1000,
        )

        is_valid = False
        if affine_mat is not None and inliers is not None:
            n_inliers = int(inliers.sum())
            if n_inliers >= 8 and (n_inliers / len(good_prev)) >= 0.40:
                is_valid = True
                identity_affine = affine_mat

        # Advance state
        self.prev_gray = curr_gray
        self.prev_pts = self._find_background_features(curr_gray, player_boxes)

        return identity_affine, is_valid

    def _find_background_features(
        self,
        gray: np.ndarray,
        player_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Optional[np.ndarray]:
        """Extracts corners while masking out player bounding boxes."""
        h, w = gray.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if player_boxes:
            for b in player_boxes:
                x1 = int(max(0, b[0] - 5))
                y1 = int(max(0, b[1] - 5))
                x2 = int(min(w, b[2] + 5))
                y2 = int(min(h, b[3] + 5))
                mask[y1:y2, x1:x2] = 0

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        return pts

    @staticmethod
    def warp_point(pt: Tuple[float, float], affine_mat: np.ndarray) -> Tuple[float, float]:
        """Applies affine transformation to a 2D point (x, y)."""
        x, y = pt
        wx = affine_mat[0, 0] * x + affine_mat[0, 1] * y + affine_mat[0, 2]
        wy = affine_mat[1, 0] * x + affine_mat[1, 1] * y + affine_mat[1, 2]
        return float(wx), float(wy)

    @staticmethod
    def propagate_homography(H_prev: np.ndarray, affine_mat: np.ndarray) -> np.ndarray:
        """
        Propagates homography H_prev across camera motion affine_mat:
        Let x_curr = A @ x_prev. Then x_prev = inv(A) @ x_curr.
        H_curr @ x_curr = H_prev @ x_prev = H_prev @ inv(A) @ x_curr.
        Therefore: H_curr = H_prev @ inv(A_3x3).
        """
        A_3x3 = np.eye(3, dtype=np.float64)
        A_3x3[:2, :] = affine_mat.astype(np.float64)
        try:
            inv_A = np.linalg.inv(A_3x3)
            H_curr = np.dot(H_prev.astype(np.float64), inv_A)
            if abs(H_curr[2, 2]) > 1e-8:
                H_curr /= H_curr[2, 2]
            return H_curr
        except Exception:
            return H_prev
