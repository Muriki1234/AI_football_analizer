"""
homography_stabilizer.py - Robust Pitch Homography Stabilizer & Keypoint Outlier Rejection

Architectural Foundations:
1. RANSAC Inlier Strict Pruning: Eliminates keypoint misdetections from raw YOLO outputs.
2. Canonical Anchor Projection Smoothing (BroadTrack & SoccerNet SOTA):
   Avoids non-Euclidean direct matrix averaging by projecting fixed pitch corner anchors
   to Euclidean image coordinates, smoothing the image anchors via adaptive EMA, and
   re-estimating a conditioned planar homography.
3. Degeneracy & Condition Number Gate: Rejects ill-conditioned, negative-determinant,
   or collinear matrices to prevent minimap player teleportation.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class HomographyConditioner:
    """
    Performs keypoint geometry validation, strict RANSAC inlier filtering,
    and projective condition-number screening.
    """

    def __init__(
        self,
        min_points: int = 6,
        min_x_span: float = 3000.0,
        min_y_span: float = 1500.0,
        ransac_thresh: float = 400.0,
        min_inlier_ratio: float = 0.5,
        max_condition_number: float = 1e5,
    ):
        self.min_points = min_points
        self.min_x_span = min_x_span
        self.min_y_span = min_y_span
        self.ransac_thresh = ransac_thresh
        self.min_inlier_ratio = min_inlier_ratio
        self.max_condition_number = max_condition_number

    def estimate_conditioned_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Estimates a conditioned homography mapping src (image pixels) to dst (pitch coords).
        Returns (H_matrix, diagnostics).
        """
        diag: Dict[str, Any] = {
            "total_points": len(src_points) if src_points is not None else 0,
            "inliers_count": 0,
            "inlier_ratio": 0.0,
            "status": "REJECTED",
            "reason": "",
        }

        if src_points is None or len(src_points) < self.min_points:
            diag["reason"] = f"Insufficient points: {len(src_points) if src_points is not None else 0} < {self.min_points}"
            return None, diag

        src_arr = np.asarray(src_points, dtype=np.float32)
        dst_arr = np.asarray(dst_points, dtype=np.float32)

        x_span = dst_arr[:, 0].max() - dst_arr[:, 0].min()
        y_span = dst_arr[:, 1].max() - dst_arr[:, 1].min()
        if x_span < self.min_x_span or y_span < self.min_y_span:
            diag["reason"] = f"Spatial span too small: x_span={x_span:.0f}, y_span={y_span:.0f}"
            return None, diag

        # 1. RANSAC to detect inliers
        H_ransac, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, self.ransac_thresh)
        if H_ransac is None or mask is None:
            diag["reason"] = "RANSAC failed to find valid initial homography"
            return None, diag

        inlier_mask = mask.ravel() == 1
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count) / len(src_arr)

        diag["inliers_count"] = inlier_count
        diag["inlier_ratio"] = round(inlier_ratio, 3)

        if inlier_count < 4 or inlier_ratio < self.min_inlier_ratio:
            diag["reason"] = f"Low inlier ratio: {inlier_ratio:.2f} < {self.min_inlier_ratio} (inliers={inlier_count})"
            return None, diag

        # 2. Re-fit ONLY on inliers (crucial: purge noisy outliers)
        src_clean = src_arr[inlier_mask]
        dst_clean = dst_arr[inlier_mask]
        H_clean, _ = cv2.findHomography(src_clean, dst_clean, 0)
        if H_clean is None:
            diag["reason"] = "Inlier re-fitting failed"
            return None, diag

        # 3. Geometric Condition check: determinant and normalized Hartley condition number
        det = np.linalg.det(H_clean[:2, :2])
        if det <= 0:
            diag["reason"] = f"Degenerate determinant ({det:.4e} <= 0) indicates flipped orientation"
            return None, diag

        # Normalize coordinates by domain scales (image: 1920x1080, pitch: 12000x7000)
        T_img = np.diag([1.0 / 1920.0, 1.0 / 1080.0, 1.0])
        T_pitch = np.diag([1.0 / 12000.0, 1.0 / 7000.0, 1.0])
        H_normalized = T_pitch @ H_clean @ np.linalg.inv(T_img)
        if H_normalized[2, 2] != 0:
            H_normalized = H_normalized / H_normalized[2, 2]

        s = np.linalg.svd(H_normalized, compute_uv=False)
        cond_num = float(s[0] / (s[-1] + 1e-12))
        diag["condition_number"] = cond_num
        if cond_num > self.max_condition_number:
            diag["reason"] = f"Ill-conditioned matrix: normalized cond={cond_num:.2e} > {self.max_condition_number:.2e}"
            return None, diag

        diag["status"] = "ACCEPTED"
        return H_clean, diag


class CanonicalAnchorSmoother:
    """
    Maintains temporal consistency by filtering 4 canonical virtual pitch anchors
    in Euclidean image space (p_img = H^-1 * P_pitch) rather than averaging non-Euclidean matrices.
    """

    def __init__(
        self,
        pitch_width: float = 12000.0,
        pitch_height: float = 7000.0,
        alpha: float = 0.25,
        motion_threshold: float = 30.0,
    ):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.default_alpha = alpha
        self.motion_threshold = motion_threshold

        # 4 canonical corners in pitch space
        self.canonical_pitch_anchors = np.array([
            [0.0, 0.0],
            [pitch_width, 0.0],
            [pitch_width, pitch_height],
            [0.0, pitch_height],
        ], dtype=np.float32)

        self.smoothed_image_anchors: Optional[np.ndarray] = None
        self.last_valid_H: Optional[np.ndarray] = None

    def update(self, H_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Updates temporal smoother with newly measured raw homography H_raw.
        Returns stabilized H_smooth.
        """
        if H_raw is None:
            return self.last_valid_H

        try:
            H_inv = np.linalg.inv(H_raw)
        except np.linalg.LinAlgError:
            return self.last_valid_H

        # Project canonical pitch anchors into image plane: p_img = H^-1 * P_pitch
        pitch_anchors_homo = np.hstack([
            self.canonical_pitch_anchors,
            np.ones((4, 1), dtype=np.float32)
        ])
        projected = (H_inv @ pitch_anchors_homo.T).T
        image_anchors = projected[:, :2] / (projected[:, 2:3] + 1e-12)

        # Initial anchor bootstrap
        if self.smoothed_image_anchors is None:
            self.smoothed_image_anchors = image_anchors.copy()
            self.last_valid_H = H_raw.copy()
            return self.last_valid_H

        # Adaptive alpha: if camera pan/zoom is fast, increase alpha to prevent lagging
        anchor_motion = float(np.mean(np.linalg.norm(image_anchors - self.smoothed_image_anchors, axis=1)))
        if anchor_motion > self.motion_threshold:
            effective_alpha = min(0.8, self.default_alpha * 2.5)
        else:
            effective_alpha = self.default_alpha

        # Euclidean EMA smoothing on image coordinates
        self.smoothed_image_anchors = (
            effective_alpha * image_anchors + (1.0 - effective_alpha) * self.smoothed_image_anchors
        )

        # Re-fit homography from smoothed image anchors to canonical pitch anchors
        H_smooth, _ = cv2.findHomography(
            self.smoothed_image_anchors,
            self.canonical_pitch_anchors,
            0
        )
        if H_smooth is not None:
            self.last_valid_H = H_smooth
            return H_smooth

        return self.last_valid_H

    def reset(self):
        self.smoothed_image_anchors = None
        self.last_valid_H = None


from server.pipeline.keypoint_gating import HomographyCache


class TemporalHomographyStabilizer:
    """
    End-to-end coordinator integrating HomographyConditioner + CanonicalAnchorSmoother.
    Processes video frame sequences, purges outliers, smooths jitter, and transforms tracks.
    """

    def __init__(
        self,
        min_grass_ratio: float = 0.28,
        min_points: int = 6,
        alpha: float = 0.25,
        enable_cache: bool = True,
        cache_tolerance: float = 0.85,
    ):
        self.conditioner = HomographyConditioner(min_points=min_points)
        self.smoother = CanonicalAnchorSmoother(alpha=alpha)
        self.enable_cache = enable_cache
        self.cache = HomographyCache(tolerance_px=cache_tolerance) if enable_cache else None

    def process_frame(
        self,
        src_points: Optional[np.ndarray],
        dst_points: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Processes a single frame keypoint set, returns (H_stabilized, diagnostics).
        """
        if self.enable_cache and self.cache is not None and src_points is not None:
            cached_res = self.cache.get(src_points, dst_points)
            if cached_res is not None:
                H_cached, diag_cached = cached_res
                diag_cached["smoothed"] = True
                return H_cached, diag_cached

        H_raw, diag = self.conditioner.estimate_conditioned_homography(src_points, dst_points)
        H_smooth = self.smoother.update(H_raw)
        diag["smoothed"] = (H_smooth is not None)
        diag["cache_hit"] = False

        if self.enable_cache and self.cache is not None and H_smooth is not None and src_points is not None:
            self.cache.put(src_points, dst_points, H_smooth, diag)

        return H_smooth, diag

    def transform_points(
        self,
        H: Optional[np.ndarray],
        points: np.ndarray,
        clamp_to_pitch: bool = True,
    ) -> np.ndarray:
        """
        Transforms 2D points (e.g. player feet) from image space to pitch coordinates.
        """
        if H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        transformed = (H @ pts_homo.T).T
        out = transformed[:, :2] / (transformed[:, 2:3] + 1e-12)

        if clamp_to_pitch:
            # Margin of 500 units (~4m) outside touchlines
            out[:, 0] = np.clip(out[:, 0], -500.0, 12500.0)
            out[:, 1] = np.clip(out[:, 1], -500.0, 7500.0)

        return out

    def reset(self) -> None:
        """Resets smoother and homography cache state."""
        self.smoother.reset()
        if self.cache is not None:
            self.cache.reset()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns cache lookup, hit, and ratio statistics."""
        if self.cache is not None:
            return self.cache.get_stats()
        return {"lookups": 0, "hits": 0, "misses": 0, "hit_ratio": 0.0}

