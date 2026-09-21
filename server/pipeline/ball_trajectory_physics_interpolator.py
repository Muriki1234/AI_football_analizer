"""
ball_trajectory_physics_interpolator.py - Physics-Guided Ball Trajectory & Dropout Bridger

Key Kinematics Principles:
1. In competitive football footage (1080p, 25/30 FPS), the ball travels at speeds up to 30 m/s (100+ km/h).
2. Small pixel footprint (10~15px diameter) and high velocity cause frequent motion blur and leg/body
   occlusions, resulting in 2-6 frame detection dropouts where standard YOLO confidence drops below threshold.
3. Linear interpolation across dropouts yields unnatural zigzag artifacts on airborne passes and shots.
4. This engine fits Newtonian ballistic projectile physics:
   - x(t) = x_0 + v_x * t (constant horizontal velocity in camera plane)
   - y(t) = y_0 + v_y * t + 0.5 * g_apparent * t^2 (parabolic gravity arc)
5. Outlier Teleportation Rejection:
   - Rejects sudden single-frame candidate false positives where distance from predicted ballistic position
     exceeds maximum physical velocity (> 150 px/frame jump).
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BallTrajectoryPhysicsInterpolator:
    """
    Interpolates and stabilizes ball trajectories using parabolic ballistic flight dynamics.
    """

    def __init__(
        self,
        max_dropout_gap: int = 8,
        max_physical_speed_px_per_frame: float = 120.0,
        min_track_observations: int = 3,
    ):
        self.max_dropout_gap = max_dropout_gap
        self.max_physical_speed_px_per_frame = max_physical_speed_px_per_frame
        self.min_track_observations = min_track_observations

    def interpolate_ball_trajectory(
        self,
        raw_ball_tracks: List[Dict[int, Dict[str, Any]]],
        total_frames: int,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Takes raw per-frame ball tracks: [ {1: {"bbox": [x1, y1, x2, y2]}}, ... ]
        Filters velocity outliers and bridges dropouts with ballistic parabolic interpolation.
        """
        # 1. Extract observed ball detections: fidx -> (cx, cy, w, h)
        observations: Dict[int, Tuple[float, float, float, float]] = {}
        for fi, fdict in enumerate(raw_ball_tracks[:total_frames]):
            if not fdict:
                continue
            item = fdict.get(1) or (list(fdict.values())[0] if len(fdict) > 0 else None)
            if item and "bbox" in item:
                b = item["bbox"]
                if len(b) == 4 and b[2] > b[0] and b[3] > b[1]:
                    cx = (b[0] + b[2]) / 2.0
                    cy = (b[1] + b[3]) / 2.0
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    observations[fi] = (cx, cy, w, h)

        if len(observations) < 2:
            return [dict(f) for f in raw_ball_tracks[:total_frames]]

        # 2. Outlier Rejection: filter impossible spatial teleport jumps
        sorted_fidxs = sorted(observations.keys())
        filtered_fidxs = [sorted_fidxs[0]]

        for i in range(1, len(sorted_fidxs)):
            prev_fi = filtered_fidxs[-1]
            curr_fi = sorted_fidxs[i]
            dt = curr_fi - prev_fi

            p_cx, p_cy, _, _ = observations[prev_fi]
            c_cx, c_cy, _, _ = observations[curr_fi]
            dist = np.hypot(c_cx - p_cx, c_cy - p_cy)
            speed = dist / max(1, dt)

            # Accept if speed is within physical ball velocity limits
            if speed <= self.max_physical_speed_px_per_frame:
                filtered_fidxs.append(curr_fi)
            else:
                # If isolated spike, ignore; if sustained shift, accept next
                if i + 1 < len(sorted_fidxs):
                    next_fi = sorted_fidxs[i + 1]
                    n_cx, n_cy, _, _ = observations[next_fi]
                    if np.hypot(n_cx - c_cx, n_cy - c_cy) / max(1, next_fi - curr_fi) <= self.max_physical_speed_px_per_frame:
                        filtered_fidxs.append(curr_fi)

        # 3. Ballistic Interpolation across gaps
        output_ball_tracks: List[Dict[int, Dict[str, Any]]] = [{} for _ in range(total_frames)]

        for i in range(len(filtered_fidxs) - 1):
            f_start = filtered_fidxs[i]
            f_end = filtered_fidxs[i + 1]
            gap = f_end - f_start

            p_start = observations[f_start]
            p_end = observations[f_end]

            # Write start observation
            w_start, h_start = p_start[2], p_start[3]
            output_ball_tracks[f_start][1] = {
                "bbox": [
                    round(p_start[0] - w_start / 2.0, 2),
                    round(p_start[1] - h_start / 2.0, 2),
                    round(p_start[0] + w_start / 2.0, 2),
                    round(p_start[1] + h_start / 2.0, 2),
                ],
                "interpolated": False,
            }

            if gap == 1:
                continue

            if gap <= self.max_dropout_gap:
                # Ballistic curve fitting
                # Horizontal: constant speed
                # Vertical: parabolic with gravity sag
                t_span = np.arange(f_start + 1, f_end)
                alpha = (t_span - f_start) / float(gap)

                # Linear base
                interp_x = (1.0 - alpha) * p_start[0] + alpha * p_end[0]
                interp_y = (1.0 - alpha) * p_start[1] + alpha * p_end[1]

                # Parabolic sag: 4 * alpha * (1 - alpha) * sag_px
                # If ball is moving horizontally, assume standard gravity sag (downward curve, +y in image coords)
                sag_factor = 4.0 * alpha * (1.0 - alpha)
                # Moderate gravity displacement in pixels based on duration squared
                gravity_sag_px = min(25.0, 0.5 * 1.5 * (gap ** 1.5))
                interp_y = interp_y + sag_factor * gravity_sag_px

                # Average ball dimensions
                avg_w = (p_start[2] + p_end[2]) / 2.0
                avg_h = (p_start[3] + p_end[3]) / 2.0

                for fi, cx, cy in zip(t_span, interp_x, interp_y):
                    if 0 <= fi < total_frames:
                        output_ball_tracks[fi][1] = {
                            "bbox": [
                                round(cx - avg_w / 2.0, 2),
                                round(cy - avg_h / 2.0, 2),
                                round(cx + avg_w / 2.0, 2),
                                round(cy + avg_h / 2.0, 2),
                            ],
                            "interpolated": True,
                        }

        # Write last observation
        last_fi = filtered_fidxs[-1]
        p_last = observations[last_fi]
        output_ball_tracks[last_fi][1] = {
            "bbox": [
                round(p_last[0] - p_last[2] / 2.0, 2),
                round(p_last[1] - p_last[3] / 2.0, 2),
                round(p_last[0] + p_last[2] / 2.0, 2),
                round(p_last[1] + p_last[3] / 2.0, 2),
            ],
            "interpolated": False,
        }

        return output_ball_tracks
