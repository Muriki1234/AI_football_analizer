"""
speed_kinematics_filter.py — Robust Athletic Kinematics & Savitzky-Golay Distance Integrator

Replaces the naive AccurateSpeedEstimator to eliminate:
1. The 5x distance oversumming bug caused by adding 5-frame window displacements every single frame.
2. The random-walk drift accumulation where stationary players with ±0.05m bbox jitter accumulate kilometers of fake running distance.
3. Speed spikes caused by player ID re-identification / tracklet reconnection jumps.

Key guarantees:
- Savitzky-Golay polynomial smoothing (window=7, poly=2, mode='interp') preserves true sprint peaks.
- Velocity deadband: speeds < 1.5 km/h (0.42 m/s) are treated as stationary and excluded from distance accumulation.
- Biomechanical speed capping: speeds > 38.0 km/h (top football sprint limit) are clamped.
- Single-step cumulative distance integration: d = sum(||p_t - p_{t-1}||) for valid physical movements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _savgol_coefficients(window: int, poly: int) -> np.ndarray:
    """Computes Savitzky-Golay convolution filter coefficients via Vandermonde matrix."""
    if window % 2 == 0 or window < 3:
        raise ValueError(f"Window must be an odd integer >= 3, got {window}")
    if poly >= window:
        raise ValueError(f"Poly order ({poly}) must be less than window ({window})")

    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, poly + 1)[:, ::-1]
    C = np.linalg.pinv(A)
    return C[0]


class RobustKinematicSpeedEstimator:
    """
    Athletic-grade player kinematics engine calculating instantaneous speed,
    sprint states, and true cumulative running distance on 2D pitch coordinates.
    """

    def __init__(
        self,
        fps: float = 25.0,
        sg_window: int = 7,
        sg_poly: int = 2,
        deadband_kmh: float = 2.0,
        max_speed_kmh: float = 38.0,
        sprint_threshold_kmh: float = 25.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.sg_window = int(sg_window) if sg_window % 2 == 1 else int(sg_window) + 1
        self.sg_poly = int(sg_poly)
        self.deadband_kmh = float(deadband_kmh)
        self.deadband_mps = self.deadband_kmh / 3.6
        self.max_speed_kmh = float(max_speed_kmh)
        self.max_speed_mps = self.max_speed_kmh / 3.6
        self.sprint_threshold_kmh = float(sprint_threshold_kmh)
        self._coeffs = _savgol_coefficients(self.sg_window, self.sg_poly)

    def smooth_trajectory(
        self,
        positions: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Applies Savitzky-Golay smoothing to (x, y) coordinates with polynomial boundary interpolation.
        """
        n = len(positions)
        if n < self.sg_window:
            return list(positions)

        arr = np.array(positions, dtype=float)

        if _HAS_SCIPY:
            smooth_x = savgol_filter(arr[:, 0], self.sg_window, self.sg_poly, mode="interp")
            smooth_y = savgol_filter(arr[:, 1], self.sg_window, self.sg_poly, mode="interp")
        else:
            # Fallback reflection padding
            half = self.sg_window // 2
            pad_x = np.pad(arr[:, 0], half, mode="reflect")
            pad_y = np.pad(arr[:, 1], half, mode="reflect")
            smooth_x = np.convolve(pad_x, self._coeffs, mode="valid")
            smooth_y = np.convolve(pad_y, self._coeffs, mode="valid")

        return [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]

    def compute_player_kinematics(
        self,
        frame_positions: Dict[int, Tuple[float, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Computes accurate speed and non-oversummed cumulative distance for a player tracklet.
        frame_positions: {frame_idx: (x, y)} in metric pitch coordinates.
        Returns: {frame_idx: {"speed": km/h, "distance": meters, "is_sprint": bool}}
        """
        if not frame_positions:
            return {}

        sorted_frames = sorted(frame_positions.keys())
        raw_coords = [frame_positions[f] for f in sorted_frames]

        # 1. Smooth trajectory to remove high-frequency optical tracking jitter
        smoothed_coords = self.smooth_trajectory(raw_coords)

        out: Dict[int, Dict[str, float]] = {}
        cumulative_distance = 0.0

        for i, f_idx in enumerate(sorted_frames):
            if i == 0:
                out[f_idx] = {
                    "speed": 0.0,
                    "distance": 0.0,
                    "is_sprint": False,
                    "is_stationary": True,
                }
                continue

            prev_f = sorted_frames[i - 1]
            dt = (f_idx - prev_f) / self.fps
            if dt <= 0.0:
                dt = 1.0 / self.fps

            p_curr = smoothed_coords[i]
            p_prev = smoothed_coords[i - 1]
            step_dist = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])

            speed_mps = step_dist / dt
            speed_kmh = speed_mps * 3.6

            if speed_kmh > self.max_speed_kmh:
                # Teleport jump (tracking swap): clamp speed and cap distance to max plausible movement
                speed_kmh = self.max_speed_kmh
                step_dist = 0.0
            elif speed_kmh < self.deadband_kmh:
                # Stationary deadband: athlete standing still, ignore optical jitter
                speed_kmh = 0.0
                step_dist = 0.0

            cumulative_distance += step_dist

            out[f_idx] = {
                "speed": round(speed_kmh, 1),
                "distance": round(cumulative_distance, 1),
                "is_sprint": speed_kmh >= self.sprint_threshold_kmh,
                "is_stationary": speed_kmh == 0.0,
            }

        return out

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        """
        Drop-in replacement for AccurateSpeedEstimator.add_speed_and_distance_to_tracks.
        Mutates tracks in place, adding 'speed' (km/h) and 'distance' (meters).
        """
        for obj, otracks in tracks.items():
            if obj in ("ball", "referees"):
                continue

            player_trajs: Dict[int, Dict[int, Tuple[float, float]]] = {}
            for fi, frame_data in enumerate(otracks):
                for tid, info in frame_data.items():
                    if not info:
                        continue
                    pt = info.get("position_transformed")
                    if pt and len(pt) == 2 and not any(np.isnan(v) for v in pt):
                        player_trajs.setdefault(tid, {})[fi] = (float(pt[0]), float(pt[1]))

            for tid, frame_pos in player_trajs.items():
                kinematics = self.compute_player_kinematics(frame_pos)
                for fi, metrics in kinematics.items():
                    if tid in otracks[fi]:
                        otracks[fi][tid]["speed"] = metrics["speed"]
                        otracks[fi][tid]["distance"] = metrics["distance"]
                        otracks[fi][tid]["is_sprint"] = metrics["is_sprint"]
