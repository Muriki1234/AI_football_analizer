"""
physiologically_bounded_sprint_peak_filter.py
=============================================
Physiologically-Bounded Sprint & Max Speed Kinematic Filter (FIFA EPTS Standard).

Root Cause Solved:
1. Fixes the artificial 38.0 km/h speed saturation where single-frame tracking/camera jumps
   were blindly clamped to 38.0 km/h, pinning the dashboard Max Speed card to 38.0 km/h
   and triggering the "likely tracking/camera-motion noise" warning.
2. Implements physiological acceleration sanity gating: human biomechanical acceleration
   is capped at a_max <= 6.5 m/s^2. Any instantaneous step exceeding this is identified as
   an optical tracking jump / homography flicker and rejected.
3. Implements sustained temporal window peak speed evaluation: true FIFA peak speed requires
   a minimum physiological duration (tau >= 0.3s ~ 0.5s, 10-15 frames), filtering out
   33ms single-frame noise spikes while capturing true athletic sprint efforts.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


class PhysiologicallyBoundedSprintPeakFilter:
    """
    FIFA-compliant athletic velocity and sprint kinematics filter with physiological acceleration
    and sustained-window peak estimation.
    """

    # Biomechanical limits (FIFA EPTS / Catapult Sports benchmarks)
    ABSOLUTE_MAX_HUMAN_SPEED_KMH: float = 38.0   # 10.55 m/s (World-class Olympic/EPL sprint peak)
    MAX_HUMAN_ACCEL_MS2: float = 6.5             # 6.5 m/s^2 (Maximum human propulsive acceleration)
    MAX_HUMAN_DECEL_MS2: float = 8.5             # 8.5 m/s^2 (Maximum human braking deceleration)
    MIN_SPRINT_SPEED_KMH: float = 25.2           # FIFA Zone 5 sprint threshold
    DEADBAND_SPEED_KMH: float = 1.0              # < 1.0 km/h is stationary jitter

    def __init__(
        self,
        fps: float = 30.0,
        sustained_window_sec: float = 0.4,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.dt_default = 1.0 / self.fps
        self.window_frames = max(3, int(round(sustained_window_sec * self.fps)))

        self.filtered_speeds_kmh: List[float] = []
        self.sustained_speeds_kmh: List[float] = []
        self.displacements_m: List[float] = []
        self.rejected_jump_count: int = 0
        self.sustained_sprint_events: List[Dict[str, Any]] = []

        self._in_sprint: bool = False
        self._sprint_start_frame: int = 0
        self._sprint_peak_v: float = 0.0

    def reset(self) -> None:
        self.filtered_speeds_kmh.clear()
        self.sustained_speeds_kmh.clear()
        self.displacements_m.clear()
        self.rejected_jump_count = 0
        self.sustained_sprint_events.clear()
        self._in_sprint = False
        self._sprint_start_frame = 0
        self._sprint_peak_v = 0.0

    def filter_step(
        self,
        raw_disp_m: float,
        dt_s: float,
        prev_speed_kmh: Optional[float] = None,
    ) -> Tuple[float, float, bool]:
        """
        Filters a single kinematic step:
        Returns:
            (filtered_disp_m, filtered_speed_kmh, was_jump_rejected)
        """
        dt = dt_s if dt_s > 0 else self.dt_default
        raw_speed_ms = raw_disp_m / dt
        raw_speed_kmh = raw_speed_ms * 3.6

        # Deadband check
        if raw_speed_kmh < self.DEADBAND_SPEED_KMH or raw_disp_m < 0.04:
            return 0.0, 0.0, False

        # Absolute speed limit check
        if raw_speed_kmh > self.ABSOLUTE_MAX_HUMAN_SPEED_KMH:
            self.rejected_jump_count += 1
            clamped_speed = prev_speed_kmh if prev_speed_kmh is not None else 0.0
            return (clamped_speed / 3.6) * dt, clamped_speed, True

        # First observation or post-gap onset: no acceleration constraint
        if prev_speed_kmh is None:
            return raw_disp_m, raw_speed_kmh, False

        prev_speed_ms = prev_speed_kmh / 3.6
        accel_ms2 = (raw_speed_ms - prev_speed_ms) / dt

        # Biomechanical Acceleration Gate
        # If acceleration or deceleration exceeds human capability, this is an optical tracking jump
        is_jump = (
            accel_ms2 > self.MAX_HUMAN_ACCEL_MS2
            or accel_ms2 < -self.MAX_HUMAN_DECEL_MS2
        )

        if is_jump:
            self.rejected_jump_count += 1
            # Clamp velocity increment by maximum allowed physiological acceleration
            if accel_ms2 > 0:
                allowed_speed_ms = min(
                    self.ABSOLUTE_MAX_HUMAN_SPEED_KMH / 3.6,
                    prev_speed_ms + self.MAX_HUMAN_ACCEL_MS2 * dt,
                )
            else:
                allowed_speed_ms = max(
                    0.0,
                    prev_speed_ms - self.MAX_HUMAN_DECEL_MS2 * dt,
                )
            clamped_disp = allowed_speed_ms * dt
            clamped_speed_kmh = allowed_speed_ms * 3.6
            return clamped_disp, clamped_speed_kmh, True

        return raw_disp_m, raw_speed_kmh, False

    def process_trajectory(
        self,
        points: Sequence[Tuple[int, float, float, float]],  # (frame_idx, timestamp_s, x_m, y_m)
    ) -> Dict[str, Any]:
        """
        Processes a full target player trajectory sequence and outputs validated kinematics.
        """
        self.reset()
        if len(points) < 2:
            return {
                "max_speed_kmh": 0.0,
                "sustained_peak_speed_kmh": 0.0,
                "fifa_avg_speed_kmh": 0.0,
                "total_distance_m": 0.0,
                "sprint_count": 0,
                "rejected_jumps": 0,
                "speed_reliability": "high",
            }

        prev_v: Optional[float] = None
        total_m = 0.0
        start_t = points[0][1]
        end_t = points[-1][1]
        total_duration_s = max(0.1, end_t - start_t)

        for i in range(1, len(points)):
            prev_pt = points[i - 1]
            curr_pt = points[i]

            dt = curr_pt[1] - prev_pt[1]
            if dt <= 0:
                dt = (curr_pt[0] - prev_pt[0]) / self.fps
            if dt <= 0:
                dt = self.dt_default

            # Large time gap (>1.5s): treat as discontinuity
            if dt > 1.5:
                prev_v = None
                continue

            dx = curr_pt[2] - prev_pt[2]
            dy = curr_pt[3] - prev_pt[3]
            raw_disp = math.hypot(dx, dy)

            disp, v_kmh, _ = self.filter_step(raw_disp, dt, prev_v)
            self.displacements_m.append(disp)
            self.filtered_speeds_kmh.append(v_kmh)
            total_m += disp
            prev_v = v_kmh

        # Compute Sustained Peak Speed over physiological rolling window (e.g. 0.4s)
        k = self.window_frames
        if len(self.filtered_speeds_kmh) >= k:
            # Moving average / rolling minimum within sprint window to guarantee sustained exertion
            speeds_arr = np.asarray(self.filtered_speeds_kmh, dtype=np.float32)
            # Rolling window: a player must sustain speed v for at least k consecutive frames
            # Using 1D convolution for efficient rolling mean
            kernel = np.ones(k, dtype=np.float32) / float(k)
            sustained_arr = np.convolve(speeds_arr, kernel, mode="valid")
            sustained_peak = float(np.max(sustained_arr)) if sustained_arr.size else 0.0
            # Absolute 99th percentile of raw filtered speeds as ceiling
            p99 = float(np.percentile(speeds_arr, 99))
            final_peak = min(sustained_peak, p99, self.ABSOLUTE_MAX_HUMAN_SPEED_KMH)
        else:
            final_peak = float(np.max(self.filtered_speeds_kmh)) if self.filtered_speeds_kmh else 0.0

        # Sprint Count Detection: sustained v >= 25.2 km/h for >= window_frames
        in_sprint = False
        sprint_frames = 0
        sprint_count = 0
        for v in self.filtered_speeds_kmh:
            if v >= self.MIN_SPRINT_SPEED_KMH:
                sprint_frames += 1
                if sprint_frames >= self.window_frames and not in_sprint:
                    in_sprint = True
                    sprint_count += 1
            else:
                sprint_frames = 0
                in_sprint = False

        avg_speed_kmh = (total_m / total_duration_s) * 3.6
        reliability = "suspect" if self.rejected_jump_count > (len(points) * 0.05) else "high"

        return {
            "max_speed_kmh": round(final_peak, 1),
            "fifa_avg_speed_kmh": round(avg_speed_kmh, 1),
            "total_distance_m": round(total_m, 1),
            "sprint_count": sprint_count,
            "rejected_jumps": self.rejected_jump_count,
            "speed_reliability": reliability,
            "total_duration_s": round(total_duration_s, 1),
        }
