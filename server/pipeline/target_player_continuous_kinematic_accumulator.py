"""
target_player_continuous_kinematic_accumulator.py
=================================================
Continuous Tracklet Kinematic Distance & FIFA Speed Accumulator.

Solves the systemic distance and speed estimation flaws:
1. Replaces the broken `max(distances) - min(distances)` sawtooth drop with
   continuous step-by-step physical displacement accumulation across arbitrary
   ByteTrack ID switches.
2. Replaces the 5x over-counting moving window summation with proper temporal
   differentiation.
3. Implements homography micro-jitter deadbands (<4cm/frame, <1.0 km/h) to
   prevent stationary players from accumulating ghost kilometers.
4. Calculates true FIFA / Catapult match average speed (Total Distance / Match Time)
   alongside active moving speed (v >= 1.0 km/h).
5. Categorizes athletic exertion into standard FIFA 5-zone distance breakdowns
   (Walking, Jogging, Running, High-Speed Running, Sprinting) with sprint count.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class KinematicPoint:
    frame_idx: int
    timestamp_s: float
    x_m: float
    y_m: float
    track_id: Optional[int] = None
    confidence: float = 1.0
    v_kmh: float = 0.0
    is_gap: bool = False
    is_clamped: bool = False


class TargetPlayerContinuousKinematicAccumulator:
    """
    Continuous physical trajectory kinematic accumulator for target player analysis.
    """

    # Physical Human Football Bounds
    MAX_VALID_SPEED_KMH: float = 38.0      # 10.55 m/s (World-class sprint peak)
    MAX_VALID_ACCEL_MS2: float = 6.5       # 6.5 m/s^2 maximum human acceleration
    DEADBAND_SPEED_KMH: float = 1.0        # < 1.0 km/h (0.28 m/s) is stationary noise
    DEADBAND_DISP_M: float = 0.04          # 4 cm per frame min displacement
    MAX_CONTINUOUS_GAP_S: float = 1.5      # Gaps > 1.5s are treated as discontinuous

    # FIFA 5-Zone Speed Thresholds (km/h)
    ZONE_WALKING_MAX: float = 7.2          # 0.0 - 7.2 km/h
    ZONE_JOGGING_MAX: float = 14.4         # 7.2 - 14.4 km/h
    ZONE_RUNNING_MAX: float = 19.8         # 14.4 - 19.8 km/h
    ZONE_HSR_MAX: float = 25.2             # 19.8 - 25.2 km/h (High-speed running)
    # Zone 5 Sprinting: >= 25.2 km/h

    def __init__(self, fps: float = 30.0, smoothing_window: int = 5) -> None:
        self.fps = fps
        self.smoothing_window = smoothing_window

        self.history: List[KinematicPoint] = []
        self.total_distance_m: float = 0.0
        self.total_duration_s: float = 0.0
        self.active_moving_duration_s: float = 0.0

        # Zone Distances (meters)
        self.zone_walking_m: float = 0.0
        self.zone_jogging_m: float = 0.0
        self.zone_running_m: float = 0.0
        self.zone_hsr_m: float = 0.0
        self.zone_sprinting_m: float = 0.0

        # Sprint Event Tracking
        self.sprint_count: int = 0
        self._current_sprint_frames: int = 0
        self.min_sprint_duration_frames: int = max(3, int(round(fps * 0.5)))  # >= 0.5s

        # Anomaly counters
        self.jitter_suppressed_count: int = 0
        self.velocity_clamped_count: int = 0
        self.gap_count: int = 0

    def add_point(
        self,
        frame_idx: int,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        track_id: Optional[int] = None,
        confidence: float = 1.0,
    ) -> KinematicPoint:
        """
        Add a projected pitch coordinate observation (x_m, y_m).
        Calculates instantaneous step displacement, applies jitter deadband and velocity clamps,
        and accumulates physical distance.
        """
        if not self.history:
            pt = KinematicPoint(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                x_m=x_m,
                y_m=y_m,
                track_id=track_id,
                confidence=confidence,
                v_kmh=0.0,
                is_gap=False,
                is_clamped=False,
            )
            self.history.append(pt)
            return pt

        prev = self.history[-1]
        dt = timestamp_s - prev.timestamp_s
        if dt <= 0:
            dt = 1.0 / self.fps

        # Check for tracking gap
        if dt > self.MAX_CONTINUOUS_GAP_S or (frame_idx - prev.frame_idx) > int(self.fps * self.MAX_CONTINUOUS_GAP_S):
            self.gap_count += 1
            pt = KinematicPoint(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                x_m=x_m,
                y_m=y_m,
                track_id=track_id,
                confidence=confidence,
                v_kmh=0.0,
                is_gap=True,
                is_clamped=False,
            )
            self.history.append(pt)
            self.total_duration_s += dt
            return pt

        # Compute raw Euclidean displacement
        dx = x_m - prev.x_m
        dy = y_m - prev.y_m
        disp = math.hypot(dx, dy)
        raw_speed_ms = disp / dt
        raw_speed_kmh = raw_speed_ms * 3.6

        is_clamped = False
        # Velocity / Acceleration Sanity Check
        if raw_speed_kmh > self.MAX_VALID_SPEED_KMH:
            self.velocity_clamped_count += 1
            is_clamped = True
            disp = (self.MAX_VALID_SPEED_KMH / 3.6) * dt
            raw_speed_kmh = self.MAX_VALID_SPEED_KMH

        # Jitter Deadband Filter
        # Suppress micro-vibrations from detector bounding box wobble
        effective_disp = disp
        if disp < self.DEADBAND_DISP_M or raw_speed_kmh < self.DEADBAND_SPEED_KMH:
            self.jitter_suppressed_count += 1
            effective_disp = 0.0
            speed_kmh = 0.0
        else:
            speed_kmh = raw_speed_kmh

        # Accumulate metrics
        self.total_distance_m += effective_disp
        self.total_duration_s += dt

        if speed_kmh >= self.DEADBAND_SPEED_KMH:
            self.active_moving_duration_s += dt

        # Speed Zone Distribution
        if speed_kmh < self.ZONE_WALKING_MAX:
            self.zone_walking_m += effective_disp
        elif speed_kmh < self.ZONE_JOGGING_MAX:
            self.zone_jogging_m += effective_disp
        elif speed_kmh < self.ZONE_RUNNING_MAX:
            self.zone_running_m += effective_disp
        elif speed_kmh < self.ZONE_HSR_MAX:
            self.zone_hsr_m += effective_disp
        else:
            self.zone_sprinting_m += effective_disp

        # Sprint Effort Detection (sustained >= 25.2 km/h)
        if speed_kmh >= self.ZONE_HSR_MAX:
            self._current_sprint_frames += 1
        else:
            if self._current_sprint_frames >= self.min_sprint_duration_frames:
                self.sprint_count += 1
            self._current_sprint_frames = 0

        pt = KinematicPoint(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            x_m=x_m,
            y_m=y_m,
            track_id=track_id,
            confidence=confidence,
            v_kmh=round(speed_kmh, 2),
            is_gap=False,
            is_clamped=is_clamped,
        )
        self.history.append(pt)
        return pt

    def get_summary(self) -> Dict[str, Any]:
        """
        Produce verified match physical kinematics summary.
        """
        # Close any active sprint at end of sequence
        if self._current_sprint_frames >= self.min_sprint_duration_frames:
            self.sprint_count += 1
            self._current_sprint_frames = 0

        total_pts = len(self.history)
        if total_pts < 2 or self.total_duration_s <= 0:
            return {
                "total_distance_m": 0.0,
                "total_distance_km": 0.0,
                "fifa_avg_speed_kmh": 0.0,
                "active_moving_avg_speed_kmh": 0.0,
                "max_speed_kmh": 0.0,
                "duration_s": 0.0,
                "sprint_count": 0,
                "speed_zones_m": {
                    "walking": 0.0,
                    "jogging": 0.0,
                    "running": 0.0,
                    "high_speed_running": 0.0,
                    "sprinting": 0.0,
                },
                "reliability_score": 0.0,
                "frames_analyzed": total_pts,
            }

        # True FIFA match average speed (Total distance / Total match duration)
        fifa_avg_speed_kmh = (self.total_distance_m / max(0.1, self.total_duration_s)) * 3.6

        # Active moving speed (distance / moving duration)
        active_speed_kmh = (
            (self.total_distance_m / max(0.1, self.active_moving_duration_s)) * 3.6
            if self.active_moving_duration_s > 0
            else 0.0
        )

        # Smooth peak speed using a 5-frame moving average to prevent single-frame spikes
        speeds = [pt.v_kmh for pt in self.history if not pt.is_gap]
        if len(speeds) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smoothed_speeds = np.convolve(speeds, kernel, mode="valid")
            max_speed = float(np.max(smoothed_speeds))
        elif speeds:
            max_speed = float(np.max(speeds))
        else:
            max_speed = 0.0

        max_speed = min(self.MAX_VALID_SPEED_KMH, max_speed)

        # Reliability score (continuous float [0.0, 1.0])
        total_eval = max(1, total_pts)
        glitch_ratio = (self.velocity_clamped_count + self.gap_count) / total_eval
        reliability_score = max(0.0, min(1.0, 1.0 - glitch_ratio * 3.0))

        return {
            "total_distance_m": round(self.total_distance_m, 2),
            "total_distance_km": round(self.total_distance_m / 1000.0, 3),
            "fifa_avg_speed_kmh": round(fifa_avg_speed_kmh, 2),
            "active_moving_avg_speed_kmh": round(active_speed_kmh, 2),
            "max_speed_kmh": round(max_speed, 2),
            "duration_s": round(self.total_duration_s, 2),
            "sprint_count": self.sprint_count,
            "speed_zones_m": {
                "walking": round(self.zone_walking_m, 2),
                "jogging": round(self.zone_jogging_m, 2),
                "running": round(self.zone_running_m, 2),
                "high_speed_running": round(self.zone_hsr_m, 2),
                "sprinting": round(self.zone_sprinting_m, 2),
            },
            "speed_zones_pct": {
                "walking": round(self.zone_walking_m / max(0.01, self.total_distance_m) * 100, 1),
                "jogging": round(self.zone_jogging_m / max(0.01, self.total_distance_m) * 100, 1),
                "running": round(self.zone_running_m / max(0.01, self.total_distance_m) * 100, 1),
                "high_speed_running": round(self.zone_hsr_m / max(0.01, self.total_distance_m) * 100, 1),
                "sprinting": round(self.zone_sprinting_m / max(0.01, self.total_distance_m) * 100, 1),
            },
            "reliability_score": round(reliability_score, 3),
            "frames_analyzed": total_pts,
            "jitter_suppressed_frames": self.jitter_suppressed_count,
            "velocity_clamped_frames": self.velocity_clamped_count,
            "gaps_detected": self.gap_count,
        }
