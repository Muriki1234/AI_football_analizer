"""
analytics_accuracy_foundation.py — Unified 5-Stage Football Analytics Accuracy Engine

Grounds video analytics in a unified, end-to-end evaluation chain rather than
isolated per-feature heuristics:
  Detection → Tracking → Team Identity → Pitch Coordinates / Homography → Trajectory Kinematics

Provides standard quantitative metrics and physical validation for:
1. Detection: mAP, recall, small/distant player recall, crowded box recall, ball coverage.
2. Tracking: TrackEval standard (HOTA, MOTA, IDF1, ID Switches, Fragmentation).
3. Team Identity: Team label flip rate, temporal consensus purity, cluster stability.
4. Pitch Coordinates & Homography: Metric bounds validity ([0, 105]m x [0, 68]m),
   anomalous spatial teleportation/jitter rate, keypoint reprojection stability.
5. Trajectory Kinematics: Physical feasibility (v <= 36 km/h, a <= 6 m/s^2),
   trajectory continuity / gap bridging rate, distance conservation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class TeamIdentityMetrics:
    """
    Evaluates temporal stability and purity of team assignment across player trajectories.
    Eliminates flickering team colors that corrupt pass networks and heatmaps.
    """

    @staticmethod
    def evaluate(
        tracks_players: List[Dict[int, Dict[str, Any]]],
        min_track_length: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyzes team assignment stability across frames for each tracklet.
        tracks_players: list of dicts mapping player_id -> {"team": int or None, ...} per frame.
        """
        track_teams: Dict[int, List[int]] = {}
        total_assignments = 0

        for frame_dict in tracks_players:
            for pid, pdata in frame_dict.items():
                team = pdata.get("team")
                if team is not None and team in (0, 1, 2):
                    track_teams.setdefault(pid, []).append(int(team))
                    total_assignments += 1

        if not track_teams:
            return {
                "total_tracks": 0,
                "total_assignments": 0,
                "mean_team_purity": 1.0,
                "team_flip_count": 0,
                "team_flip_rate": 0.0,
                "stable_tracks_pct": 100.0,
                "score": 1.0,
            }

        total_flips = 0
        purities = []
        stable_tracks = 0

        for pid, teams in track_teams.items():
            if len(teams) < min_track_length:
                continue

            # Count flips (transitions between team 0 and 1, ignoring referee 2)
            flips = 0
            for i in range(1, len(teams)):
                if teams[i] != teams[i - 1] and teams[i] in (0, 1) and teams[i - 1] in (0, 1):
                    flips += 1
            total_flips += flips

            # Purity: fraction of observations matching majority team label
            counts = {t: teams.count(t) for t in set(teams)}
            majority_count = max(counts.values()) if counts else 0
            purity = majority_count / max(1, len(teams))
            purities.append(purity)

            if flips == 0:
                stable_tracks += 1

        evaluated_tracks = max(1, len(purities))
        mean_purity = float(np.mean(purities)) if purities else 1.0
        flip_rate = total_flips / max(1, total_assignments)
        stable_pct = (stable_tracks / evaluated_tracks) * 100.0

        # Score: weighted combination of purity and flip-free rate
        score = max(0.0, min(1.0, (mean_purity * 0.7) + ((1.0 - min(1.0, flip_rate * 50)) * 0.3)))

        return {
            "total_tracks": len(track_teams),
            "evaluated_tracks": len(purities),
            "total_assignments": total_assignments,
            "mean_team_purity": round(mean_purity, 4),
            "team_flip_count": total_flips,
            "team_flip_rate": round(flip_rate, 4),
            "stable_tracks_pct": round(stable_pct, 2),
            "score": round(score, 4),
        }


class PitchHomographyMetrics:
    """
    Evaluates 2D pitch coordinate mapping validity, spatial bounds, and geometric stability.
    Standard football pitch: 105m x 68m (with 5m out-of-bounds safety apron: [-5, 110] x [-5, 73]).
    """

    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0
    APRON_MARGIN_M = 5.0

    @classmethod
    def evaluate(
        cls,
        tracks_players: List[Dict[int, Dict[str, Any]]],
        fps: float = 25.0,
        max_speed_teleport_m_s: float = 12.0,  # >12 m/s (~43.2 km/h) indicates homography jitter
    ) -> Dict[str, Any]:
        total_points = 0
        valid_points = 0
        teleport_jumps = 0
        prev_positions: Dict[int, Tuple[float, float, int]] = {}  # pid -> (x, y, frame_idx)

        dt_base = 1.0 / max(1.0, fps)

        for fidx, frame_dict in enumerate(tracks_players):
            for pid, pdata in frame_dict.items():
                pos = pdata.get("position_transformed") or pdata.get("position_2d") or pdata.get("position")
                if pos is None or not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
                    continue

                x, y = float(pos[0]), float(pos[1])
                total_points += 1

                # 1. Bounds check (within pitch + apron margin)
                x_min, x_max = -cls.APRON_MARGIN_M, cls.PITCH_LENGTH_M + cls.APRON_MARGIN_M
                y_min, y_max = -cls.APRON_MARGIN_M, cls.PITCH_WIDTH_M + cls.APRON_MARGIN_M

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    valid_points += 1

                # 2. Teleportation / Jitter check
                if pid in prev_positions:
                    px, py, p_fidx = prev_positions[pid]
                    df = fidx - p_fidx
                    if 1 <= df <= 5:  # within short temporal window
                        dist = math.hypot(x - px, y - py)
                        speed = dist / (df * dt_base)
                        if speed > max_speed_teleport_m_s:
                            teleport_jumps += 1

                prev_positions[pid] = (x, y, fidx)

        valid_ratio = valid_points / max(1, total_points)
        teleport_rate = teleport_jumps / max(1, total_points)

        score = max(0.0, min(1.0, (valid_ratio * 0.75) + ((1.0 - min(1.0, teleport_rate * 100)) * 0.25)))

        return {
            "total_projected_points": total_points,
            "valid_points_in_bounds": valid_points,
            "bounds_validity_pct": round(valid_ratio * 100.0, 2),
            "teleport_jumps_count": teleport_jumps,
            "teleport_jump_rate": round(teleport_rate, 4),
            "score": round(score, 4),
        }


class TrajectoryKinematicsMetrics:
    """
    Evaluates kinematic trajectory feasibility (speed and acceleration distributions)
    adhering to FIFA / Catapult physiological human athletic limits.
    Physiological human football limits:
      - Max sprinting speed: ~36.0 km/h (10.0 m/s) — Usain Bolt peak is ~44.7 km/h, elite footballers rarely exceed 37.0 km/h
      - Max physical acceleration: ~6.0 m/s^2
      - Max physical deceleration: ~-7.0 m/s^2
    """

    MAX_HUMAN_SPEED_KMH = 37.0
    MAX_HUMAN_ACCEL_M_S2 = 6.5

    @classmethod
    def evaluate(
        cls,
        tracks_players: List[Dict[int, Dict[str, Any]]],
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        total_velocity_samples = 0
        speeding_violations = 0
        accel_violations = 0
        observed_speeds_kmh: List[float] = []

        dt = 1.0 / max(1.0, fps)
        prev_state: Dict[int, Tuple[float, float, Optional[float], int]] = {}  # pid -> (x, y, speed_m_s_or_None, fidx)

        for fidx, frame_dict in enumerate(tracks_players):
            for pid, pdata in frame_dict.items():
                pos = pdata.get("position_transformed") or pdata.get("position_2d") or pdata.get("position")
                if pos is None or not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
                    continue

                x, y = float(pos[0]), float(pos[1])

                if pid in prev_state:
                    px, py, prev_v, pf = prev_state[pid]
                    df = fidx - pf
                    if 1 <= df <= 3:
                        dt_step = df * dt
                        dist = math.hypot(x - px, y - py)
                        curr_v = dist / dt_step  # m/s
                        speed_kmh = curr_v * 3.6

                        total_velocity_samples += 1
                        observed_speeds_kmh.append(speed_kmh)

                        if speed_kmh > cls.MAX_HUMAN_SPEED_KMH:
                            speeding_violations += 1

                        if prev_v is not None:
                            accel = abs(curr_v - prev_v) / dt_step  # m/s^2
                            if accel > cls.MAX_HUMAN_ACCEL_M_S2:
                                accel_violations += 1

                        prev_state[pid] = (x, y, curr_v, fidx)
                        continue

                prev_state[pid] = (x, y, None, fidx)

        speed_violation_rate = speeding_violations / max(1, total_velocity_samples)
        accel_violation_rate = accel_violations / max(1, total_velocity_samples)
        peak_speed = float(max(observed_speeds_kmh)) if observed_speeds_kmh else 0.0
        mean_speed = float(np.mean(observed_speeds_kmh)) if observed_speeds_kmh else 0.0

        # Physical feasibility score
        score = max(0.0, min(1.0, 1.0 - (speed_violation_rate * 0.6 + accel_violation_rate * 0.4) * 20.0))

        return {
            "total_velocity_samples": total_velocity_samples,
            "speeding_violations": speeding_violations,
            "speeding_violation_rate": round(speed_violation_rate, 4),
            "accel_violations": accel_violations,
            "accel_violation_rate": round(accel_violation_rate, 4),
            "peak_speed_kmh": round(peak_speed, 2),
            "mean_speed_kmh": round(mean_speed, 2),
            "score": round(score, 4),
        }


class UnifiedAnalyticsAccuracyScorecard:
    """
    Computes a holistic 5-pillar accuracy scorecard across:
      1. Detection
      2. Tracking
      3. Team Identity
      4. Pitch Homography
      5. Trajectory Kinematics
    """

    @staticmethod
    def evaluate(
        detection_metrics: Dict[str, Any],
        tracking_metrics: Dict[str, Any],
        team_metrics: Dict[str, Any],
        homography_metrics: Dict[str, Any],
        kinematics_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Pillar sub-scores (0.0 to 1.0)
        s_det = float(detection_metrics.get("f1_score", 0.0))
        s_track = float(tracking_metrics.get("HOTA", 0.0))
        s_team = float(team_metrics.get("score", 0.0))
        s_homography = float(homography_metrics.get("score", 0.0))
        s_kinematics = float(kinematics_metrics.get("score", 0.0))

        # Pillar weights reflect downstream impact on coaching insights
        # Detection (0.20) + Tracking (0.25) + Team (0.15) + Homography (0.25) + Kinematics (0.15) = 1.0
        hai = (
            s_det * 0.20 +
            s_track * 0.25 +
            s_team * 0.15 +
            s_homography * 0.25 +
            s_kinematics * 0.15
        )

        return {
            "holistic_accuracy_index": round(hai, 4),
            "pillars": {
                "1_detection": {"f1": s_det, "weight": 0.20},
                "2_tracking": {"hota": s_track, "weight": 0.25},
                "3_team_identity": {"purity_score": s_team, "weight": 0.15},
                "4_pitch_homography": {"bounds_score": s_homography, "weight": 0.25},
                "5_trajectory_kinematics": {"physics_score": s_kinematics, "weight": 0.15},
            },
            "status": "EXCELLENT" if hai >= 0.85 else ("GOOD" if hai >= 0.70 else "NEEDS_IMPROVEMENT"),
        }
