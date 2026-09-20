"""
sprint_burst_debouncer.py - Robust Kinematic Sprint Burst Spotter & Dropout Debouncer

Architectural Foundations:
1. FIFA EPTS & Catapult Kinematic Standards:
   Calibrates athletic sprint threshold (>= 24.0 km/h) and realistic explosive duration
   (>= 0.8s / 20 frames at 25 fps) instead of legacy rigid 2.0s barrier.
2. Hysteresis Dropout Debouncing:
   Prevents single-frame bounding box noise, occlusion jitter, or brief speed drops
   (<= 3 frames, speed >= 20.0 km/h) from shattering a continuous sprint into
   discarded fragments.
3. Metric Fidelity:
   Integrates true Euclidean pitch trajectory distance (m), peak velocity,
   average burst velocity, and timestamps into structured event telemetry.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SprintBurstDebouncer:
    """
    Kinematic analyzer for high-velocity sprint bursts with dropout debouncing.
    """

    def __init__(
        self,
        sprint_speed_kmh: float = 24.0,
        min_duration_s: float = 0.8,
        max_dropout_frames: int = 3,
        dropout_speed_floor_kmh: float = 20.0,
        fps: float = 25.0,
    ):
        self.sprint_speed_kmh = float(sprint_speed_kmh)
        self.min_duration_s = float(min_duration_s)
        self.max_dropout_frames = int(max_dropout_frames)
        self.dropout_speed_floor_kmh = float(dropout_speed_floor_kmh)
        self.fps = max(float(fps), 1.0)
        self.min_frames = int(self.min_duration_s * self.fps)

    def detect_sprints(
        self,
        speeds: List[float],
        positions: List[Optional[Tuple[float, float]]],
    ) -> Dict[str, Any]:
        """
        Detects sustained sprint bursts from per-frame speed and pitch positions.
        Returns aggregate athletic metrics and verified events.
        """
        if not speeds:
            return {
                "sprint_count": 0,
                "total_sprint_distance_m": 0.0,
                "avg_sprint_distance_m": 0.0,
                "avg_duration_s": 0.0,
                "max_duration_s": 0.0,
                "max_speed_kmh": 0.0,
                "avg_sprint_speed_kmh": 0.0,
                "events": [],
                "segments": [],
            }

        sprint_segments: List[Tuple[int, int, List[Tuple[float, float]]]] = []
        events: List[Dict[str, Any]] = []

        in_sprint = False
        sprint_start = 0
        sprint_pts: List[Tuple[float, float]] = []
        dropout_count = 0
        burst_speeds: List[float] = []

        def _finalize_burst(start_f: int, end_f: int, pts: List[Tuple[float, float]], b_speeds: List[float]):
            duration_frames = (end_f - start_f + 1)
            if duration_frames >= self.min_frames and len(pts) >= 2:
                # Compute trajectory distance (meters)
                dist_m = 0.0
                for k in range(len(pts) - 1):
                    p1, p2 = pts[k], pts[k + 1]
                    dist_m += float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

                dur_s = round(duration_frames / self.fps, 2)
                peak_spd = round(float(np.max(b_speeds)), 1) if b_speeds else 0.0
                avg_spd = round(float(np.mean(b_speeds)), 1) if b_speeds else 0.0

                sprint_segments.append((start_f, end_f, list(pts)))
                start_sec = round(start_f / self.fps, 2)
                mm = int(start_sec // 60)
                ss = int(start_sec % 60)

                events.append({
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "time_sec": start_sec,
                    "time_mm_ss": f"{mm:02d}:{ss:02d}",
                    "duration_s": dur_s,
                    "distance_m": round(dist_m, 2),
                    "peak_speed_kmh": peak_spd,
                    "avg_speed_kmh": avg_spd,
                    "positions": pts,
                })

        for i, spd in enumerate(speeds):
            pos = positions[i] if (positions and i < len(positions)) else None

            if spd >= self.sprint_speed_kmh:
                if not in_sprint:
                    in_sprint = True
                    sprint_start = i
                    sprint_pts = []
                    burst_speeds = []
                    dropout_count = 0

                dropout_count = 0
                burst_speeds.append(spd)
                if pos is not None:
                    sprint_pts.append(pos)

            elif in_sprint:
                # Speed dropped below threshold: test hysteresis tolerance
                if spd >= self.dropout_speed_floor_kmh and dropout_count < self.max_dropout_frames:
                    dropout_count += 1
                    burst_speeds.append(spd)
                    if pos is not None:
                        sprint_pts.append(pos)
                else:
                    # Burst ended: trim trailing dropout frames
                    effective_end = i - 1 - dropout_count
                    if effective_end >= sprint_start:
                        valid_pts = sprint_pts[:len(sprint_pts) - dropout_count] if dropout_count > 0 else sprint_pts
                        valid_speeds = burst_speeds[:len(burst_speeds) - dropout_count] if dropout_count > 0 else burst_speeds
                        _finalize_burst(sprint_start, effective_end, valid_pts, valid_speeds)
                    in_sprint = False
                    sprint_pts = []
                    burst_speeds = []
                    dropout_count = 0

        # Handle burst active at video boundary
        if in_sprint:
            effective_end = len(speeds) - 1 - dropout_count
            if effective_end >= sprint_start:
                valid_pts = sprint_pts[:len(sprint_pts) - dropout_count] if dropout_count > 0 else sprint_pts
                valid_speeds = burst_speeds[:len(burst_speeds) - dropout_count] if dropout_count > 0 else burst_speeds
                _finalize_burst(sprint_start, effective_end, valid_pts, valid_speeds)

        durations = [e["duration_s"] for e in events]
        distances = [e["distance_m"] for e in events]
        burst_avg_speeds = [e["avg_speed_kmh"] for e in events]
        total_dist = round(float(np.sum(distances)), 2) if distances else 0.0

        return {
            "sprint_count": len(events),
            "total_sprint_distance_m": total_dist,
            "avg_sprint_distance_m": round(float(np.mean(distances)), 2) if distances else 0.0,
            "avg_duration_s": round(float(np.mean(durations)), 2) if durations else 0.0,
            "max_duration_s": round(float(np.max(durations)), 2) if durations else 0.0,
            "max_speed_kmh": round(float(np.max(speeds)), 1) if speeds else 0.0,
            "avg_sprint_speed_kmh": round(float(np.mean(burst_avg_speeds)), 1) if burst_avg_speeds else 0.0,
            "events": events[:50],
            "segments": sprint_segments,
        }

    def render_visualization(
        self,
        segments: List[Tuple[int, int, List[Tuple[float, float]]]],
        stats: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Renders a publication-grade pitch diagram of sprint tracks and key metrics.
        """
        BG = "#1a1a2e"
        GREEN = "#2d6a1e"
        COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#e91e63"]

        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
        ax.set_facecolor(GREEN)

        # Standard pitch geometry (105m x 68m)
        for rect in [
            plt.Rectangle((0, 0), 105, 68, fill=False, ec="white", lw=2),
            plt.Rectangle((0, 13.84), 16.5, 40.32, fill=False, ec="white"),
            plt.Rectangle((88.5, 13.84), 16.5, 40.32, fill=False, ec="white"),
            plt.Rectangle((0, 24.84), 5.5, 18.32, fill=False, ec="white"),
            plt.Rectangle((99.5, 24.84), 5.5, 18.32, fill=False, ec="white"),
        ]:
            ax.add_patch(rect)

        ax.axvline(x=52.5, color="white", lw=1.5)
        circle = plt.Circle((52.5, 34), 9.15, fill=False, ec="white", lw=1.5)
        ax.add_patch(circle)

        # Draw sprint bursts
        for idx, (sf, ef, pts) in enumerate(segments):
            col = COLORS[idx % len(COLORS)]
            arr = np.array(pts)
            if len(arr) >= 2:
                ax.plot(arr[:, 0], arr[:, 1], color=col, lw=3, label=f"Sprint #{idx+1} ({sf}-{ef})")
                # Start marker
                ax.scatter(arr[0, 0], arr[0, 1], color="white", s=60, zorder=5, edgecolors=col, lw=2)
                # End marker
                ax.scatter(arr[-1, 0], arr[-1, 1], color=col, s=80, zorder=5, marker="^")

        ax.set_xlim(-2, 107)
        ax.set_ylim(-2, 70)
        sprint_cnt = stats.get("sprint_count", 0)
        max_spd = stats.get("max_speed_kmh", 0.0)
        tot_dist = stats.get("total_sprint_distance_m", 0.0)

        ax.set_title(
            f"High-Intensity Sprint Burst Analysis — {sprint_cnt} Burst(s) | Max: {max_spd} km/h | Total: {tot_dist:.1f}m",
            color="white", fontsize=13, fontweight="bold", pad=12
        )
        if segments:
            ax.legend(facecolor=BG, labelcolor="white", fontsize=8, loc="upper right")
        ax.tick_params(colors="white")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
