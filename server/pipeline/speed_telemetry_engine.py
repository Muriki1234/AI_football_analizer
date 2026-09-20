"""
speed_telemetry_engine.py - 5-Zone Athletic Speed Kinematics & Downsampled Timeline Telemetry Engine

Architectural Foundations:
1. FIFA Standard 5-Zone Physical Intensity Breakdown:
   - Zone 1 (Walking): 0.0 - 7.2 km/h
   - Zone 2 (Jogging): 7.2 - 14.4 km/h
   - Zone 3 (Low-Medium Running): 14.4 - 19.8 km/h
   - Zone 4 (High-Speed Running / HSR): 19.8 - 25.2 km/h
   - Zone 5 (Sprinting): >= 25.2 km/h
   Quantifies distance (m), percentage (%), and time spent (s) across each intensity tier.

2. Occlusion Gap Smoothing:
   Eliminates single-frame and brief tracking dropout spikes to 0 km/h by interpolating
   isolated missing frames (<= 4 frames) between valid player tracklets.

3. LTTB (Largest Triangle Three Buckets) Downsampling:
   Preserves explosive sprint acceleration peaks and key morphological inflections
   while reducing raw multi-thousand-point timeseries to lightweight JSON payloads (100-200 pts)
   tailored for high-performance frontend chart rendering.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIFA_ZONES = [
    ("zone_1_walking", "Walking", 0.0, 7.2, "#94a3b8"),
    ("zone_2_jogging", "Jogging", 7.2, 14.4, "#38bdf8"),
    ("zone_3_running", "Running", 14.4, 19.8, "#22c55e"),
    ("zone_4_hsr", "High-Speed Running", 19.8, 25.2, "#f59e0b"),
    ("zone_5_sprinting", "Sprinting", 25.2, 999.0, "#ef4444"),
]


def interpolate_occlusion_gaps(
    raw_speeds: List[Optional[float]],
    max_gap: int = 4,
) -> List[float]:
    """
    Interpolates isolated missing frames (None or 0 when player track drops briefly)
    to prevent artificial zero-speed downward plunge spikes.
    """
    n = len(raw_speeds)
    if n == 0:
        return []

    # Replace None with NaN
    arr = np.array([float(s) if (s is not None and not np.isnan(s)) else np.nan for s in raw_speeds], dtype=float)

    # If all NaN, return zeros
    if np.all(np.isnan(arr)):
        return [0.0] * n

    # Find isolated NaN or zero gaps between valid positive values
    out = arr.copy()
    valid_indices = np.where(~np.isnan(out) & (out > 0.5))[0]

    if len(valid_indices) >= 2:
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]
            gap = idx2 - idx1 - 1
            if 0 < gap <= max_gap:
                v1 = out[idx1]
                v2 = out[idx2]
                for g in range(1, gap + 1):
                    alpha = g / (gap + 1)
                    out[idx1 + g] = (1.0 - alpha) * v1 + alpha * v2

    # Fill remaining NaNs with 0.0
    out = np.nan_to_num(out, nan=0.0)
    return [round(float(v), 2) for v in out]


def lttb_downsample(
    times: List[float],
    values: List[float],
    target_points: int = 150,
) -> List[Dict[str, float]]:
    """
    Largest Triangle Three Buckets (LTTB) algorithm for timeseries downsampling.
    Preserves visual extrema and peaks without smoothing artifacts.
    """
    n = len(times)
    if n <= target_points or target_points < 3:
        return [{"time": round(times[i], 2), "value": round(values[i], 2)} for i in range(n)]

    sampled: List[Dict[str, float]] = [{"time": round(times[0], 2), "value": round(values[0], 2)}]

    bucket_size = (n - 2) / (target_points - 2)
    a = 0  # previously selected point index

    for i in range(target_points - 2):
        # Current bucket range
        c_start = int(np.floor((i + 0) * bucket_size)) + 1
        c_end = int(np.floor((i + 1) * bucket_size)) + 1
        c_end = min(c_end, n)

        # Next bucket range (for centroid average)
        n_start = int(np.floor((i + 1) * bucket_size)) + 1
        n_end = int(np.floor((i + 2) * bucket_size)) + 1
        n_end = min(n_end, n)

        # Average point in next bucket
        if n_start < n_end:
            avg_x = float(np.mean(times[n_start:n_end]))
            avg_y = float(np.mean(values[n_start:n_end]))
        else:
            avg_x = times[-1]
            avg_y = values[-1]

        # Point A coordinates
        ax = times[a]
        ay = values[a]

        max_area = -1.0
        best_idx = c_start

        for j in range(c_start, c_end):
            bx = times[j]
            by = values[j]
            # Triangle area: 0.5 * |ax(by - avg_y) + bx(avg_y - ay) + avg_x(ay - by)|
            area = abs(ax * (by - avg_y) + bx * (avg_y - ay) + avg_x * (ay - by)) * 0.5
            if area > max_area:
                max_area = area
                best_idx = j

        sampled.append({"time": round(times[best_idx], 2), "value": round(values[best_idx], 2)})
        a = best_idx

    sampled.append({"time": round(times[-1], 2), "value": round(values[-1], 2)})
    return sampled


class SpeedTelemetryEngine:
    """
    Computes 5-zone athletic metrics, cleans tracking gaps,
    downsamples timeseries for web clients, and renders tactical charts.
    """

    def __init__(self, fps: float = 25.0, target_downsample_points: int = 150):
        self.fps = max(float(fps), 1.0)
        self.target_points = max(int(target_downsample_points), 20)

    def process_telemetry(
        self,
        raw_speeds: List[Optional[float]],
        raw_distances: List[Optional[float]],
    ) -> Dict[str, Any]:
        """
        Processes full match speeds and distances into structured athletic telemetry.
        """
        n_frames = len(raw_speeds)
        if n_frames == 0:
            return {
                "max_speed_kmh": 0.0,
                "avg_speed_kmh": 0.0,
                "total_distance_m": 0.0,
                "zone_breakdown": {},
                "downsampled_timeline": [],
            }

        # 1. Clean occlusion dropouts
        clean_speeds = interpolate_occlusion_gaps(raw_speeds, max_gap=4)

        # 2. Reconstruct monotonic distance if needed
        clean_distances: List[float] = []
        curr_d = 0.0
        for i, d in enumerate(raw_distances):
            if d is not None and not np.isnan(d):
                curr_d = max(curr_d, float(d))
            elif i > 0 and clean_speeds[i] > 0.5:
                # Add incremental motion estimate: speed (km/h) / 3.6 / fps (m/frame)
                curr_d += (clean_speeds[i] / 3.6) / self.fps
            clean_distances.append(round(curr_d, 2))

        # 3. 5-Zone breakdown
        zone_stats = {
            z_id: {
                "name": z_name,
                "distance_m": 0.0,
                "duration_s": 0.0,
                "percentage": 0.0,
                "color": z_col,
            }
            for z_id, z_name, _, _, z_col in FIFA_ZONES
        }

        total_d = clean_distances[-1] if clean_distances else 0.0

        for i in range(1, n_frames):
            spd = clean_speeds[i]
            d_delta = max(0.0, clean_distances[i] - clean_distances[i - 1])
            frame_time_s = 1.0 / self.fps

            for z_id, _, min_v, max_v, _ in FIFA_ZONES:
                if min_v <= spd < max_v:
                    zone_stats[z_id]["distance_m"] += d_delta
                    zone_stats[z_id]["duration_s"] += frame_time_s
                    break

        for z_id in zone_stats:
            z_d = zone_stats[z_id]["distance_m"]
            zone_stats[z_id]["distance_m"] = round(z_d, 2)
            zone_stats[z_id]["duration_s"] = round(zone_stats[z_id]["duration_s"], 2)
            pct = (z_d / max(total_d, 1e-6)) * 100.0
            zone_stats[z_id]["percentage"] = round(pct, 1)

        # 4. LTTB downsampling for web charts
        times = [i / self.fps for i in range(n_frames)]
        downsampled_speeds = lttb_downsample(times, clean_speeds, self.target_points)
        downsampled_dists = lttb_downsample(times, clean_distances, self.target_points)

        timeline = []
        for s_pt, d_pt in zip(downsampled_speeds, downsampled_dists):
            timeline.append({
                "time": s_pt["time"],
                "speed_kmh": s_pt["value"],
                "distance_m": d_pt["value"],
            })

        max_spd = round(float(np.max(clean_speeds)), 1) if clean_speeds else 0.0
        avg_spd = round(float(np.mean([s for s in clean_speeds if s > 1.0])), 1) if any(s > 1.0 for s in clean_speeds) else 0.0

        return {
            "max_speed_kmh": max_spd,
            "avg_speed_kmh": avg_spd,
            "total_distance_m": round(total_d, 1),
            "zone_breakdown": zone_stats,
            "downsampled_timeline": timeline,
            "clean_speeds": clean_speeds,
            "clean_distances": clean_distances,
            "times": times,
        }

    def render_chart(self, telemetry: Dict[str, Any], output_path: Path) -> None:
        """
        Renders publication-grade 2-panel speed & distance visualization.
        """
        times = telemetry.get("times", [])
        speeds = telemetry.get("clean_speeds", [])
        distances = telemetry.get("clean_distances", [])

        if not times or not speeds:
            return

        BG = "#1a1a2e"
        PANEL = "#16213e"
        RED = "#e74c3c"
        BLUE = "#3498db"
        YELLOW = "#f1c40f"
        CYAN = "#1abc9c"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG, sharex=True)
        fig.subplots_adjust(hspace=0.08)

        # 1. Speed Plot
        ax1.set_facecolor(PANEL)
        ax1.plot(times, speeds, color=RED, linewidth=1.5, label="Speed (km/h)")
        ax1.fill_between(times, speeds, alpha=0.15, color=RED)
        ax1.axhline(y=25.2, color=RED, ls="--", lw=1.2, alpha=0.8, label="Sprint (25.2 km/h)")
        ax1.axhline(y=19.8, color=YELLOW, ls="--", lw=1.0, alpha=0.8, label="HSR (19.8 km/h)")
        ax1.axhline(y=7.2, color=CYAN, ls=":", lw=1.0, alpha=0.6, label="Jogging (7.2 km/h)")

        max_spd = telemetry.get("max_speed_kmh", 0.0)
        avg_spd = telemetry.get("avg_speed_kmh", 0.0)
        tot_dist = telemetry.get("total_distance_m", 0.0)

        ax1.set_ylabel("Speed (km/h)", color="white", fontsize=11)
        ax1.set_title(
            f"Tracked Player — Speed & Distance Profile (Max: {max_spd} km/h | Avg: {avg_spd} km/h | Total: {tot_dist:.0f}m)",
            color="white", fontsize=14, fontweight="bold", pad=10
        )
        ax1.legend(facecolor=PANEL, labelcolor="white", fontsize=8, loc="upper right")
        ax1.tick_params(colors="white", labelsize=9)
        ax1.grid(alpha=0.15, color="white")
        for spine in ax1.spines.values():
            spine.set_color("#444")

        # 2. Cumulative Distance Plot
        ax2.set_facecolor(PANEL)
        ax2.plot(times, distances, color=BLUE, linewidth=2.0, label="Cumulative Distance (m)")
        ax2.fill_between(times, distances, alpha=0.18, color=BLUE)
        ax2.set_ylabel("Distance (m)", color="white", fontsize=11)
        ax2.set_xlabel("Time (s)", color="white", fontsize=11)
        ax2.legend(facecolor=PANEL, labelcolor="white", fontsize=9, loc="upper left")
        ax2.tick_params(colors="white", labelsize=9)
        ax2.grid(alpha=0.15, color="white")
        for spine in ax2.spines.values():
            spine.set_color("#444")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
