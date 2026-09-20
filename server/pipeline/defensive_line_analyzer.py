"""
defensive_line_analyzer.py - Bidirectional Defensive Line & Progressive Penetration Analyzer

Architectural Foundations:
1. Bidirectional Attack Resolution (StatsBomb / FIFA Training Centre Standard):
   Automatically identifies team attacking direction (+x / 'right' vs -x / 'left')
   based on team spatial centroids. Deepest defenders protecting their goal are
   dynamically extracted from the correct pitch extreme (min x for left-attacking,
   max x for right-attacking).

2. Sustained Line-Break Event Spotting:
   Penetrations are verified when an attacker crosses the opposing defensive line
   and sustains progressive positioning for >= min_consecutive_frames (default 3 frames),
   eliminating transient bounding box noise or brief deflections.

3. Spatial Metrics & Timeline Telemetry:
   Produces exact penetration timestamps, spatial pitch coordinates, penetration depth (m),
   and directional pitch visualizations with attacking direction indicators.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DefensiveLineAnalyzer:
    """
    Evaluates defensive line positioning, team attacking orientation,
    and line-break penetration events.
    """

    def __init__(
        self,
        defender_count: int = 4,
        min_consecutive_frames: int = 3,
        fps: float = 25.0,
    ):
        self.defender_count = int(defender_count)
        self.min_consecutive_frames = int(min_consecutive_frames)
        self.fps = max(float(fps), 1.0)

    def resolve_attack_direction(
        self,
        tracks: Dict[str, Any],
        tracked_team: int,
        opponent_team: int,
    ) -> str:
        """
        Determines whether the tracked team is attacking towards the right (+x)
        or towards the left (-x) based on global team spatial centroids.
        Returns 'right' or 'left'.
        """
        tracked_xs: List[float] = []
        opponent_xs: List[float] = []

        players_tracks = tracks.get("players", [])
        for f_idx, frame_players in enumerate(players_tracks):
            if not frame_players:
                continue
            for pid, info in frame_players.items():
                if not info:
                    continue
                pos = info.get("position_minimap") or info.get("position_transformed")
                if not (pos and len(pos) == 2 and not any(np.isnan(p) for p in pos)):
                    continue
                team = info.get("team")
                if team == tracked_team:
                    tracked_xs.append(float(pos[0]))
                elif team == opponent_team:
                    opponent_xs.append(float(pos[0]))

        if not tracked_xs or not opponent_xs:
            return "right"  # default fallback

        # If tracked team centroid is to the left of opponent centroid, they attack right (+x)
        return "right" if np.mean(tracked_xs) < np.mean(opponent_xs) else "left"

    def compute_defensive_line_series(
        self,
        tracks: Dict[str, Any],
        tracked_bboxes: Dict[int, Tuple[float, float, float, float]],
        tracked_team: int,
        opponent_team: int,
        attacking_direction: str,
    ) -> List[Optional[Tuple[float, float, Tuple[float, float]]]]:
        """
        Computes per-frame (tracked_x, defense_line_x, tracked_pos).
        """
        frame_data: List[Optional[Tuple[float, float, Tuple[float, float]]]] = []
        players_tracks = tracks.get("players", [])

        for i in range(len(players_tracks)):
            tracked_pos = None
            if i in tracked_bboxes:
                samurai_bbox = tracked_bboxes[i]
                if len(samurai_bbox) == 4:
                    sx, sy, sw, sh = samurai_bbox
                    target_center = (sx + sw / 2.0, sy + sh / 2.0)
                    max_dist = max(150.0, max(sw, sh) * 0.8)
                else:
                    target_center = (float(samurai_bbox[0]), float(samurai_bbox[1]))
                    max_dist = 150.0

                best_dist, matched_info = max_dist, None
                frame_dict = players_tracks[i]
                for pid, pinfo in frame_dict.items():
                    if not pinfo:
                        continue
                    bb = pinfo.get("bbox")
                    if bb and len(bb) == 4:
                        cx = (bb[0] + bb[2]) / 2.0
                        cy = (bb[1] + bb[3]) / 2.0
                        dist = ((cx - target_center[0])**2 + (cy - target_center[1])**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            matched_info = pinfo

                if matched_info:
                    pos = matched_info.get("position_minimap") or matched_info.get("position_transformed")
                    if pos and len(pos) == 2 and not any(np.isnan(p) for p in pos):
                        tracked_pos = (float(pos[0]), float(pos[1]))

            # Opponent player x-coordinates
            opp_xs: List[float] = []
            for pid, pinfo in players_tracks[i].items():
                if not pinfo or pinfo.get("team") != opponent_team:
                    continue
                pp = pinfo.get("position_minimap") or pinfo.get("position_transformed")
                if pp and len(pp) == 2 and not any(np.isnan(v) for v in pp):
                    opp_xs.append(float(pp[0]))

            if opp_xs and tracked_pos:
                opp_xs.sort()
                # If attacking right (+x), defense protects right goal (largest x)
                # If attacking left (-x), defense protects left goal (smallest x)
                k = min(self.defender_count, len(opp_xs))
                deepest = opp_xs[-k:] if attacking_direction == "right" else opp_xs[:k]
                defense_x = float(np.mean(deepest))
                frame_data.append((tracked_pos[0], defense_x, tracked_pos))
            else:
                frame_data.append(None)

        return frame_data

    def detect_penetration_events(
        self,
        frame_data: List[Optional[Tuple[float, float, Tuple[float, float]]]],
        attacking_direction: str,
        tracked_team: int,
        opponent_team: int,
    ) -> Dict[str, Any]:
        """
        Identifies penetration events where the tracked player sustains a position
        beyond the opposing defensive line for >= min_consecutive_frames.
        """
        penetrations: List[Tuple[float, float]] = []
        events: List[Dict[str, Any]] = []
        depths: List[float] = []

        behind = False
        consec = 0
        pen_start_frame = 0
        pen_start_pos = (0.0, 0.0)

        for i, fd in enumerate(frame_data):
            if fd is None:
                consec = 0
                behind = False
                continue

            tx, dx, tp = fd
            # Crossing defense line condition:
            # If attacking right: forward penetration is tx > dx
            # If attacking left: forward penetration is tx < dx
            is_beyond = (tx > dx) if attacking_direction == "right" else (tx < dx)
            depth_m = abs(tx - dx)

            if is_beyond:
                consec += 1
                if consec == 1:
                    pen_start_frame = i
                    pen_start_pos = tp
                if consec >= self.min_consecutive_frames and not behind:
                    penetrations.append(tp)
                    depths.append(depth_m)
                    time_sec = round(pen_start_frame / self.fps, 2)
                    mm = int(time_sec // 60)
                    ss = int(time_sec % 60)
                    events.append({
                        "frame_idx": pen_start_frame,
                        "time_sec": time_sec,
                        "time_mm_ss": f"{mm:02d}:{ss:02d}",
                        "position": [round(pen_start_pos[0], 2), round(pen_start_pos[1], 2)],
                        "penetration_depth_m": round(depth_m, 2),
                    })
                    behind = True
            else:
                consec = 0
                behind = False

        valid_dxs = [fd[1] for fd in frame_data if fd is not None]
        avg_defense_line_x = round(float(np.median(valid_dxs)), 2) if valid_dxs else 52.5
        max_depth = round(float(np.max(depths)), 2) if depths else 0.0

        return {
            "penetration_count": len(penetrations),
            "tracked_team": tracked_team,
            "opponent_team": opponent_team,
            "attacking_direction": attacking_direction,
            "avg_defense_line_x": avg_defense_line_x,
            "max_penetration_depth_m": max_depth,
            "penetrations": penetrations,
            "events": events[:50],
        }

    def analyze_defensive_line(
        self,
        tracks: Dict[str, Any],
        tracked_bboxes: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[str, Any]:
        """
        One-stop end-to-end analysis of team attack orientation, defensive line series,
        and penetration events.
        """
        tracked_teams = []
        players_tracks = tracks.get("players", [])
        for i in range(len(players_tracks)):
            if i not in tracked_bboxes:
                continue
            samurai_bbox = tracked_bboxes[i]
            if len(samurai_bbox) == 4:
                sx, sy, sw, sh = samurai_bbox
                target_center = (sx + sw / 2.0, sy + sh / 2.0)
                max_dist = max(150.0, max(sw, sh) * 0.8)
            else:
                target_center = (float(samurai_bbox[0]), float(samurai_bbox[1]))
                max_dist = 150.0

            best_dist, matched_info = max_dist, None
            for pid, pinfo in players_tracks[i].items():
                if not pinfo:
                    continue
                bb = pinfo.get("bbox")
                if bb and len(bb) == 4:
                    cx = (bb[0] + bb[2]) / 2.0
                    cy = (bb[1] + bb[3]) / 2.0
                    dist = ((cx - target_center[0])**2 + (cy - target_center[1])**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        matched_info = pinfo
            if matched_info and matched_info.get("team"):
                tracked_teams.append(matched_info["team"])

        tracked_team = int(np.median(tracked_teams)) if tracked_teams else 1
        opponent_team = 2 if tracked_team == 1 else 1

        attack_dir = self.resolve_attack_direction(tracks, tracked_team, opponent_team)
        frame_data = self.compute_defensive_line_series(
            tracks, tracked_bboxes, tracked_team, opponent_team, attack_dir
        )
        res = self.detect_penetration_events(frame_data, attack_dir, tracked_team, opponent_team)
        res["frame_data"] = frame_data
        return res

    def render_visualization(
        self,
        frame_data: List[Optional[Tuple[float, float, Tuple[float, float]]]],
        result: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Renders a publication-grade tactical pitch diagram with defensive line,
        player trajectory, penetration explosion markers, and attack orientation arrow.
        """
        BG = "#1a1a2e"
        GREEN = "#2d6a1e"
        RED = "#e74c3c"
        CYAN = "#00e5ff"
        GOLD = "#f1c40f"

        track_pts = [fd[2] for fd in frame_data if fd is not None]
        penetrations = result.get("penetrations", [])
        direction = result.get("attacking_direction", "right")
        pen_count = result.get("penetration_count", 0)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
        ax.set_facecolor(GREEN)

        # Draw pitch markings (105m x 68m standard UEFA dimensions)
        for rect in [
            plt.Rectangle((0, 0), 105, 68, fill=False, ec="white", lw=2),
            plt.Rectangle((0, 13.84), 16.5, 40.32, fill=False, ec="white"),
            plt.Rectangle((88.5, 13.84), 16.5, 40.32, fill=False, ec="white"),
            plt.Rectangle((0, 24.84), 5.5, 18.32, fill=False, ec="white"),
            plt.Rectangle((99.5, 24.84), 5.5, 18.32, fill=False, ec="white"),
        ]:
            ax.add_patch(rect)

        # Center line and circle
        ax.axvline(x=52.5, color="white", lw=1.5)
        circle = plt.Circle((52.5, 34), 9.15, fill=False, ec="white", lw=1.5)
        ax.add_patch(circle)

        # Plot player trajectory
        if track_pts:
            tp = np.array(track_pts[::2])
            ax.plot(tp[:, 0], tp[:, 1], color="white", lw=1.5, alpha=0.7, label="Player Trajectory")

        # Defensive line (median)
        median_dx = result.get("avg_defense_line_x", 52.5)
        ax.axvline(x=median_dx, color=CYAN, lw=2.5, ls="--", label=f"Opponent Defense Line ({median_dx:.1f}m)")

        # Penetrations
        if penetrations:
            px = [p[0] for p in penetrations]
            py = [p[1] for p in penetrations]
            ax.scatter(px, py, c=RED, s=150, zorder=6, marker="*", edgecolor="gold", lw=1, label="Line Break Penetration")

        # Attack direction arrow
        if direction == "right":
            ax.annotate("", xy=(75, 63), xytext=(35, 63),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=3, mutation_scale=20))
            ax.text(55, 65, "ATTACK DIRECTION ->", color=GOLD, fontsize=11, fontweight="bold", ha="center")
        else:
            ax.annotate("", xy=(35, 63), xytext=(75, 63),
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=3, mutation_scale=20))
            ax.text(55, 65, "<- ATTACK DIRECTION", color=GOLD, fontsize=11, fontweight="bold", ha="center")

        ax.set_xlim(-2, 107)
        ax.set_ylim(-2, 70)
        ax.set_title(
            f"Defensive Line Penetration Analysis — {pen_count} Break(s) | Attack: {direction.upper()}",
            color="white", fontsize=13, fontweight="bold", pad=12
        )
        ax.legend(facecolor=BG, labelcolor="white", fontsize=9, loc="lower right")
        ax.tick_params(colors="white")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
