"""
team_compactness_engine.py - Tactical Convex Hull & Team Compactness / Pitch Dispersion Engine

Implements spatio-temporal team shape analytics grounded in Fernandez & Bornn (2018),
Moura et al. (2012), and FIFA Training Centre methodology.

Key capabilities:
1. Dynamic 2D Convex Hull: Outfield player convex hull polygon surface area (m^2)
   and perimeter (m), with automated goalkeeper exclusion to prevent shape skewing.
2. Tactical Stretch Index: Radial dispersion (average Euclidean distance of outfield
   players to team centroid), decomposed into longitudinal (length) and lateral (width) stretch.
3. Team Inter-Centroid Dynamics: Distance (m) between opposing team dynamic centroids,
   quantifying spatial confrontation and pressing block compression.
4. Phase-Aware Aggregation: Distinguishes team shape in possession (attacking expansion,
   typically 1200-2200 m^2) vs out of possession (defensive compactness, 400-800 m^2).
5. Dual-Panel Broadcast Visualization: Renders tactical pitch convex hull overlay alongside
   a longitudinal match compactness timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path as MplPath
except ImportError:
    plt = None


@dataclass
class TeamFrameMetrics:
    frame_idx: int
    team_id: int
    hull_area_m2: float
    hull_perimeter_m: float
    centroid_xy: Tuple[float, float]
    stretch_index_m: float
    length_m: float
    width_m: float
    player_count: int
    hull_vertices: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "team_id": int(self.team_id),
            "hull_area_m2": float(round(self.hull_area_m2, 2)),
            "hull_perimeter_m": float(round(self.hull_perimeter_m, 2)),
            "centroid_xy": [float(round(v, 2)) for v in self.centroid_xy],
            "stretch_index_m": float(round(self.stretch_index_m, 2)),
            "length_m": float(round(self.length_m, 2)),
            "width_m": float(round(self.width_m, 2)),
            "player_count": int(self.player_count),
            "hull_vertices": [[float(round(c[0], 2)), float(round(c[1], 2))] for c in self.hull_vertices],
        }


@dataclass
class FrameCompactnessSnapshot:
    frame_idx: int
    team1_metrics: TeamFrameMetrics
    team2_metrics: TeamFrameMetrics
    inter_centroid_dist_m: float
    in_possession_team: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "team1": self.team1_metrics.to_dict(),
            "team2": self.team2_metrics.to_dict(),
            "inter_centroid_dist_m": float(round(self.inter_centroid_dist_m, 2)),
            "in_possession_team": int(self.in_possession_team) if self.in_possession_team is not None else None,
        }


class TeamCompactnessEngine:
    """
    Computes dynamic team spatial footprint, convex hull area, and stretch index.
    Pitch coordinate space: Length: 105m ([-52.5, 52.5]), Width: 68m ([-34, 34]).
    """

    def __init__(self, fps: float = 25.0, exclude_goalkeeper: bool = True):
        self.fps = fps
        self.exclude_goalkeeper = exclude_goalkeeper

    def _graham_scan_convex_hull(self, points: List[Tuple[float, float]]) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Pure Python fallback for 2D Convex Hull area, perimeter, and vertices.
        """
        pts = sorted(set(points))
        if len(pts) < 3:
            return 0.0, 0.0, pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull_pts = lower[:-1] + upper[:-1]
        n = len(hull_pts)
        if n < 3:
            return 0.0, 0.0, hull_pts

        # Shoelace formula for area
        area = 0.0
        perimeter = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull_pts[i][0] * hull_pts[j][1] - hull_pts[j][0] * hull_pts[i][1]
            perimeter += math.hypot(hull_pts[i][0] - hull_pts[j][0], hull_pts[i][1] - hull_pts[j][1])
        area = abs(area) / 2.0
        return area, perimeter, hull_pts

    def compute_team_shape(
        self,
        frame_idx: int,
        team_id: int,
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int],
        attacking_direction: Optional[int] = None,
    ) -> TeamFrameMetrics:
        """
        Computes convex hull, centroid, and stretch index for a single team at frame_idx.
        """
        team_members = [
            (pid, pos)
            for pid, pos in player_positions.items()
            if player_teams.get(pid) == team_id
        ]

        if not team_members:
            return TeamFrameMetrics(
                frame_idx=frame_idx,
                team_id=team_id,
                hull_area_m2=0.0,
                hull_perimeter_m=0.0,
                centroid_xy=(0.0, 0.0),
                stretch_index_m=0.0,
                length_m=0.0,
                width_m=0.0,
                player_count=0,
                hull_vertices=[],
            )

        # Exclude goalkeeper (isolated deepest player towards defending goal)
        coords = [pos for _, pos in team_members]
        if self.exclude_goalkeeper and len(coords) >= 5:
            if attacking_direction is not None:
                defending_goal_x = -52.5 if attacking_direction > 0 else 52.5
            else:
                sorted_xs = sorted(c[0] for c in coords)
                left_gap = sorted_xs[1] - sorted_xs[0]
                right_gap = sorted_xs[-1] - sorted_xs[-2]
                defending_goal_x = -52.5 if left_gap >= right_gap else 52.5

            coords_sorted = sorted(coords, key=lambda c: abs(c[0] - defending_goal_x))
            outfield_coords = coords_sorted[1:]  # Exclude the deepest player (GK)
        else:
            outfield_coords = coords

        n = len(outfield_coords)
        if n == 0:
            outfield_coords = coords
            n = len(outfield_coords)

        xs = [c[0] for c in outfield_coords]
        ys = [c[1] for c in outfield_coords]
        centroid_x = sum(xs) / n
        centroid_y = sum(ys) / n
        centroid = (centroid_x, centroid_y)

        length_m = max(xs) - min(xs) if xs else 0.0
        width_m = max(ys) - min(ys) if ys else 0.0

        # Stretch Index: average Euclidean distance to centroid
        radial_dists = [math.hypot(x - centroid_x, y - centroid_y) for x, y in zip(xs, ys)]
        stretch_index = sum(radial_dists) / n if n else 0.0

        # 2D Convex Hull
        if n >= 3:
            if ConvexHull is not None:
                try:
                    pts_np = np.array(outfield_coords)
                    hull = ConvexHull(pts_np)
                    hull_area = float(hull.volume)  # In 2D, hull.volume is the 2D area
                    hull_perimeter = float(hull.area)  # In 2D, hull.area is the perimeter
                    hull_vertices = [tuple(pts_np[v]) for v in hull.vertices]
                except Exception:
                    hull_area, hull_perimeter, hull_vertices = self._graham_scan_convex_hull(outfield_coords)
            else:
                hull_area, hull_perimeter, hull_vertices = self._graham_scan_convex_hull(outfield_coords)
        else:
            hull_area = 0.0
            hull_perimeter = 0.0
            hull_vertices = outfield_coords

        return TeamFrameMetrics(
            frame_idx=frame_idx,
            team_id=team_id,
            hull_area_m2=hull_area,
            hull_perimeter_m=hull_perimeter,
            centroid_xy=centroid,
            stretch_index_m=stretch_index,
            length_m=length_m,
            width_m=width_m,
            player_count=len(coords),
            hull_vertices=hull_vertices,
        )

    def analyze_match_timeline(
        self,
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        player_teams: Dict[int, int],
        ball_possession_timeline: Optional[Dict[int, int]] = None,
        stride: int = 1,
    ) -> List[FrameCompactnessSnapshot]:
        """
        Processes multi-frame tracking sequence and returns compactness snapshots.
        """
        frames = sorted(player_trajectories.keys())
        snapshots: List[FrameCompactnessSnapshot] = []

        for fi in frames[::stride]:
            f_players = player_trajectories[fi]
            t1 = self.compute_team_shape(fi, 1, f_players, player_teams)
            t2 = self.compute_team_shape(fi, 2, f_players, player_teams)

            inter_dist = math.hypot(
                t1.centroid_xy[0] - t2.centroid_xy[0],
                t1.centroid_xy[1] - t2.centroid_xy[1],
            )
            possession_team = ball_possession_timeline.get(fi) if ball_possession_timeline else None

            snapshots.append(
                FrameCompactnessSnapshot(
                    frame_idx=fi,
                    team1_metrics=t1,
                    team2_metrics=t2,
                    inter_centroid_dist_m=inter_dist,
                    in_possession_team=possession_team,
                )
            )

        return snapshots

    def generate_compactness_summary(
        self,
        snapshots: List[FrameCompactnessSnapshot],
    ) -> Dict[str, Any]:
        """
        Aggregates match-wide tactical compactness metrics for reporting and AI coach.
        """
        if not snapshots:
            return {
                "team1": {"mean_hull_area_m2": 0.0, "mean_stretch_index_m": 0.0},
                "team2": {"mean_hull_area_m2": 0.0, "mean_stretch_index_m": 0.0},
                "mean_inter_centroid_dist_m": 0.0,
            }

        t1_areas = [s.team1_metrics.hull_area_m2 for s in snapshots if s.team1_metrics.hull_area_m2 > 0]
        t2_areas = [s.team2_metrics.hull_area_m2 for s in snapshots if s.team2_metrics.hull_area_m2 > 0]
        t1_stretches = [s.team1_metrics.stretch_index_m for s in snapshots if s.team1_metrics.stretch_index_m > 0]
        t2_stretches = [s.team2_metrics.stretch_index_m for s in snapshots if s.team2_metrics.stretch_index_m > 0]
        inter_dists = [s.inter_centroid_dist_m for s in snapshots if s.inter_centroid_dist_m > 0]

        # In possession vs defending areas
        t1_poss_areas = [s.team1_metrics.hull_area_m2 for s in snapshots if s.in_possession_team == 1 and s.team1_metrics.hull_area_m2 > 0]
        t1_def_areas = [s.team1_metrics.hull_area_m2 for s in snapshots if s.in_possession_team == 2 and s.team1_metrics.hull_area_m2 > 0]
        t2_poss_areas = [s.team2_metrics.hull_area_m2 for s in snapshots if s.in_possession_team == 2 and s.team2_metrics.hull_area_m2 > 0]
        t2_def_areas = [s.team2_metrics.hull_area_m2 for s in snapshots if s.in_possession_team == 1 and s.team2_metrics.hull_area_m2 > 0]

        def _safe_mean(lst: List[float]) -> float:
            return float(round(sum(lst) / len(lst), 2)) if lst else 0.0

        def _safe_max(lst: List[float]) -> float:
            return float(round(max(lst), 2)) if lst else 0.0

        def _safe_min(lst: List[float]) -> float:
            return float(round(min(lst), 2)) if lst else 0.0

        return {
            "total_frames_analyzed": len(snapshots),
            "team1": {
                "mean_hull_area_m2": _safe_mean(t1_areas),
                "max_hull_area_m2": _safe_max(t1_areas),
                "min_hull_area_m2": _safe_min(t1_areas),
                "mean_stretch_index_m": _safe_mean(t1_stretches),
                "possession_phase_area_m2": _safe_mean(t1_poss_areas),
                "defensive_phase_area_m2": _safe_mean(t1_def_areas),
                "tactical_expansion_ratio": round((_safe_mean(t1_poss_areas) / _safe_mean(t1_def_areas)), 2) if _safe_mean(t1_def_areas) > 0 else 1.0,
            },
            "team2": {
                "mean_hull_area_m2": _safe_mean(t2_areas),
                "max_hull_area_m2": _safe_max(t2_areas),
                "min_hull_area_m2": _safe_min(t2_areas),
                "mean_stretch_index_m": _safe_mean(t2_stretches),
                "possession_phase_area_m2": _safe_mean(t2_poss_areas),
                "defensive_phase_area_m2": _safe_mean(t2_def_areas),
                "tactical_expansion_ratio": round((_safe_mean(t2_poss_areas) / _safe_mean(t2_def_areas)), 2) if _safe_mean(t2_def_areas) > 0 else 1.0,
            },
            "inter_team": {
                "mean_centroid_distance_m": _safe_mean(inter_dists),
                "min_centroid_distance_m": _safe_min(inter_dists),
                "max_centroid_distance_m": _safe_max(inter_dists),
            },
        }

    def render_compactness_dashboard(
        self,
        snapshots: List[FrameCompactnessSnapshot],
        sample_frame_idx: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> Optional[np.ndarray]:
        """
        Renders dual-panel dashboard:
        - Top: 2D Pitch with Team 1 and Team 2 Convex Hull Polygons and Centroid vectors.
        - Bottom: Match timeline of Team Convex Hull Area (m^2) and Inter-Centroid Distance.
        """
        if plt is None or not snapshots:
            return None

        # Pick sample snapshot
        if sample_frame_idx is not None:
            chosen = next((s for s in snapshots if s.frame_idx == sample_frame_idx), snapshots[len(snapshots)//2])
        else:
            chosen = snapshots[len(snapshots)//2]

        fig, (ax_pitch, ax_time) = plt.subplots(2, 1, figsize=figsize, facecolor="#0f172a", gridspec_kw={"height_ratios": [1.4, 1.0]})

        # --- PANEL 1: 2D PITCH CONVEX HULL OVERLAY ---
        ax_pitch.set_facecolor("#15803d")
        hl, hw = 52.5, 34.0
        # Pitch lines
        ax_pitch.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color="white", lw=2)
        ax_pitch.plot([0, 0], [-hw, hw], color="white", lw=1.5, linestyle="--", alpha=0.7)
        circle = patches.Circle((0, 0), 9.15, fill=False, color="white", lw=1.5, alpha=0.7)
        ax_pitch.add_patch(circle)

        # Team 1 Polygon (Blue)
        t1 = chosen.team1_metrics
        if len(t1.hull_vertices) >= 3:
            poly1 = patches.Polygon(t1.hull_vertices, closed=True, facecolor="#3b82f6", edgecolor="#60a5fa", lw=2, alpha=0.35, label=f"Team 1 Hull ({t1.hull_area_m2:.0f} m²)")
            ax_pitch.add_patch(poly1)
            # Scatter vertices
            vx = [v[0] for v in t1.hull_vertices]
            vy = [v[1] for v in t1.hull_vertices]
            ax_pitch.scatter(vx, vy, color="#60a5fa", s=60, zorder=5)

        # Team 2 Polygon (Red)
        t2 = chosen.team2_metrics
        if len(t2.hull_vertices) >= 3:
            poly2 = patches.Polygon(t2.hull_vertices, closed=True, facecolor="#ef4444", edgecolor="#f87171", lw=2, alpha=0.35, label=f"Team 2 Hull ({t2.hull_area_m2:.0f} m²)")
            ax_pitch.add_patch(poly2)
            vx = [v[0] for v in t2.hull_vertices]
            vy = [v[1] for v in t2.hull_vertices]
            ax_pitch.scatter(vx, vy, color="#f87171", s=60, zorder=5)

        # Centroids
        ax_pitch.scatter([t1.centroid_xy[0]], [t1.centroid_xy[1]], color="#2563eb", s=180, edgecolors="white", lw=2, zorder=6)
        ax_pitch.scatter([t2.centroid_xy[0]], [t2.centroid_xy[1]], color="#dc2626", s=180, edgecolors="white", lw=2, zorder=6)
        # Inter-centroid connection
        ax_pitch.plot([t1.centroid_xy[0], t2.centroid_xy[0]], [t1.centroid_xy[1], t2.centroid_xy[1]], color="#facc15", lw=2, linestyle=":", label=f"Centroid Gap ({chosen.inter_centroid_dist_m:.1f}m)")

        ax_pitch.set_xlim(-hl - 3, hl + 3)
        ax_pitch.set_ylim(-hw - 3, hw + 3)
        ax_pitch.set_aspect("equal")
        ax_pitch.set_title(f"TACTICAL CONVEX HULL & TEAM COMPACTNESS (Frame {chosen.frame_idx})", color="white", fontsize=12, fontweight="bold", pad=10)
        ax_pitch.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)
        ax_pitch.axis("off")

        # --- PANEL 2: LONGITUDINAL MATCH TIMELINE ---
        ax_time.set_facecolor("#1e293b")
        times_sec = [s.frame_idx / self.fps for s in snapshots]
        areas_t1 = [s.team1_metrics.hull_area_m2 for s in snapshots]
        areas_t2 = [s.team2_metrics.hull_area_m2 for s in snapshots]
        centroid_gaps = [s.inter_centroid_dist_m for s in snapshots]

        ax_time.plot(times_sec, areas_t1, color="#60a5fa", lw=2, label="Team 1 Area (m²)")
        ax_time.plot(times_sec, areas_t2, color="#f87171", lw=2, label="Team 2 Area (m²)")
        ax_time.set_ylabel("Convex Hull Area (m²)", color="white", fontsize=10)
        ax_time.tick_params(colors="white")
        ax_time.grid(color="#334155", linestyle="--", alpha=0.5)

        # Secondary axis for centroid gap
        ax_gap = ax_time.twinx()
        ax_gap.plot(times_sec, centroid_gaps, color="#facc15", lw=1.5, linestyle="--", alpha=0.8, label="Centroid Gap (m)")
        ax_gap.set_ylabel("Centroid Distance (m)", color="#facc15", fontsize=10)
        ax_gap.tick_params(colors="#facc15")

        ax_time.set_xlabel("Match Time (seconds)", color="white", fontsize=10)
        ax_time.set_title("TEMPORAL COMPACTNESS DYNAMICS (Compression vs Expansion)", color="white", fontsize=11, fontweight="bold", pad=8)

        # Combined legend
        lines_1, labels_1 = ax_time.get_legend_handles_labels()
        lines_2, labels_2 = ax_gap.get_legend_handles_labels()
        ax_time.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            return None

        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())
        except AttributeError:
            rgba = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return rgba
