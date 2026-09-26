"""
dynamic_formation_line_analyzer.py - Dynamic Formation & Inter-Line Spatial Spacing Analyzer

Theoretical & Mathematical Framework:
- Bialkowski et al. (2014) "Large-Scale Analysis of Formations in Soccer" & Shaw & Glickman (2019):
  Soccer formations are fluid, time-varying spatial configurations that adapt dynamically
  between attacking possession and defensive phases.
- Outfield Player Spatial Clustering:
  1. Decouples the isolated goalkeeper via boundary spatial isolation gap analysis (PITFALL-03)
     under oriented attacking direction (+X or -X).
  2. Projects 10 outfield player positions onto the longitudinal depth axis and clusters them
     into 3 or 4 tactical lines (Defense, Midfield, Attack / Defensive Midfield, Attacking Midfield).
  3. Classifies dynamic formation strings (e.g., '4-3-3', '4-2-3-1', '4-4-2', '3-2-5', '5-3-2').
  4. Quantifies inter-line spacing (d_def_mid, d_mid_fwd), total team depth, and line lateral widths.
  5. Spots "Space Between the Lines" vulnerability when inter-line distance expands beyond 20m.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TacticalLine:
    line_name: str  # "Defensive Line", "Midfield Line", "Forward Line", etc.
    line_index: int  # 0: defense, 1: midfield, 2: attack
    player_ids: List[Union[int, str]]
    count: int
    mean_depth_x: float  # Longitudinal position along pitch length
    min_depth_x: float
    max_depth_x: float
    width_y: float  # Lateral spread between outermost players in this line
    y_centroid: float  # Lateral center of mass
    player_coords: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DynamicFormationSnapshot:
    team_id: Union[int, str]
    attacking_direction: str  # "+X" or "-X"
    formation_name: str  # e.g., "4-3-3", "4-2-3-1", "4-4-2", "3-2-5"
    line_counts: List[int]  # e.g., [4, 3, 3]
    tactical_lines: List[TacticalLine]
    goalkeeper_id: Optional[Union[int, str]]
    inter_line_distances: List[float]  # [d_def_to_mid, d_mid_to_fwd, ...]
    total_team_depth_m: float  # distance from deepest defender to highest forward
    mean_line_width_m: float
    space_between_lines_vulnerability: bool  # True if any inter-line gap > 20m
    max_inter_line_gap_m: float
    timestamp_sec: Optional[float] = None
    frame_idx: Optional[int] = None


class DynamicFormationAnalyzer:
    """
    Identifies dynamic team formations and inter-line spacing from 2D player tracking data.
    """

    def __init__(
        self,
        min_line_gap_m: float = 6.0,
        max_inter_line_safe_gap_m: float = 20.0,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        self.min_line_gap = float(min_line_gap_m)
        self.max_safe_gap = float(max_inter_line_safe_gap_m)
        self.half_len = pitch_length_m / 2.0
        self.half_wid = pitch_width_m / 2.0

    def isolate_goalkeeper(
        self,
        player_positions: Dict[Union[int, str], Tuple[float, float]],
        attacking_direction: str = "+X",
    ) -> Tuple[Optional[Union[int, str]], Dict[Union[int, str], Tuple[float, float]]]:
        """
        Isolates the goalkeeper from outfield players using boundary gap analysis (PITFALL-03).
        In +X attacking direction, defending goal is at x = -52.5m, keeper is furthest left.
        In -X attacking direction, defending goal is at x = +52.5m, keeper is furthest right.
        """
        if len(player_positions) <= 1:
            return None, dict(player_positions)

        att_dir = str(attacking_direction).strip().upper()
        reverse = att_dir in ("-X", "NEGATIVE", "LEFT")

        # Projected depth: in direction towards opponent goal
        # If attacking +X: defending goal is negative x, so sorted by x ascending (deepest defender first)
        # If attacking -X: defending goal is positive x, so sorted by x descending
        items = list(player_positions.items())
        if not reverse:
            sorted_players = sorted(items, key=lambda item: item[1][0])
        else:
            sorted_players = sorted(items, key=lambda item: -item[1][0])

        if len(sorted_players) < 4:
            return None, dict(player_positions)

        # Check gap between #1 (deepest) and #2
        p1_id, p1_pos = sorted_players[0]
        p2_id, p2_pos = sorted_players[1]
        p1_depth = p1_pos[0] if not reverse else -p1_pos[0]
        p2_depth = p2_pos[0] if not reverse else -p2_pos[0]
        deepest_gap = p2_depth - p1_depth

        # If gap >= 10.0m and player is in defensive third (or len == 11), they are isolated goalkeeper
        is_in_def_third = p1_depth < -20.0
        if (len(sorted_players) >= 11 and deepest_gap >= 7.0) or (deepest_gap >= 12.0 and is_in_def_third):
            keeper_id = p1_id
            outfield = {pid: pos for pid, pos in sorted_players[1:]}
            return keeper_id, outfield

        return None, dict(player_positions)

    def cluster_into_lines(
        self,
        outfield_players: Dict[Union[int, str], Tuple[float, float]],
        attacking_direction: str = "+X",
        target_lines: int = 3,
    ) -> List[TacticalLine]:
        """
        Clusters 10 outfield players into tactical lines along attacking longitudinal axis.
        """
        if not outfield_players:
            return []

        reverse = str(attacking_direction).strip().upper() in ("-X", "NEGATIVE", "LEFT")
        items = list(outfield_players.items())

        # Sort outfield players from defense to attack
        if not reverse:
            sorted_players = sorted(items, key=lambda item: item[1][0])
        else:
            sorted_players = sorted(items, key=lambda item: -item[1][0])

        n_players = len(sorted_players)
        if n_players < 3:
            # Degenerate case: single line
            coords = [p[1] for p in sorted_players]
            ys = [c[1] for c in coords]
            xs = [c[0] for c in coords]
            return [
                TacticalLine(
                    line_name="Single Line",
                    line_index=0,
                    player_ids=[p[0] for p in sorted_players],
                    count=n_players,
                    mean_depth_x=float(np.mean(xs)),
                    min_depth_x=float(np.min(xs)),
                    max_depth_x=float(np.max(xs)),
                    width_y=float(np.max(ys) - np.min(ys)) if ys else 0.0,
                    y_centroid=float(np.mean(ys)) if ys else 0.0,
                    player_coords=coords,
                )
            ]

        depths = np.array([p[1][0] if not reverse else -p[1][0] for p in sorted_players])
        consecutive_gaps = np.diff(depths)

        # 1D line splitting based on largest depth gaps
        # We need (target_lines - 1) split cutoffs
        num_splits = min(target_lines - 1, max(1, n_players - 1))
        # Find indices of largest gaps
        largest_gap_indices = np.argsort(consecutive_gaps)[-num_splits:]
        split_indices = sorted([int(idx) + 1 for idx in largest_gap_indices])

        # Slice sorted players into lines
        line_slices = []
        prev_idx = 0
        for s_idx in split_indices:
            line_slices.append(sorted_players[prev_idx:s_idx])
            prev_idx = s_idx
        line_slices.append(sorted_players[prev_idx:])

        # Build TacticalLine objects
        line_names_3 = ["Defensive Line", "Midfield Line", "Forward Line"]
        line_names_4 = ["Defensive Line", "Defensive Midfield", "Attacking Midfield", "Forward Line"]

        lines: List[TacticalLine] = []
        for i, slice_players in enumerate(line_slices):
            if not slice_players:
                continue
            p_ids = [p[0] for p in slice_players]
            coords = [p[1] for p in slice_players]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            if len(line_slices) == 3:
                l_name = line_names_3[min(i, 2)]
            elif len(line_slices) == 4:
                l_name = line_names_4[min(i, 3)]
            else:
                l_name = f"Tactical Line {i + 1}"

            lines.append(
                TacticalLine(
                    line_name=l_name,
                    line_index=i,
                    player_ids=p_ids,
                    count=len(p_ids),
                    mean_depth_x=round(float(np.mean(xs)), 2),
                    min_depth_x=round(float(np.min(xs)), 2),
                    max_depth_x=round(float(np.max(xs)), 2),
                    width_y=round(float(np.max(ys) - np.min(ys)), 2),
                    y_centroid=round(float(np.mean(ys)), 2),
                    player_coords=coords,
                )
            )

        return lines

    def evaluate_frame(
        self,
        player_positions: Dict[Union[int, str], Tuple[float, float]],
        team_id: Union[int, str] = 1,
        attacking_direction: str = "+X",
        timestamp_sec: Optional[float] = None,
        frame_idx: Optional[int] = None,
    ) -> DynamicFormationSnapshot:
        """
        Evaluates dynamic formation structure for a single frame of player positions.
        """
        keeper_id, outfield = self.isolate_goalkeeper(player_positions, attacking_direction)

        # Decide whether 3 lines or 4 lines best fits the depth profile
        # Test 3 lines first
        lines_3 = self.cluster_into_lines(outfield, attacking_direction, target_lines=3)
        lines = lines_3

        # Compute line counts: e.g. [4, 3, 3] or [4, 4, 2]
        line_counts = [line.count for line in lines]
        formation_str = "-".join(str(c) for c in line_counts)

        reverse = str(attacking_direction).strip().upper() in ("-X", "NEGATIVE", "LEFT")
        # Inter-line longitudinal distances
        inter_line_distances: List[float] = []
        for i in range(len(lines) - 1):
            if not reverse:
                gap = lines[i + 1].mean_depth_x - lines[i].mean_depth_x
            else:
                gap = lines[i].mean_depth_x - lines[i + 1].mean_depth_x
            inter_line_distances.append(round(max(0.0, gap), 2))

        # Overall team depth: from deepest defender to highest forward
        if lines:
            if not reverse:
                total_depth = max(0.0, lines[-1].max_depth_x - lines[0].min_depth_x)
            else:
                total_depth = max(0.0, lines[0].max_depth_x - lines[-1].min_depth_x)
            mean_width = float(np.mean([l.width_y for l in lines]))
        else:
            total_depth = 0.0
            mean_width = 0.0

        max_gap = max(inter_line_distances) if inter_line_distances else 0.0
        is_vulnerable = max_gap > self.max_safe_gap

        return DynamicFormationSnapshot(
            team_id=team_id,
            attacking_direction=attacking_direction,
            formation_name=formation_str,
            line_counts=line_counts,
            tactical_lines=lines,
            goalkeeper_id=keeper_id,
            inter_line_distances=inter_line_distances,
            total_team_depth_m=round(total_depth, 2),
            mean_line_width_m=round(mean_width, 2),
            space_between_lines_vulnerability=is_vulnerable,
            max_inter_line_gap_m=round(max_gap, 2),
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
        )

    def render_formation_tactical_diagram(
        self,
        snapshot: DynamicFormationSnapshot,
        output_path: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Renders a broadcast 2D pitch view displaying the tactical formation lines,
        player positions, inter-line distances, and formation badge.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="#0B132B")
        ax.set_facecolor("#0B132B")

        # Pitch markings
        pitch_col = "#2A3B5C"
        lw = 1.2
        ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color=pitch_col, lw=lw)
        ax.plot([0, 0], [-34, 34], color=pitch_col, lw=lw)
        ax.add_patch(Circle((0, 0), 9.15, edgecolor=pitch_col, facecolor="none", lw=lw))
        ax.plot([-52.5, -36.0, -36.0, -52.5], [-20.16, -20.16, 20.16, 20.16], color=pitch_col, lw=lw)
        ax.plot([52.5, 36.0, 36.0, 52.5], [-20.16, -20.16, 20.16, 20.16], color=pitch_col, lw=lw)

        line_colors = ["#4A90E2", "#50E3C2", "#F5A623", "#FF6B6B"]

        # Draw lines and players
        for i, line in enumerate(snapshot.tactical_lines):
            col = line_colors[i % len(line_colors)]
            coords = sorted(line.player_coords, key=lambda c: c[1])
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            # Connect players within the same line
            if len(xs) >= 2:
                ax.plot(xs, ys, color=col, linestyle="--", lw=1.8, alpha=0.85, zorder=3)

            # Draw player markers
            ax.scatter(xs, ys, s=260, color=col, edgecolors="#FFFFFF", lw=1.5, zorder=5)
            for pid, (px, py) in zip(line.player_ids, coords):
                ax.text(px, py, str(pid), color="#FFFFFF", fontsize=9, fontweight="bold",
                        ha="center", va="center", zorder=6)

            # Label line
            line_lbl = f"{line.line_name} ({line.count})"
            ax.text(line.mean_depth_x, min(ys) - 4.5, line_lbl, color=col, fontsize=9,
                    fontweight="bold", ha="center", va="top")

        # Draw goalkeeper if isolated
        if snapshot.goalkeeper_id is not None:
            # Look up keeper coords in reverse or standard
            k_x = -45.0 if snapshot.attacking_direction == "+X" else 45.0
            ax.scatter([k_x], [0.0], s=260, color="#E94A59", edgecolors="#FFFFFF", lw=1.5, zorder=5)
            ax.text(k_x, 0.0, str(snapshot.goalkeeper_id), color="#FFFFFF", fontsize=9,
                    fontweight="bold", ha="center", va="center", zorder=6)
            ax.text(k_x, -4.5, "Goalkeeper", color="#E94A59", fontsize=9,
                    fontweight="bold", ha="center", va="top")

        # Inter-line distance annotations
        for i, gap in enumerate(snapshot.inter_line_distances):
            if i + 1 < len(snapshot.tactical_lines):
                x1 = snapshot.tactical_lines[i].mean_depth_x
                x2 = snapshot.tactical_lines[i + 1].mean_depth_x
                mid_x = (x1 + x2) / 2.0
                gap_col = "#FF6B6B" if gap > 20.0 else "#E0E6ED"
                ax.annotate(
                    f"Gap: {gap:.1f}m",
                    xy=(mid_x, 26.0 - i * 5.0),
                    color=gap_col,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#16223F", edgecolor=gap_col, lw=1.0),
                )

        chart_title = title or f"Team {snapshot.team_id} Tactical Shape: {snapshot.formation_name} (Depth: {snapshot.total_team_depth_m:.1f}m)"
        ax.set_title(chart_title, color="#FFFFFF", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlim(-55, 55)
        ax.set_ylim(36, -36)
        ax.set_xticks([])
        ax.set_yticks([])

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return output_path
