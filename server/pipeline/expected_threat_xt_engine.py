"""
expected_threat_xt_engine.py - Expected Threat (xT) Pitch Discretization & Action Valuator

Mathematical Framework:
- Karun Singh (2018/2019) possession-value model solving recursive Bellman payoff equation:
    xT(z) = (s_z * g_z) + (m_z * sum_{z'} T_{z -> z'} * xT(z'))
  where:
    - z = (cx, cy) is a discrete pitch zone in a 16x12 grid (192 zones).
    - s_z is probability of shooting from zone z.
    - g_z is probability of scoring given a shot from zone z.
    - m_z = 1.0 - s_z is probability of moving (passing / carrying).
    - T_{z -> z'} is the Markov transition matrix between pitch cells.
- Evaluates marginal threat delta for any on-ball action:
    Delta xT = xT(z_end) - xT(z_start)
- Distinguishes progressive playmakers who advance the ball into high-threat zones (e.g., Zone 14, half-spaces)
  even when the action does not directly result in a shot or goal assist.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# Standard pitch dimensions in meters (FIFA recommended)
DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0

# Standard xT grid resolution: 16 columns (length) x 12 rows (width)
DEFAULT_GRID_COLS = 16
DEFAULT_GRID_ROWS = 12


@dataclass
class ThreatAction:
    """Represents an on-ball action evaluated for Expected Threat."""
    action_type: str  # "pass", "carry", "cross", "dribble"
    player_id: Union[int, str]
    team_id: Union[int, str]
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    start_zone: Tuple[int, int]
    end_zone: Tuple[int, int]
    start_xt: float
    end_xt: float
    delta_xt: float
    is_progressive: bool  # True if delta_xt > 0.015
    timestamp_sec: Optional[float] = None
    frame_idx: Optional[int] = None


@dataclass
class PlayerThreatSummary:
    """Aggregated threat metrics for an individual player."""
    player_id: Union[int, str]
    team_id: Union[int, str]
    actions_count: int = 0
    progressive_actions_count: int = 0
    total_xt_created: float = 0.0
    xt_from_passes: float = 0.0
    xt_from_carries: float = 0.0
    max_single_action_xt: float = 0.0
    threat_share_pct: float = 0.0


class ExpectedThreatEngine:
    """
    16x12 Expected Threat (xT) model with Markov value iteration.
    """

    def __init__(
        self,
        cols: int = DEFAULT_GRID_COLS,
        rows: int = DEFAULT_GRID_ROWS,
        pitch_length: float = DEFAULT_PITCH_LENGTH_M,
        pitch_width: float = DEFAULT_PITCH_WIDTH_M,
        retention_rate: float = 0.78,
        convergence_tol: float = 1e-6,
        max_iterations: int = 100,
    ) -> None:
        self.cols = int(cols)
        self.rows = int(rows)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.retention_rate = float(retention_rate)
        self.cell_w = self.pitch_length / self.cols
        self.cell_h = self.pitch_width / self.rows
        self.tol = float(convergence_tol)
        self.max_iter = int(max_iterations)

        # Coordinate bounds: supports centered [-52.5, 52.5] x [-34, 34]
        # and uncentered [0, 105] x [0, 68]
        self.x_min = -self.pitch_length / 2.0
        self.x_max = self.pitch_length / 2.0
        self.y_min = -self.pitch_width / 2.0
        self.y_max = self.pitch_width / 2.0

        # Initialize base probability matrices: shape (rows, cols)
        # Note: row 0 is top sideline (y = -34), row 11 is bottom sideline (y = +34)
        # col 0 is own goal line (x = -52.5), col 15 is opponent goal line (x = +52.5)
        self.shoot_prob = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.goal_prob = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.move_prob = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.xt_surface = np.zeros((self.rows, self.cols), dtype=np.float64)

        # Precompute transition matrix and solve Bellman equation
        self._build_action_probabilities()
        self._solve_value_iteration()

    def _build_action_probabilities(self) -> None:
        """
        Constructs empirical action probabilities (shoot, score|shoot, move) across all zones.
        Assumes standard attack direction: left-to-right (towards x = +52.5, y = 0.0).
        """
        target_goal_x = self.x_max
        target_goal_y = 0.0

        for r in range(self.rows):
            for c in range(self.cols):
                cx_m, cy_m = self.zone_to_coords(c, r)
                dx = target_goal_x - cx_m
                dy = target_goal_y - cy_m
                dist_to_goal = math.hypot(dx, dy)

                # Goal angle subtended between the two posts (7.32m wide)
                p1_y = -3.66
                p2_y = 3.66
                v1 = (target_goal_x - cx_m, p1_y - cy_m)
                v2 = (target_goal_x - cx_m, p2_y - cy_m)
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.hypot(v1[0], v1[1])
                mag2 = math.hypot(v2[0], v2[1])
                cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2 + 1e-9)))
                goal_angle = math.acos(cos_theta)

                # 1. Shoot probability s_z:
                # Players only shoot inside the final 35 meters, probability grows sharply within penalty box
                if cx_m >= 15.0:
                    s_raw = math.exp(-0.08 * max(0.0, dist_to_goal - 8.0)) * (goal_angle / (math.pi / 2.0))
                    self.shoot_prob[r, c] = max(0.01, min(0.70, s_raw))
                else:
                    self.shoot_prob[r, c] = 0.001

                # 2. Goal probability given shot g_z (xG from zone center):
                g_raw = math.exp(-0.10 * max(0.0, dist_to_goal - 4.0)) * (goal_angle / (math.pi / 2.0)) ** 0.7
                self.goal_prob[r, c] = max(0.005, min(0.75, g_raw))

                # 3. Move probability m_z:
                self.move_prob[r, c] = 1.0 - self.shoot_prob[r, c]

    def _build_local_transition_kernel(self, c: int, r: int) -> np.ndarray:
        """
        Generates transition probabilities T_{(c,r) -> (c',r')} from zone (c,r) to all other zones.
        Models football passing and carrying tendencies:
        - Distance decay: typical passes/carries travel 8m to 35m (Gaussian kernel around ~15m).
        - Forward progression bias: positive dx is favored over backward passes.
        - Central convergence bias: moves towards the central corridor (y ≈ 0) are preferred in attacking third.
        - Accounts for turnover / possession loss via self.retention_rate.
        """
        kernel = np.zeros((self.rows, self.cols), dtype=np.float64)
        c_x, c_y = self.zone_to_coords(c, r)

        for r_prime in range(self.rows):
            for c_prime in range(self.cols):
                if r_prime == r and c_prime == c:
                    continue  # Self-transitions are not moves

                cp_x, cp_y = self.zone_to_coords(c_prime, r_prime)
                dx = cp_x - c_x
                dy = cp_y - c_y
                dist = math.hypot(dx, dy)

                if dist > 45.0 or dist < 2.0:
                    continue

                # Gaussian distance preference peak around 15m
                dist_weight = math.exp(-0.5 * ((dist - 15.0) / 12.0) ** 2)

                # Forward progression factor:
                progression_factor = 2.0 if dx > 0 else (0.7 if dx == 0 else 0.3)

                # Central attacking corridor bonus in opponent half
                central_factor = 1.2 if (cp_x > 0 and abs(cp_y) < abs(c_y)) else 1.0

                kernel[r_prime, c_prime] = dist_weight * progression_factor * central_factor

        total_weight = np.sum(kernel)
        if total_weight > 0:
            kernel = (kernel / total_weight) * self.retention_rate
        else:
            kernel = np.ones((self.rows, self.cols), dtype=np.float64)
            kernel[r, c] = 0.0
            kernel = (kernel / np.sum(kernel)) * self.retention_rate

        return kernel

    def _solve_value_iteration(self) -> None:
        """
        Solves the Bellman equation via value iteration:
          xT^{(k+1)}(c, r) = s(c,r)*g(c,r) + m(c,r) * sum(T_{(c,r)->(c',r')} * xT^{(k)}(c',r'))
        """
        # Precompute transition kernels for all 192 cells
        kernels = {}
        for r in range(self.rows):
            for c in range(self.cols):
                kernels[(c, r)] = self._build_local_transition_kernel(c, r)

        # Direct payoff: immediate goal probability from shooting
        direct_payoff = self.shoot_prob * self.goal_prob

        # Initialize value with direct payoff
        v = direct_payoff.copy()

        for iteration in range(1, self.max_iter + 1):
            v_next = np.zeros_like(v)
            for r in range(self.rows):
                for c in range(self.cols):
                    expected_future_threat = np.sum(kernels[(c, r)] * v)
                    v_next[r, c] = direct_payoff[r, c] + self.move_prob[r, c] * expected_future_threat

            max_delta = float(np.max(np.abs(v_next - v)))
            v = v_next

            if max_delta < self.tol:
                log.debug("[xT] Value iteration converged in %d iterations (delta=%.2e)", iteration, max_delta)
                break

        # Normalize so baseline threat lies comfortably in [0.001, 0.45]
        self.xt_surface = np.clip(v, 0.0, 1.0)

    def coords_to_zone(
        self,
        x: float,
        y: float,
        attacking_direction: str = "+X",
        corner_origin: bool = False,
    ) -> Tuple[int, int]:
        """
        Maps physical metric coordinates to grid cell (col_idx, row_idx).
        Accepts:
          - Centered coordinates: x in [-52.5, 52.5], y in [-34, 34]
          - Uncentered coordinates: x in [0, 105], y in [0, 68] (via corner_origin=True or auto-detection)
        Accounts for attacking_direction:
          - '+X': attacking towards positive x (+52.5 or 105m) [standard]
          - '-X': attacking towards negative x (-52.5 or 0m) -> mirrored horizontally and vertically
        """
        # Convert uncentered [0, 105] to centered [-52.5, 52.5]
        if corner_origin or x > self.x_max or y > self.y_max:
            x = x - self.pitch_length / 2.0
            y = y - self.pitch_width / 2.0

        x = max(self.x_min, min(self.x_max, x))
        y = max(self.y_min, min(self.y_max, y))

        # Mirror coordinates if attacking in -X direction
        if str(attacking_direction).strip().upper() in ("-X", "NEGATIVE", "LEFT"):
            x = -x
            y = -y

        # Compute cell index
        col = int((x - self.x_min) / self.cell_w)
        row = int((y - self.y_min) / self.cell_h)

        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        return col, row

    def zone_to_coords(self, col: int, row: int) -> Tuple[float, float]:
        """Returns the physical center coordinates (x_m, y_m) of a zone in centered frame."""
        col = max(0, min(self.cols - 1, int(col)))
        row = max(0, min(self.rows - 1, int(row)))
        cx = self.x_min + (col + 0.5) * self.cell_w
        cy = self.y_min + (row + 0.5) * self.cell_h
        return cx, cy

    def get_xt(
        self,
        x: float,
        y: float,
        attacking_direction: str = "+X",
    ) -> float:
        """Look up the Expected Threat of a physical coordinate."""
        col, row = self.coords_to_zone(x, y, attacking_direction)
        return float(self.xt_surface[row, col])

    def evaluate_action(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        action_type: str = "pass",
        player_id: Union[int, str] = 0,
        team_id: Union[int, str] = 0,
        attacking_direction: str = "+X",
        timestamp_sec: Optional[float] = None,
        frame_idx: Optional[int] = None,
    ) -> ThreatAction:
        """
        Evaluates the net threat delta created by an on-ball action.
        Delta xT = xT(end_zone) - xT(start_zone)
        """
        c_start, r_start = self.coords_to_zone(start_x, start_y, attacking_direction)
        c_end, r_end = self.coords_to_zone(end_x, end_y, attacking_direction)

        xt_start = float(self.xt_surface[r_start, c_start])
        xt_end = float(self.xt_surface[r_end, c_end])
        delta_xt = xt_end - xt_start

        # An action is considered progressive if Delta xT >= 0.015
        is_progressive = delta_xt >= 0.015

        return ThreatAction(
            action_type=str(action_type),
            player_id=player_id,
            team_id=team_id,
            start_x=float(start_x),
            start_y=float(start_y),
            end_x=float(end_x),
            end_y=float(end_y),
            start_zone=(c_start, r_start),
            end_zone=(c_end, r_end),
            start_xt=xt_start,
            end_xt=xt_end,
            delta_xt=round(delta_xt, 6),
            is_progressive=is_progressive,
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
        )

    def evaluate_pass_network(
        self,
        pass_events: List[Dict[str, Any]],
        team_attacking_directions: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch evaluates a list of pass events and attributes xT to players and teams.
        """
        team_dirs = team_attacking_directions or {}
        evaluated_actions: List[ThreatAction] = []
        player_summaries: Dict[Any, PlayerThreatSummary] = {}
        team_xt_totals: Dict[Any, float] = {}

        for p in pass_events:
            passer_id = p.get("passer_id") or p.get("player_id") or p.get("from_player", 0)
            team_id = p.get("team_id") or p.get("team") or p.get("passer_team", 0)
            att_dir = team_dirs.get(team_id, "+X")

            start_x = p.get("start_x", p.get("x1", 0.0))
            start_y = p.get("start_y", p.get("y1", 0.0))
            end_x = p.get("end_x", p.get("x2", 0.0))
            end_y = p.get("end_y", p.get("y2", 0.0))
            t_sec = p.get("timestamp_sec") or p.get("time_sec")
            f_idx = p.get("frame_idx") or p.get("frame")

            action = self.evaluate_action(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                action_type="pass",
                player_id=passer_id,
                team_id=team_id,
                attacking_direction=att_dir,
                timestamp_sec=t_sec,
                frame_idx=f_idx,
            )
            evaluated_actions.append(action)

            # Update player summary
            if passer_id not in player_summaries:
                player_summaries[passer_id] = PlayerThreatSummary(
                    player_id=passer_id,
                    team_id=team_id,
                )
            psum = player_summaries[passer_id]
            psum.actions_count += 1
            if action.is_progressive:
                psum.progressive_actions_count += 1
            psum.total_xt_created += action.delta_xt
            psum.xt_from_passes += action.delta_xt
            psum.max_single_action_xt = max(psum.max_single_action_xt, action.delta_xt)

            # Update team total
            team_xt_totals[team_id] = team_xt_totals.get(team_id, 0.0) + action.delta_xt

        # Calculate threat share percentage
        for psum in player_summaries.values():
            team_tot = team_xt_totals.get(psum.team_id, 0.0)
            if team_tot > 0:
                psum.threat_share_pct = round((max(0.0, psum.total_xt_created) / team_tot) * 100.0, 1)

        # Sort top creators
        top_creators = sorted(
            player_summaries.values(),
            key=lambda x: x.total_xt_created,
            reverse=True,
        )

        return {
            "total_actions": len(evaluated_actions),
            "progressive_actions": sum(1 for a in evaluated_actions if a.is_progressive),
            "team_xt_totals": {str(k): round(v, 4) for k, v in team_xt_totals.items()},
            "top_threat_creators": [
                {
                    "player_id": p.player_id,
                    "team_id": p.team_id,
                    "actions_count": p.actions_count,
                    "progressive_count": p.progressive_actions_count,
                    "total_xt": round(p.total_xt_created, 4),
                    "max_action_xt": round(p.max_single_action_xt, 4),
                    "threat_share_pct": p.threat_share_pct,
                }
                for p in top_creators
            ],
            "actions": [
                {
                    "player_id": a.player_id,
                    "team_id": a.team_id,
                    "action_type": a.action_type,
                    "delta_xt": a.delta_xt,
                    "is_progressive": a.is_progressive,
                    "start_zone": a.start_zone,
                    "end_zone": a.end_zone,
                }
                for a in evaluated_actions
            ],
        }

    def render_xt_heatmap_plot(
        self,
        output_path: str,
        evaluated_actions: Optional[List[ThreatAction]] = None,
        title: str = "Expected Threat (xT) Pitch Surface & Threat Flow",
    ) -> str:
        """
        Renders a broadcast-quality matplotlib visualization of the 16x12 xT grid,
        overlaying progressive threat vectors if provided.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Arc, Circle

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0B132B")
        ax.set_facecolor("#0B132B")

        # Plot 2D xT surface as an image extent=[-52.5, 52.5, 34, -34]
        # Invert row order for standard Cartesian orientation (row 0 at top y=-34)
        im = ax.imshow(
            self.xt_surface,
            extent=[self.x_min, self.x_max, self.y_max, self.y_min],
            cmap="YlOrRd",
            alpha=0.65,
            aspect="equal",
            interpolation="bicubic",
        )

        # Draw pitch outline and markings
        pitch_color = "#FFFFFF"
        lw = 1.2
        # Outline
        ax.plot([self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
                [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min], color=pitch_color, lw=lw)
        # Halfway line
        ax.plot([0, 0], [self.y_min, self.y_max], color=pitch_color, lw=lw)
        # Center circle
        circle = Circle((0, 0), 9.15, edgecolor=pitch_color, facecolor="none", lw=lw)
        ax.add_patch(circle)
        # Penalty areas (16.5m x 40.32m)
        # Left penalty area
        ax.plot([self.x_min, self.x_min + 16.5, self.x_min + 16.5, self.x_min],
                [-20.16, -20.16, 20.16, 20.16], color=pitch_color, lw=lw)
        # Right penalty area
        ax.plot([self.x_max, self.x_max - 16.5, self.x_max - 16.5, self.x_max],
                [-20.16, -20.16, 20.16, 20.16], color=pitch_color, lw=lw)

        # Overlay progressive action arrows if provided
        if evaluated_actions:
            prog_actions = [a for a in evaluated_actions if a.is_progressive][:25]
            for a in prog_actions:
                ax.annotate(
                    "",
                    xy=(a.end_x, a.end_y),
                    xytext=(a.start_x, a.start_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#00F0FF",
                        lw=2.2,
                        mutation_scale=14,
                    ),
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Expected Threat (xT) Value", color="#E0E6ED", fontsize=11, fontweight="bold")
        cbar.ax.yaxis.set_tick_params(color="#E0E6ED")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#E0E6ED")

        ax.set_title(title, color="#FFFFFF", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(-55, 55)
        ax.set_ylim(36, -36)  # Inverted Y for TV perspective
        ax.set_xticks([])
        ax.set_yticks([])

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return output_path
