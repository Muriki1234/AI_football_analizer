"""
offside_var_evaluator.py - Automated VAR Offside Line & Infringement Detection Engine

Implements FIFA Law 11 (Offside) and Semi-Automated Offside Technology (SAOT)
geometric principles for 2D pitch-calibrated video tracking.

Key capabilities:
1. Kick-Point Freeze-Frame Evaluation (t_release): Evaluates player positioning
   at the exact moment a pass is released.
2. Second-Last Opponent Identification: Reliably identifies the deepest (often GK)
   and second-last defender, accounting for both attack directions (left-to-right, right-to-left).
3. FIFA Law 11 Constraints:
   - Halfway line exemption: No player can be offside in their own defensive half.
   - Ball reference: Attackers behind or level with the ball cannot be offside.
   - Metric Margin of Error: Calibrated tolerance threshold (default 0.05m / 5cm).
4. Metric Margin Quantification: Exact distance (meters) by which an attacker is
   ahead of or behind the legal offside line.
5. 2D Tactical VAR Visualization: Generates publication-quality pitch graphics
   with calibrated offside lines, second-last defender anchors, and metric measurement callouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    plt = None


@dataclass
class PlayerOffsideStatus:
    player_id: int
    team_id: int
    xy: Tuple[float, float]
    is_offside: bool
    margin_meters: float  # >0: ahead of line (offside), <0: behind line (onside)
    reason: str  # "OFFSIDE", "ONSIDE", "OWN_HALF", "BEHIND_BALL", "NO_DEFENDER"
    is_receiver: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": int(self.player_id),
            "team_id": int(self.team_id),
            "xy": [float(round(v, 2)) for v in self.xy],
            "is_offside": bool(self.is_offside),
            "margin_meters": float(round(self.margin_meters, 3)),
            "reason": self.reason,
            "is_receiver": bool(self.is_receiver),
        }


@dataclass
class OffsideEvaluationResult:
    frame_idx: int
    passer_id: int
    passer_team: int
    receiver_id: Optional[int]
    is_offside: bool
    decision: str  # "OFFSIDE", "ONSIDE", "NO_RECEIVER", "INVALID_TRACK"
    margin_meters: float
    offside_line_x: float
    attacking_direction: int  # +1: attacking toward +X (right), -1: toward -X (left)
    second_last_defender_id: Optional[int]
    second_last_defender_xy: Optional[Tuple[float, float]]
    deepest_defender_id: Optional[int]
    deepest_defender_xy: Optional[Tuple[float, float]]
    ball_xy: Tuple[float, float]
    attacker_statuses: Dict[int, PlayerOffsideStatus] = field(default_factory=dict)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "passer_id": int(self.passer_id),
            "passer_team": int(self.passer_team),
            "receiver_id": int(self.receiver_id) if self.receiver_id is not None else None,
            "is_offside": bool(self.is_offside),
            "decision": self.decision,
            "margin_meters": float(round(self.margin_meters, 3)),
            "offside_line_x": float(round(self.offside_line_x, 2)),
            "attacking_direction": int(self.attacking_direction),
            "second_last_defender_id": int(self.second_last_defender_id) if self.second_last_defender_id is not None else None,
            "second_last_defender_xy": [float(round(v, 2)) for v in self.second_last_defender_xy] if self.second_last_defender_xy else None,
            "deepest_defender_id": int(self.deepest_defender_id) if self.deepest_defender_id is not None else None,
            "deepest_defender_xy": [float(round(v, 2)) for v in self.deepest_defender_xy] if self.deepest_defender_xy else None,
            "ball_xy": [float(round(v, 2)) for v in self.ball_xy],
            "attacker_statuses": {
                str(pid): status.to_dict() for pid, status in self.attacker_statuses.items()
            },
            "summary_text": self.summary_text,
        }


class VAROffsideEvaluator:
    """
    FIFA Law 11 & SAOT Geometric Offside Line Evaluator.
    Pitch coordinate space:
      Length: 105.0m (X: [-52.5, +52.5])
      Width: 68.0m (Y: [-34.0, +34.0])
      Center spot: (0.0, 0.0)
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        tolerance_meters: float = 0.05,  # 5cm VAR calibration tolerance
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.half_len = pitch_length / 2.0
        self.half_wid = pitch_width / 2.0
        self.tolerance = tolerance_meters

    def _infer_attacking_direction(
        self,
        passer_team: int,
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int],
        ball_xy: Optional[Tuple[float, float]] = None,
    ) -> int:
        """
        Infers attacking direction (+1 for attacking right, -1 for left).
        Based on defending team's average outfield centroid or goalkeeper side.
        """
        defending_team = 2 if passer_team == 1 else 1
        def_xs = [
            pos[0]
            for pid, pos in player_positions.items()
            if player_teams.get(pid) == defending_team
        ]
        if not def_xs:
            return 1  # Default right

        mean_def_x = sum(def_xs) / len(def_xs)
        # If defenders are on average on positive X side, attack direction is +1 (toward positive X)
        return 1 if mean_def_x >= 0.0 else -1

    def evaluate_pass_release(
        self,
        frame_idx: int,
        passer_id: int,
        passer_team: int,
        ball_xy: Tuple[float, float],
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, int],
        receiver_id: Optional[int] = None,
        attacking_direction: Optional[int] = None,
    ) -> OffsideEvaluationResult:
        """
        Evaluates offside condition at kick-point frame t_release.
        """
        if attacking_direction is None:
            direction = self._infer_attacking_direction(passer_team, player_positions, player_teams, ball_xy)
        else:
            direction = 1 if attacking_direction >= 0 else -1

        defending_team = 2 if passer_team == 1 else 1

        # Collect defenders
        defenders = [
            (pid, pos)
            for pid, pos in player_positions.items()
            if player_teams.get(pid) == defending_team
        ]

        if len(defenders) < 2:
            # Degenerate case: fewer than 2 defenders on pitch
            # Under FIFA rules, offside line falls back to halfway line
            offside_line_x = 0.0
            second_last_id = None
            second_last_xy = None
            deepest_id = defenders[0][0] if defenders else None
            deepest_xy = defenders[0][1] if defenders else None
        else:
            # Sort defenders along attack axis (deepest towards defending goal)
            if direction > 0:
                # Goal is at +half_len; deepest has maximum x
                sorted_defs = sorted(defenders, key=lambda d: d[1][0], reverse=True)
            else:
                # Goal is at -half_len; deepest has minimum x
                sorted_defs = sorted(defenders, key=lambda d: d[1][0], reverse=False)

            deepest_id, deepest_xy = sorted_defs[0]
            second_last_id, second_last_xy = sorted_defs[1]

            # Offside line is drawn at second-last defender's x position
            raw_offside_x = second_last_xy[0]

            # Halfway line constraint: cannot be offside in own half
            if direction > 0:
                offside_line_x = max(0.0, raw_offside_x)
            else:
                offside_line_x = min(0.0, raw_offside_x)

        # Collect attacking teammates (excluding passer)
        attackers = [
            (pid, pos)
            for pid, pos in player_positions.items()
            if player_teams.get(pid) == passer_team and pid != passer_id
        ]

        attacker_statuses: Dict[int, PlayerOffsideStatus] = {}
        receiver_status: Optional[PlayerOffsideStatus] = None

        for att_id, att_xy in attackers:
            is_recv = (att_id == receiver_id)
            att_x = att_xy[0]
            ball_x = ball_xy[0]

            # 1. Halfway line check
            in_opponent_half = (att_x > 0.0) if direction > 0 else (att_x < 0.0)
            if not in_opponent_half:
                margin = (att_x - offside_line_x) if direction > 0 else (offside_line_x - att_x)
                status = PlayerOffsideStatus(
                    player_id=att_id,
                    team_id=passer_team,
                    xy=att_xy,
                    is_offside=False,
                    margin_meters=margin,
                    reason="OWN_HALF",
                    is_receiver=is_recv,
                )
                attacker_statuses[att_id] = status
                if is_recv:
                    receiver_status = status
                continue

            # 2. Behind or level with ball check
            ahead_of_ball = (att_x > ball_x) if direction > 0 else (att_x < ball_x)
            if not ahead_of_ball:
                margin = (att_x - offside_line_x) if direction > 0 else (offside_line_x - att_x)
                status = PlayerOffsideStatus(
                    player_id=att_id,
                    team_id=passer_team,
                    xy=att_xy,
                    is_offside=False,
                    margin_meters=margin,
                    reason="BEHIND_BALL",
                    is_receiver=is_recv,
                )
                attacker_statuses[att_id] = status
                if is_recv:
                    receiver_status = status
                continue

            # 3. Offside line comparison
            if direction > 0:
                margin = att_x - offside_line_x
            else:
                margin = offside_line_x - att_x

            if margin > self.tolerance:
                is_off = True
                reason = "OFFSIDE"
            else:
                is_off = False
                reason = "ONSIDE"

            status = PlayerOffsideStatus(
                player_id=att_id,
                team_id=passer_team,
                xy=att_xy,
                is_offside=is_off,
                margin_meters=margin,
                reason=reason,
                is_receiver=is_recv,
            )
            attacker_statuses[att_id] = status
            if is_recv:
                receiver_status = status

        # Final decision based on intended receiver if specified, else overall check
        if receiver_status is not None:
            is_offside = receiver_status.is_offside
            decision = receiver_status.reason
            final_margin = receiver_status.margin_meters
            if is_offside:
                summary = (
                    f"VAR OFFSIDE: Receiver #{receiver_id} was {final_margin:+.2f}m beyond the "
                    f"offside line (x={offside_line_x:.2f}m, second-last defender #{second_last_id}) at release."
                )
            else:
                summary = (
                    f"VAR ONSIDE: Receiver #{receiver_id} was legal ({receiver_status.reason}, "
                    f"margin {final_margin:+.2f}m relative to line x={offside_line_x:.2f}m)."
                )
        else:
            any_offside = any(s.is_offside for s in attacker_statuses.values())
            is_offside = any_offside
            decision = "POTENTIAL_OFFSIDE" if any_offside else "ONSIDE"
            final_margin = max((s.margin_meters for s in attacker_statuses.values()), default=0.0)
            summary = (
                f"VAR CHECK: No specific receiver targeted; "
                f"{sum(1 for s in attacker_statuses.values() if s.is_offside)} attacker(s) in offside positions."
            )

        return OffsideEvaluationResult(
            frame_idx=frame_idx,
            passer_id=passer_id,
            passer_team=passer_team,
            receiver_id=receiver_id,
            is_offside=is_offside,
            decision=decision,
            margin_meters=final_margin,
            offside_line_x=offside_line_x,
            attacking_direction=direction,
            second_last_defender_id=second_last_id,
            second_last_defender_xy=second_last_xy,
            deepest_defender_id=deepest_id,
            deepest_defender_xy=deepest_xy,
            ball_xy=ball_xy,
            attacker_statuses=attacker_statuses,
            summary_text=summary,
        )

    def render_var_freeze_frame(
        self,
        result: OffsideEvaluationResult,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Optional[np.ndarray]:
        """
        Renders a broadcast-quality 2D VAR Pitch Graphic with the calibrated offside line.
        """
        if plt is None:
            return None

        fig, ax = plt.subplots(figsize=figsize, facecolor="#1e293b")
        ax.set_facecolor("#15803d")  # Green pitch

        # Draw pitch boundaries
        hw, hl = self.half_wid, self.half_len
        ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color="white", lw=2)
        # Halfway line
        ax.plot([0, 0], [-hw, hw], color="white", lw=1.5, linestyle="--", alpha=0.7)
        # Center circle
        center_circle = patches.Circle((0, 0), 9.15, fill=False, color="white", lw=1.5, alpha=0.7)
        ax.add_patch(center_circle)

        # Penalty boxes (16.5m depth, 40.3m width)
        box_depth = 16.5
        box_hw = 20.15
        # Right box
        ax.plot([hl - box_depth, hl - box_depth, hl], [-box_hw, box_hw, box_hw], color="white", lw=1.5)
        ax.plot([hl - box_depth, hl], [-box_hw, -box_hw], color="white", lw=1.5)
        # Left box
        ax.plot([-hl + box_depth, -hl + box_depth, -hl], [-box_hw, box_hw, box_hw], color="white", lw=1.5)
        ax.plot([-hl + box_depth, -hl], [-box_hw, -box_hw], color="white", lw=1.5)

        # Offside Line
        off_x = result.offside_line_x
        line_color = "#ef4444" if result.is_offside else "#22c55e"
        ax.plot([off_x, off_x], [-hw, hw], color=line_color, lw=2.5, linestyle="-", label=f"Offside Line ({off_x:.1f}m)")

        # Ball
        bx, by = result.ball_xy
        ax.scatter([bx], [by], color="#fbbf24", s=180, edgecolors="black", lw=1.5, zorder=6, label="Ball (Kick-point)")

        # Deepest defender
        if result.deepest_defender_xy:
            dx, dy = result.deepest_defender_xy
            ax.scatter([dx], [dy], color="#64748b", s=140, edgecolors="white", lw=1.5, zorder=5)
            ax.text(dx, dy + 1.8, f"Def #{result.deepest_defender_id} (GK)", color="#f8fafc", fontsize=9, ha="center")

        # Second-last defender
        if result.second_last_defender_xy:
            sx, sy = result.second_last_defender_xy
            ax.scatter([sx], [sy], color="#38bdf8", s=180, edgecolors="white", lw=2, zorder=5, label=f"2nd-Last Def (#{result.second_last_defender_id})")
            ax.text(sx, sy + 1.8, f"2nd Last #{result.second_last_defender_id}", color="#38bdf8", fontsize=9, fontweight="bold", ha="center")

        # Attackers
        for pid, status in result.attacker_statuses.items():
            px, py = status.xy
            if status.is_receiver:
                c = "#dc2626" if status.is_offside else "#16a34a"
                label = f"Receiver #{pid} ({'OFFSIDE' if status.is_offside else 'ONSIDE'})"
                ax.scatter([px], [py], color=c, s=240, edgecolors="#fef08a", lw=2.5, zorder=7, label=label)
                ax.text(px, py - 2.5, f"#{pid} ({status.margin_meters:+.2f}m)", color=c, fontsize=10, fontweight="bold", ha="center")
            else:
                c = "#f87171" if status.is_offside else "#86efac"
                ax.scatter([px], [py], color=c, s=110, edgecolors="black", lw=1, zorder=4)
                ax.text(px, py + 1.5, f"#{pid}", color="white", fontsize=8, ha="center")

        # Direction arrow
        arrow_dir = result.attacking_direction
        arrow_x = -20 if arrow_dir > 0 else 20
        ax.annotate(
            "Attacking Direction",
            xy=(arrow_x + 15 * arrow_dir, hw - 3),
            xytext=(arrow_x, hw - 3),
            arrowprops=dict(facecolor="#facc15", edgecolor="none", width=2, headwidth=8),
            color="#facc15",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

        ax.set_xlim(-hl - 3, hl + 3)
        ax.set_ylim(-hw - 3, hw + 3)
        ax.set_aspect("equal")
        ax.set_title(
            f"VAR OFFSIDE ANALYSIS - Frame {result.frame_idx} | Decision: {result.decision} ({result.margin_meters:+.2f}m)",
            color="white",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=8)
        ax.axis("off")

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
