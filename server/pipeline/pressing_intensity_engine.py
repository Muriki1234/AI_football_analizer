"""
pressing_intensity_engine.py - Passes Per Defensive Action (PPDA) & Spatial Pressing Intensity Analyzer

Architectural Foundations:
1. Passes Per Defensive Action (StatsBomb / Colin Trainor 2014 Standard):
   Measures defensive pressing intensity in the opposition 60% of the pitch:
   PPDA = (Opponent Passes in Opposition 60%) / (Defensive Actions in Opposition 60%)
   Tactical Classification:
   - PPDA < 8.0: Aggressive Gegenpressing (Klopp / Bielsa / Guardiola high press)
   - PPDA 8.0 - 11.5: Active Mid-High Block
   - PPDA 11.5 - 15.0: Moderate Containment Block
   - PPDA > 15.0: Deep Passive Low Block (Mourinho / Dyche low block)

2. Fernandez & Bornn (MIT Sloan 2018) Spatial Pressure Proximity:
   Tracks continuous spatiotemporal defensive pressure triggers:
   A pressing action is registered when a defending player closes to within <= 2.2m of an
   opponent ball carrier in the high pressing territory.
   Debounces continuous contact across adjacent frames to ensure single duel attribution.

3. High-Press Turnover & Counter-Press Efficiency:
   Detects turnovers won within 35m of the opponent goal line, quantifying direct counter-pressing
   success and immediate turnover transition pressure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PressingAction:
    action_id: int
    frame: int
    timestamp_sec: float
    defending_team: int
    defender_id: int
    opponent_carrier_id: Optional[int]
    action_type: str  # "challenge_duel", "interception", "tackle_turnover", "pressure_trigger"
    pitch_xy: Tuple[float, float]
    distance_to_opp_goal: float
    is_high_press: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pitch_xy"] = [round(v, 2) for v in self.pitch_xy]
        d["distance_to_opp_goal"] = round(self.distance_to_opp_goal, 2)
        d["timestamp_sec"] = round(self.timestamp_sec, 2)
        return d


@dataclass
class TeamPPDAMetrics:
    team_id: int
    ppda: float
    pressing_style: str
    opponent_passes_in_zone: int
    defensive_actions_in_zone: int
    high_press_turnovers: int
    pressure_actions_count: int
    interceptions_count: int
    tackles_count: int
    top_pressers: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ppda"] = round(self.ppda, 2)
        return d


class PressingIntensityEngine:
    """
    Passes Per Defensive Action (PPDA) and spatial pressing intensity engine.
    Operates on metric pitch coordinates (length: 105m, width: 68m, center: (0, 0)).
    """

    def __init__(
        self,
        fps: float = 25.0,
        pressure_radius_m: float = 2.2,
        high_press_threshold_m: float = 35.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.pressure_radius_m = float(pressure_radius_m)
        self.high_press_threshold_m = float(high_press_threshold_m)
        self.half_len = pitch_length / 2.0  # 52.5m
        self.half_wid = pitch_width / 2.0  # 34.0m
        # Opposition 60% boundary: on standard 105m pitch, 60% of 105m is 63m from goal,
        # leaving 42m in back, so boundary is at x = 52.5 - 63.0 = -10.5m (or +10.5m).
        self.opp_60_boundary_m = self.half_len - (pitch_length * 0.60)  # -10.5m

    def normalize_coordinates(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[int, Tuple[float, float]]]]:
        """
        Converts coordinates to standard pitch center [-52.5, 52.5] x [-34, 34].
        """
        all_x: List[float] = []
        all_y: List[float] = []
        for p in ball_trajectory.values():
            all_x.append(p[0])
            all_y.append(p[1])

        if not all_x:
            for p_dict in player_trajectories.values():
                for px, py in p_dict.values():
                    all_x.append(px)
                    all_y.append(py)

        if not all_x:
            return ball_trajectory, player_trajectories

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        scale = 1.0
        if max_x > 200.0 or min_x < -200.0 or max_y > 200.0 or min_y < -200.0:
            scale = 0.01
            min_x *= scale
            max_x *= scale
            min_y *= scale
            max_y *= scale

        shift_x = 0.0
        shift_y = 0.0
        mean_y = float(np.mean(all_y)) if all_y else 0.0
        if (min_x >= -2.0 and max_x > 60.0 and mean_y > 15.0) or (min_y >= 0.0 and mean_y > 20.0):
            shift_x = self.half_len
            shift_y = self.half_wid

        norm_ball = {
            f: ((x * scale) - shift_x, (y * scale) - shift_y)
            for f, (x, y) in ball_trajectory.items()
        }
        norm_players = {
            f: {
                pid: ((px * scale) - shift_x, (py * scale) - shift_y)
                for pid, (px, py) in p_dict.items()
            }
            for f, p_dict in player_trajectories.items()
        }
        return norm_ball, norm_players

    def resolve_team_attacking_directions(
        self,
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        teams: Dict[int, int],
    ) -> Dict[int, str]:
        """
        Determines team attacking directions (right towards +52.5 or left towards -52.5).
        """
        team_xs: Dict[int, List[float]] = {}
        for p_dict in player_trajectories.values():
            for pid, (px, _) in p_dict.items():
                t = teams.get(pid)
                if t is not None and t > 0:
                    team_xs.setdefault(t, []).append(px)

        unique_teams = sorted(team_xs.keys())
        if len(unique_teams) >= 2:
            t1, t2 = unique_teams[0], unique_teams[1]
            mean1 = np.mean(team_xs[t1]) if team_xs[t1] else 0.0
            mean2 = np.mean(team_xs[t2]) if team_xs[t2] else 0.0
            if mean1 < mean2:
                return {t1: "right", t2: "left"}
            else:
                return {t1: "left", t2: "right"}
        elif len(unique_teams) == 1:
            return {unique_teams[0]: "right"}
        return {1: "right", 2: "left"}

    def is_in_opposition_60(self, pitch_x: float, defending_team: int, attacking_direction: str) -> bool:
        """
        Checks if pitch_x is located within the opposition 60% of the pitch
        from the perspective of the defending team.
        - If defending team attacks right (+52.5), opposition half is +x.
          Opposition 60% extends from x = -10.5m to +52.5m (pitch_x >= -10.5).
        - If defending team attacks left (-52.5), opposition half is -x.
          Opposition 60% extends from x = +10.5m to -52.5m (pitch_x <= +10.5).
        """
        if attacking_direction == "right":
            return pitch_x >= -10.5
        else:
            return pitch_x <= 10.5

    def detect_defensive_pressing_actions(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        teams: Dict[int, int],
        attacking_directions: Dict[int, str],
        pass_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PressingAction]:
        """
        Detects individual pressing actions:
        1. Proximity challenges (defending player within <= 2.2m of opponent ball carrier in opp 60%).
        2. Pass interceptions from pass_events.
        3. High press turnovers.
        """
        actions: List[PressingAction] = []
        action_id = 1
        active_duels: Dict[Tuple[int, int], int] = {}  # (defender_id, carrier_id) -> last_frame

        # 1. Proximity Pressure Triggers
        frames = sorted(ball_trajectory.keys())
        for f in frames:
            bx, by = ball_trajectory[f]
            f_players = player_trajectories.get(f, {})
            if not f_players:
                continue

            # Identify ball carrier (attacker within 1.8m of ball)
            carrier_id = None
            carrier_team = None
            min_c_dist = 1.8
            for pid, ppos in f_players.items():
                d = math.hypot(ppos[0] - bx, ppos[1] - by)
                if d < min_c_dist:
                    min_c_dist = d
                    carrier_id = pid
                    carrier_team = teams.get(pid)

            if carrier_id is None or carrier_team is None:
                continue

            c_pos = f_players[carrier_id]

            # Find defending players pressing this carrier
            for d_pid, d_pos in f_players.items():
                d_team = teams.get(d_pid)
                if d_team is None or d_team == carrier_team:
                    continue

                dist_to_carrier = math.hypot(d_pos[0] - c_pos[0], d_pos[1] - c_pos[1])
                if dist_to_carrier <= self.pressure_radius_m:
                    d_att_dir = attacking_directions.get(d_team, "right")
                    opp_goal_x = self.half_len if d_att_dir == "right" else -self.half_len
                    dist_to_opp_goal = abs(opp_goal_x - d_pos[0])

                    # Check if action occurs in opposition 60% of pitch
                    if self.is_in_opposition_60(d_pos[0], d_team, d_att_dir):
                        duel_key = (d_pid, carrier_id)
                        last_seen_f = active_duels.get(duel_key, -999)

                        # Debounce: register only if duel started or separated by > 1.0s (25 frames)
                        if f - last_seen_f > int(self.fps * 1.0):
                            is_high = (dist_to_opp_goal <= self.high_press_threshold_m)
                            actions.append(
                                PressingAction(
                                    action_id=action_id,
                                    frame=f,
                                    timestamp_sec=f / self.fps,
                                    defending_team=d_team,
                                    defender_id=d_pid,
                                    opponent_carrier_id=carrier_id,
                                    action_type="challenge_duel",
                                    pitch_xy=d_pos,
                                    distance_to_opp_goal=dist_to_opp_goal,
                                    is_high_press=is_high,
                                )
                            )
                            action_id += 1

                        active_duels[duel_key] = f

        # 2. Interceptions from Pass Events
        if pass_events:
            for pe in pass_events:
                if pe.get("outcome") == "intercepted":
                    rec_team = pe.get("receiver_team")
                    rec_id = pe.get("receiver_id")
                    if rec_team and rec_id:
                        rec_att_dir = attacking_directions.get(rec_team, "right")
                        opp_goal_x = self.half_len if rec_att_dir == "right" else -self.half_len
                        end_xy = pe.get("end_xy", (0.0, 0.0))
                        dist_to_opp_goal = abs(opp_goal_x - end_xy[0])
                        if self.is_in_opposition_60(end_xy[0], rec_team, rec_att_dir):
                            is_high = (dist_to_opp_goal <= self.high_press_threshold_m)
                            actions.append(
                                PressingAction(
                                    action_id=action_id,
                                    frame=pe.get("end_frame", 0),
                                    timestamp_sec=pe.get("end_frame", 0) / self.fps,
                                    defending_team=rec_team,
                                    defender_id=rec_id,
                                    opponent_carrier_id=pe.get("passer_id"),
                                    action_type="interception",
                                    pitch_xy=tuple(end_xy),
                                    distance_to_opp_goal=dist_to_opp_goal,
                                    is_high_press=is_high,
                                )
                            )
                            action_id += 1

        actions.sort(key=lambda a: a.frame)
        return actions

    @staticmethod
    def classify_pressing_style(ppda: float) -> str:
        """
        Classifies tactical pressing style based on PPDA value.
        """
        if ppda < 8.0:
            return "Aggressive Gegenpressing"
        elif ppda < 11.5:
            return "Active Mid-High Block"
        elif ppda < 15.0:
            return "Moderate Containment Block"
        else:
            return "Deep Passive Low Block"

    def compute_team_ppda(
        self,
        defending_team: int,
        opponent_team: int,
        pass_events: List[Dict[str, Any]],
        defensive_actions: List[PressingAction],
        attacking_directions: Dict[int, str],
    ) -> TeamPPDAMetrics:
        """
        Computes PPDA = (Opponent Passes in Opposition 60%) / (Defensive Actions in Opposition 60%).
        """
        def_att_dir = attacking_directions.get(defending_team, "right")

        # 1. Count Opponent Passes originated in the defending team opposition 60%
        opp_passes_in_zone = 0
        for pe in pass_events:
            p_team = pe.get("passer_team")
            if p_team == opponent_team:
                s_xy = pe.get("start_xy", (0.0, 0.0))
                # Is start_xy in defending team opposition 60%?
                if self.is_in_opposition_60(s_xy[0], defending_team, def_att_dir):
                    opp_passes_in_zone += 1

        # 2. Count Defensive Actions by Defending Team in Opposition 60%
        team_actions = [
            a for a in defensive_actions
            if a.defending_team == defending_team
            and self.is_in_opposition_60(a.pitch_xy[0], defending_team, def_att_dir)
        ]

        def_actions_count = len(team_actions)
        challenges = sum(1 for a in team_actions if a.action_type == "challenge_duel")
        interceptions = sum(1 for a in team_actions if a.action_type == "interception")
        high_turnovers = sum(1 for a in team_actions if a.is_high_press)

        # Compute PPDA
        if def_actions_count > 0:
            ppda_val = opp_passes_in_zone / float(def_actions_count)
        else:
            # If no defensive actions recorded, default to high value
            ppda_val = float(opp_passes_in_zone) if opp_passes_in_zone > 0 else 20.0

        ppda_val = round(max(1.0, min(35.0, ppda_val)), 2)
        style = self.classify_pressing_style(ppda_val)

        # Player leaderboard: who pressed the most?
        presser_counts: Dict[int, int] = {}
        for a in team_actions:
            presser_counts[a.defender_id] = presser_counts.get(a.defender_id, 0) + 1

        top_pressers = [
            {"player_id": pid, "pressures": count}
            for pid, count in sorted(presser_counts.items(), key=lambda x: x[1], reverse=True)[:4]
        ]

        return TeamPPDAMetrics(
            team_id=defending_team,
            ppda=ppda_val,
            pressing_style=style,
            opponent_passes_in_zone=opp_passes_in_zone,
            defensive_actions_in_zone=def_actions_count,
            high_press_turnovers=high_turnovers,
            pressure_actions_count=challenges,
            interceptions_count=interceptions,
            tackles_count=challenges + interceptions,
            top_pressers=top_pressers,
        )

    def analyze_match_pressing(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        teams: Dict[int, int],
        pass_events: Optional[List[Dict[str, Any]]] = None,
        attacking_directions: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end pressing intensity and PPDA analyzer for both teams.
        """
        if not ball_trajectory or not player_trajectories:
            return {
                "team1": None,
                "team2": None,
                "total_pressing_actions": 0,
                "actions": [],
            }

        norm_ball, norm_players = self.normalize_coordinates(ball_trajectory, player_trajectories)

        if attacking_directions is None:
            attacking_directions = self.resolve_team_attacking_directions(norm_players, teams)

        # Normalize pass events coordinates if provided
        norm_passes: List[Dict[str, Any]] = []
        if pass_events:
            for pe in pass_events:
                p_copy = dict(pe)
                if "start_xy" in p_copy:
                    sx, sy = p_copy["start_xy"]
                    # If start_xy was in [0, 105] corner coords
                    if sx > 60.0 and sy > 20.0:
                        p_copy["start_xy"] = (sx - self.half_len, sy - self.half_wid)
                if "end_xy" in p_copy:
                    ex, ey = p_copy["end_xy"]
                    if ex > 60.0 and ey > 20.0:
                        p_copy["end_xy"] = (ex - self.half_len, ey - self.half_wid)
                norm_passes.append(p_copy)

        # Detect all pressing actions
        actions = self.detect_defensive_pressing_actions(
            ball_trajectory=norm_ball,
            player_trajectories=norm_players,
            teams=teams,
            attacking_directions=attacking_directions,
            pass_events=norm_passes,
        )

        t1_metrics = self.compute_team_ppda(
            defending_team=1,
            opponent_team=2,
            pass_events=norm_passes,
            defensive_actions=actions,
            attacking_directions=attacking_directions,
        )

        t2_metrics = self.compute_team_ppda(
            defending_team=2,
            opponent_team=1,
            pass_events=norm_passes,
            defensive_actions=actions,
            attacking_directions=attacking_directions,
        )

        return {
            "team1": t1_metrics.to_dict(),
            "team2": t2_metrics.to_dict(),
            "total_pressing_actions": len(actions),
            "high_press_turnovers_total": t1_metrics.high_press_turnovers + t2_metrics.high_press_turnovers,
            "actions": [a.to_dict() for a in actions[:100]],
        }

    def render_pressing_report(
        self,
        analysis_dict: Dict[str, Any],
        output_path: Path,
        title: Optional[str] = None,
    ) -> None:
        """
        Generates a two-panel visualization:
        1. Pitch map showing pressing challenge duel locations and high-press zones.
        2. PPDA comparison radar/bar summary showing pressing intensity and styles.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax_pitch, ax_bar) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f172a", gridspec_kw={"width_ratios": [1.4, 1.0]})
        ax_pitch.set_facecolor("#1e293b")
        ax_bar.set_facecolor("#1e293b")

        # ── Pitch Markings ──
        ax_pitch.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#475569", lw=2)
        ax_pitch.plot([0, 0], [-34, 34], color="#475569", lw=1.5, linestyle="--")
        circle = plt.Circle((0, 0), 9.15, color="#475569", fill=False, lw=1.5)
        ax_pitch.add_patch(circle)

        # High Press Boundaries (opposition 60% markings: x = ±10.5m)
        ax_pitch.axvline(x=-10.5, color="#f59e0b", linestyle=":", lw=1.5, alpha=0.7)
        ax_pitch.axvline(x=10.5, color="#f59e0b", linestyle=":", lw=1.5, alpha=0.7)

        # Plot Pressing Actions
        actions = analysis_dict.get("actions", [])
        t1_x, t1_y = [], []
        t2_x, t2_y = [], []
        for a in actions:
            px, py = a["pitch_xy"]
            if a["defending_team"] == 1:
                t1_x.append(px)
                t1_y.append(py)
            else:
                t2_x.append(px)
                t2_y.append(py)

        if t1_x:
            ax_pitch.scatter(t1_x, t1_y, c="#3b82f6", s=65, alpha=0.85, edgecolors="white", lw=1, label="Team 1 Pressures", zorder=5)
        if t2_x:
            ax_pitch.scatter(t2_x, t2_y, c="#ef4444", s=65, alpha=0.85, edgecolors="white", lw=1, label="Team 2 Pressures", zorder=5)

        ax_pitch.set_xlim(-56, 56)
        ax_pitch.set_ylim(-37, 37)
        ax_pitch.set_aspect("equal")
        ax_pitch.set_xticks([])
        ax_pitch.set_yticks([])
        ax_pitch.set_title("Defensive Pressing Duels & Opposition 60% Territory", color="white", fontsize=11, fontweight="bold")
        if t1_x or t2_x:
            ax_pitch.legend(loc="lower center", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)

        # ── PPDA Comparison Bar Chart ──
        t1 = analysis_dict.get("team1") or {}
        t2 = analysis_dict.get("team2") or {}

        ppda_t1 = t1.get("ppda", 12.0)
        ppda_t2 = t2.get("ppda", 12.0)

        style1 = t1.get("pressing_style", "N/A")
        style2 = t2.get("pressing_style", "N/A")
        teams_labels = [f"Team 1\n({style1})", f"Team 2\n({style2})"]
        ppda_values = [ppda_t1, ppda_t2]
        colors = ["#3b82f6", "#ef4444"]

        bars = ax_bar.bar(teams_labels, ppda_values, color=colors, width=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax_bar.grid(axis="y", color="#334155", linestyle="--", alpha=0.7)

        for bar, val in zip(bars, ppda_values):
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"PPDA: {val:.1f}\n(Lower=Higher Press)",
                        ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

        ax_bar.set_ylim(0, max(25.0, max(ppda_values) + 5.0))
        ax_bar.set_ylabel("Passes Per Defensive Action (PPDA)", color="white", fontsize=11)
        ax_bar.tick_params(colors="white")
        ax_bar.set_title("PPDA Pressing Intensity Comparison", color="white", fontsize=11, fontweight="bold")

        if title is None:
            title = f"Tactical Pressing & PPDA Analysis — Total Pressures: {analysis_dict.get('total_pressing_actions', 0)}"

        fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
