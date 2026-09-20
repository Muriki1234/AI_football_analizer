"""
turnover_transition_spotter.py - Defensive Turnover & Immediate Counter-Press Transition Spotter

Architectural Foundations:
1. Garganta (1997) 5-Second Post-Turnover Transition Window:
   The 5 seconds immediately following a change of possession represent the critical window
   determining whether the losing team successfully counter-presses (Gegenpressing) or gets
   exposed by a fast-break counter-attack.

2. Counter-Press Reaction Latency (tau_counterpress):
   Measures the precise elapsed time (seconds) between losing possession and the first defending
   player challenging within <= 2.8m of the new ball carrier.
   - < 1.8s: Elite Gegenpressing reaction (immediate pressure)
   - 1.8s - 3.0s: Standard organized retreat & containment
   - > 3.0s / No challenge: Passive recovery / broken defensive transition

3. Spatial Hazard & Fast-Break Classification:
   - High Turnover Won: Won in opponent final third (<= 35m to opponent goal line)
   - Dangerous Turnover Lost: Lost in own defensive third or central Zone 14
   - Fast-Break Counter-Attack: Forward ball progression >= 18m towards opponent goal within 5s
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
class TurnoverEvent:
    turnover_id: int
    frame: int
    timestamp_sec: float
    timestamp_mmss: str
    losing_team: int
    losing_player_id: Optional[int]
    winning_team: int
    winning_player_id: Optional[int]
    turnover_xy: Tuple[float, float]
    zone: str  # "attacking_third", "middle_third", "defensive_third", "zone_14"
    is_high_turnover: bool
    is_dangerous_turnover: bool
    counter_press_reacted: bool
    counter_press_reaction_sec: Optional[float]
    is_counter_attack: bool
    progression_distance_5s: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["turnover_xy"] = [round(v, 2) for v in self.turnover_xy]
        d["timestamp_sec"] = round(self.timestamp_sec, 2)
        if self.counter_press_reaction_sec is not None:
            d["counter_press_reaction_sec"] = round(self.counter_press_reaction_sec, 2)
        d["progression_distance_5s"] = round(self.progression_distance_5s, 2)
        return d


class TurnoverTransitionSpotter:
    """
    Spots possession turnovers and evaluates immediate counter-pressing
    reaction latency and fast-break counter-attacks.
    Operates on metric pitch coordinates (length: 105m, width: 68m, center: (0, 0)).
    """

    def __init__(
        self,
        fps: float = 25.0,
        control_radius_m: float = 2.0,
        min_possession_frames: int = 3,
        reaction_window_sec: float = 5.0,
        counter_press_challenge_radius_m: float = 2.8,
        fast_break_progression_m: float = 18.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.control_radius_m = float(control_radius_m)
        self.min_possession_frames = int(min_possession_frames)
        self.reaction_window_sec = float(reaction_window_sec)
        self.counter_press_challenge_radius_m = float(counter_press_challenge_radius_m)
        self.fast_break_progression_m = float(fast_break_progression_m)
        self.half_len = pitch_length / 2.0  # 52.5m
        self.half_wid = pitch_width / 2.0  # 34.0m

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

    def classify_pitch_zone(
        self,
        pitch_xy: Tuple[float, float],
        team: int,
        attacking_direction: str,
    ) -> str:
        """
        Classifies turnover spatial zone:
        - "attacking_third"
        - "defensive_third"
        - "zone_14" (central attacking region immediately outside penalty box)
        - "middle_third"
        """
        x, y = pitch_xy
        # Zone 14 check:
        # If attacking right: x in [17.5, 36.0] and abs(y) <= 13.5
        # If attacking left: x in [-36.0, -17.5] and abs(y) <= 13.5
        if attacking_direction == "right":
            if 17.5 <= x <= 36.0 and abs(y) <= 13.5:
                return "zone_14"
            elif x >= 17.5:
                return "attacking_third"
            elif x <= -17.5:
                return "defensive_third"
            else:
                return "middle_third"
        else:
            if -36.0 <= x <= -17.5 and abs(y) <= 13.5:
                return "zone_14"
            elif x <= -17.5:
                return "attacking_third"
            elif x >= 17.5:
                return "defensive_third"
            else:
                return "middle_third"

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def spot_turnovers(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        teams: Dict[int, int],
        attacking_directions: Optional[Dict[int, str]] = None,
    ) -> List[TurnoverEvent]:
        """
        Spots turnover moments and traces subsequent 5-second transition.
        """
        if not ball_trajectory or not player_trajectories:
            return []

        norm_ball, norm_players = self.normalize_coordinates(ball_trajectory, player_trajectories)

        if attacking_directions is None:
            attacking_directions = self.resolve_team_attacking_directions(norm_players, teams)

        frames = sorted(norm_ball.keys())
        if len(frames) < self.min_possession_frames * 2:
            return []

        # Step 1: Assign possession per frame (player with ball within control radius)
        frame_controllers: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        for f in frames:
            bx, by = norm_ball[f]
            f_players = norm_players.get(f, {})
            c_pid = None
            c_team = None
            min_dist = self.control_radius_m
            for pid, (px, py) in f_players.items():
                d = math.hypot(px - bx, py - by)
                if d < min_dist:
                    min_dist = d
                    c_pid = pid
                    c_team = teams.get(pid)
            frame_controllers[f] = (c_pid, c_team)

        turnovers: List[TurnoverEvent] = []
        turnover_counter = 1

        active_team: Optional[int] = None
        active_pid: Optional[int] = None
        consecutive_frames = 0
        last_turnover_frame = -999

        reaction_window_frames = int(self.reaction_window_sec * self.fps)

        for idx, f in enumerate(frames):
            pid, team = frame_controllers[f]

            if team is not None and team > 0:
                if team == active_team:
                    consecutive_frames += 1
                    active_pid = pid
                else:
                    # Potential turnover if previous team held possession for >= min_possession_frames
                    if active_team is not None and consecutive_frames >= self.min_possession_frames:
                        # Debounce turnovers within 2.0s
                        if f - last_turnover_frame > int(self.fps * 2.0):
                            losing_team = active_team
                            losing_pid = active_pid
                            winning_team = team
                            winning_pid = pid
                            t_xy = norm_ball[f]

                            win_att_dir = attacking_directions.get(winning_team, "right")
                            lose_att_dir = attacking_directions.get(losing_team, "right")

                            # Zone relative to winning team
                            zone = self.classify_pitch_zone(t_xy, winning_team, win_att_dir)
                            is_high = (zone == "attacking_third")
                            # Dangerous for losing team: defensive third or zone 14
                            lose_zone = self.classify_pitch_zone(t_xy, losing_team, lose_att_dir)
                            is_dangerous = (lose_zone in ("defensive_third", "zone_14"))

                            # Trace immediate 5-second post-turnover reaction
                            max_check_f = f + reaction_window_frames
                            challenge_frame = None
                            start_bx = t_xy[0]
                            max_progression = 0.0

                            win_target_goal_x = self.half_len if win_att_dir == "right" else -self.half_len

                            for f_ahead in range(f, min(frames[-1] + 1, max_check_f)):
                                if f_ahead not in norm_players:
                                    continue
                                ahead_players = norm_players[f_ahead]

                                # Current carrier (or winning team player closest to ball)
                                ahead_ball = norm_ball.get(f_ahead, t_xy)
                                cur_carrier_pos = None
                                if winning_pid in ahead_players:
                                    cur_carrier_pos = ahead_players[winning_pid]
                                else:
                                    cur_carrier_pos = ahead_ball

                                # Measure ball progression towards opponent goal
                                if win_att_dir == "right":
                                    prog = ahead_ball[0] - start_bx
                                else:
                                    prog = start_bx - ahead_ball[0]
                                if prog > max_progression:
                                    max_progression = prog

                                # Check if losing team player challenges within challenge radius
                                if challenge_frame is None and cur_carrier_pos:
                                    for opp_id, opp_pos in ahead_players.items():
                                        if teams.get(opp_id) == losing_team:
                                            d_chal = math.hypot(opp_pos[0] - cur_carrier_pos[0], opp_pos[1] - cur_carrier_pos[1])
                                            if d_chal <= self.counter_press_challenge_radius_m:
                                                challenge_frame = f_ahead
                                                break

                            counter_press_reacted = (challenge_frame is not None)
                            reaction_sec = (
                                (challenge_frame - f) / self.fps
                                if challenge_frame is not None
                                else None
                            )

                            is_counter = (max_progression >= self.fast_break_progression_m)

                            sec = f / self.fps
                            event = TurnoverEvent(
                                turnover_id=turnover_counter,
                                frame=f,
                                timestamp_sec=sec,
                                timestamp_mmss=self._format_mmss(sec),
                                losing_team=losing_team,
                                losing_player_id=losing_pid,
                                winning_team=winning_team,
                                winning_player_id=winning_pid,
                                turnover_xy=t_xy,
                                zone=zone,
                                is_high_turnover=is_high,
                                is_dangerous_turnover=is_dangerous,
                                counter_press_reacted=counter_press_reacted,
                                counter_press_reaction_sec=reaction_sec,
                                is_counter_attack=is_counter,
                                progression_distance_5s=max(0.0, max_progression),
                            )
                            turnovers.append(event)
                            turnover_counter += 1
                            last_turnover_frame = f

                    active_team = team
                    active_pid = pid
                    consecutive_frames = 1
            else:
                consecutive_frames = 0

        return turnovers

    def summarize_transitions(self, turnovers: List[TurnoverEvent]) -> Dict[str, Any]:
        """
        Aggregates match transition intelligence, reaction latencies, and counter-attacks.
        """
        tot = len(turnovers)
        if tot == 0:
            return {
                "total_turnovers": 0,
                "team1": {"turnovers_won": 0, "turnovers_lost": 0, "high_turnovers_won": 0, "dangerous_turnovers_lost": 0, "counter_attacks": 0, "avg_reaction_sec": None},
                "team2": {"turnovers_won": 0, "turnovers_lost": 0, "high_turnovers_won": 0, "dangerous_turnovers_lost": 0, "counter_attacks": 0, "avg_reaction_sec": None},
                "events": [],
            }

        t1_won = [t for t in turnovers if t.winning_team == 1]
        t2_won = [t for t in turnovers if t.winning_team == 2]

        t1_lost = [t for t in turnovers if t.losing_team == 1]
        t2_lost = [t for t in turnovers if t.losing_team == 2]

        def _stats(team_won: List[TurnoverEvent], team_lost: List[TurnoverEvent]) -> Dict[str, Any]:
            reactions = [t.counter_press_reaction_sec for t in team_lost if t.counter_press_reaction_sec is not None]
            avg_r = round(float(np.mean(reactions)), 2) if reactions else None
            high_won = sum(1 for t in team_won if t.is_high_turnover)
            dang_lost = sum(1 for t in team_lost if t.is_dangerous_turnover)
            counter_atts = sum(1 for t in team_won if t.is_counter_attack)
            return {
                "turnovers_won": len(team_won),
                "turnovers_lost": len(team_lost),
                "high_turnovers_won": high_won,
                "dangerous_turnovers_lost": dang_lost,
                "counter_attacks": counter_atts,
                "avg_reaction_sec": avg_r,
                "counter_press_reactions_count": len(reactions),
            }

        return {
            "total_turnovers": tot,
            "team1": _stats(t1_won, t1_lost),
            "team2": _stats(t2_won, t2_lost),
            "events": [t.to_dict() for t in turnovers],
        }

    # Alias for task runner consistency
    generate_transition_summary = summarize_transitions

    def render_turnover_map(
        self,
        turnovers: List[TurnoverEvent],
        output_path: Path,
        title: Optional[str] = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")

        # Pitch outline
        ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#475569", lw=2)
        ax.plot([0, 0], [-34, 34], color="#475569", lw=1.5, linestyle="--")
        circle = plt.Circle((0, 0), 9.15, color="#475569", fill=False, lw=1.5)
        ax.add_patch(circle)

        # Plot turnovers
        plotted_labels = set()
        for t in turnovers:
            tx, ty = t.turnover_xy
            if t.winning_team == 1:
                color = "#3b82f6"
                lbl = "Team 1 Turnover Won"
            else:
                color = "#ef4444"
                lbl = "Team 2 Turnover Won"

            label = lbl if lbl not in plotted_labels else ""
            if label:
                plotted_labels.add(lbl)

            # High turnover highlight
            marker = "o"
            size = 80
            if t.is_high_turnover:
                marker = "*"
                size = 180
            elif t.is_dangerous_turnover:
                marker = "X"
                size = 130

            ax.scatter(tx, ty, c=color, s=size, marker=marker, edgecolors="white", lw=1.5, zorder=6, label=label)

            # Annotation
            txt = f"{t.timestamp_mmss}"
            if t.is_counter_attack:
                txt += " (Counter!)"
            ax.annotate(txt, xy=(tx, ty), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", color="white", fontsize=8, fontweight="bold", zorder=7)

        ax.set_xlim(-56, 56)
        ax.set_ylim(-37, 37)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        if plotted_labels:
            ax.legend(loc="lower center", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=9)

        if title is None:
            title = f"Defensive Turnover & Transition Map — Total Turnovers: {len(turnovers)}"
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
