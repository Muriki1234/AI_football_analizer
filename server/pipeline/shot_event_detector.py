"""
shot_event_detector.py — Tactical Shot Event Spotting & Freeze-Frame Expected Goals (xG) Physics Evaluator

Architectural Foundations:
1. Kinematic Ball-Action Shot Spotting (SoccerNet CVPR Standard):
   Identifies high-velocity goal-directed trajectories (> 9.5 m/s or > 34 km/h) originating in the
   attacking final third and aimed within the goal mouth angular corridor.
   Attributes the shooter to the nearest attacking player at the release moment.
   Classifies outcomes: "goal", "saved_by_keeper", "blocked_by_defender", "off_target".

2. StatsBomb 360 & Lucey et al. Freeze-Frame Expected Goals (xG) Physics Model:
   At the exact freeze-frame moment of the shot release, calculates:
   - Distance to goal center d (exponential decay)
   - Visible goal subtended angle θ (in radians) subtended by the goal posts (7.32m width)
   - Defensive shot cone density (number of opposing outfield defenders inside the shooter-goal triangle)
   - Direct line-of-sight blocker count (defenders within ±1.2m of direct shot ray)
   - Goalkeeper positioning factor (distance off goal line, central bisector alignment)
   - Calibrated logistic sigmoid probability calibrated to empirical StatsBomb/Opta distributions:
     * Penalty kick (11m central, unblocked): ~0.76 xG
     * Central 6-yard box tap-in: ~0.70-0.85 xG
     * Edge of 18-yard box central (18m out, contested): ~0.08-0.14 xG
     * Tight byline angle / 30m screamer: ~0.02-0.04 xG
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
class ShotEvent:
    shot_id: int
    shooter_id: int
    shooter_team: int
    start_frame: int
    end_frame: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    target_goal: str  # "right" (+52.5) or "left" (-52.5)
    distance_m: float
    angle_rad: float
    angle_deg: float
    xg: float
    outcome: str  # "goal", "saved_by_keeper", "blocked_by_defender", "off_target"
    is_goal: bool
    is_on_target: bool
    defenders_in_cone: int
    blockers_in_lane: int
    gk_distance_to_goal: float
    gk_positioning_score: float
    shot_speed_mps: float
    shot_speed_kmh: float
    timestamp_sec: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start_xy"] = [round(v, 2) for v in self.start_xy]
        d["end_xy"] = [round(v, 2) for v in self.end_xy]
        d["distance_m"] = round(self.distance_m, 2)
        d["angle_rad"] = round(self.angle_rad, 3)
        d["angle_deg"] = round(self.angle_deg, 1)
        d["xg"] = round(self.xg, 3)
        d["shot_speed_mps"] = round(self.shot_speed_mps, 2)
        d["shot_speed_kmh"] = round(self.shot_speed_kmh, 1)
        d["gk_distance_to_goal"] = round(self.gk_distance_to_goal, 2)
        d["gk_positioning_score"] = round(self.gk_positioning_score, 2)
        d["timestamp_sec"] = round(self.timestamp_sec, 2)
        return d


class ShotEventDetector:
    """
    Kinematic shot spotter and freeze-frame Expected Goals (xG) physics evaluator.
    Operates on metric pitch coordinates (length: 105m, width: 68m, center: (0, 0)).
    Provides global coordinate normalization (detects [0, 105] corner origins and centimeter scales).
    """

    def __init__(
        self,
        fps: float = 25.0,
        min_shot_speed_mps: float = 9.5,  # ~34.2 km/h minimum for shot spotting
        min_flight_frames: int = 3,
        max_flight_frames: int = 75,
        control_radius_m: float = 2.8,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        goal_width: float = 7.32,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.min_shot_speed_mps = float(min_shot_speed_mps)
        self.min_flight_frames = int(min_flight_frames)
        self.max_flight_frames = int(max_flight_frames)
        self.control_radius_m = float(control_radius_m)
        self.half_len = pitch_length / 2.0  # 52.5m
        self.half_wid = pitch_width / 2.0  # 34.0m
        self.goal_half_w = goal_width / 2.0  # 3.66m

    def normalize_coordinates(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[int, Tuple[float, float]]]]:
        """
        Normalizes ball and player coordinates to standard pitch center (0, 0)
        with range x in [-52.5, 52.5] and y in [-34.0, 34.0].
        Auto-detects centimeter scales (x > 200) and corner origins [0, 105] x [0, 68].
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
        # If coordinates are corner-based [0, 105] x [0, 68]:
        # They will span positive values (min_x >= -2.0) and y will be centered near 34.0 (mean_y > 15.0)
        # Or max_x will reach > 75m with min_y >= -2.0 and mean_y > 15.0.
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
        Determines each team attacking direction (right towards +52.5 or left towards -52.5).
        Based on the mean x position of each team players across the match.
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

    @staticmethod
    def compute_subtended_goal_angle(
        shot_xy: Tuple[float, float],
        goal_x: float,
        post1_y: float = -3.66,
        post2_y: float = 3.66,
    ) -> Tuple[float, float]:
        """
        Calculates the visible goal angle subtended by the goal posts from shot_xy.
        Returns (angle_radians, angle_degrees).
        """
        sx, sy = shot_xy
        v1 = (goal_x - sx, post1_y - sy)
        v2 = (goal_x - sx, post2_y - sy)

        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 < 1e-4 or mag2 < 1e-4:
            return math.pi / 2.0, 90.0

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle_rad = math.acos(cos_val)
        angle_deg = math.degrees(angle_rad)
        return angle_rad, angle_deg

    @staticmethod
    def is_point_in_triangle(
        pt: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> bool:
        """
        Tests if 2D point pt lies inside the triangle formed by (p1, p2, p3)
        using barycentric coordinate sign consistency.
        """
        def sign(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])

        d1 = sign(pt, p1, p2)
        d2 = sign(pt, p2, p3)
        d3 = sign(pt, p3, p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def count_defenders_in_shot_cone(
        self,
        shot_xy: Tuple[float, float],
        target_goal_x: float,
        post1_y: float,
        post2_y: float,
        defenders_xy: List[Tuple[float, float]],
    ) -> Tuple[int, int]:
        """
        StatsBomb 360 Spatial Geometry:
        1. defenders_in_cone: Counts opposing players inside the triangle formed by
           (shooter, post1, post2).
        2. blockers_in_lane: Counts defenders within ±1.2m perpendicular distance to the
           direct line ray from shooter to goal center.
        """
        p_shooter = shot_xy
        p_post1 = (target_goal_x, post1_y)
        p_post2 = (target_goal_x, post2_y)
        p_goal_center = (target_goal_x, (post1_y + post2_y) / 2.0)

        in_cone_count = 0
        blocker_count = 0

        ray_x = p_goal_center[0] - p_shooter[0]
        ray_y = p_goal_center[1] - p_shooter[1]
        ray_len = math.hypot(ray_x, ray_y)

        for d_pos in defenders_xy:
            dx, dy = d_pos
            is_between = (
                (dx >= min(p_shooter[0], target_goal_x) - 0.5)
                and (dx <= max(p_shooter[0], target_goal_x) + 0.5)
            )

            if self.is_point_in_triangle(d_pos, p_shooter, p_post1, p_post2):
                in_cone_count += 1

            if ray_len > 1.0 and is_between:
                perp_dist = abs(ray_x * (dy - p_shooter[1]) - ray_y * (dx - p_shooter[0])) / ray_len
                if perp_dist <= 1.2:
                    blocker_count += 1

        return in_cone_count, blocker_count

    def evaluate_goalkeeper_positioning(
        self,
        shot_xy: Tuple[float, float],
        target_goal_x: float,
        gk_xy: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Evaluates the opposing goalkeeper position relative to the shot.
        Returns:
        - gk_distance_to_goal: distance (m) the GK is standing off their goal line
        - gk_positioning_score: [0.0, 1.0], where 1.0 is optimal central angle coverage
          and 0.0 is completely out of position or absent.
        """
        if gk_xy is None:
            return 0.0, 0.0

        gx, gy = gk_xy
        dist_off_line = abs(gx - target_goal_x)

        goal_center_y = 0.0
        to_center_x = target_goal_x - shot_xy[0]
        to_center_y = goal_center_y - shot_xy[1]
        center_dist = math.hypot(to_center_x, to_center_y)

        if center_dist < 1e-4:
            return dist_off_line, 0.5

        perp_dist = abs(to_center_x * (gy - shot_xy[1]) - to_center_y * (gx - shot_xy[0])) / center_dist

        pos_score = 1.0
        if perp_dist > 2.5:
            pos_score *= max(0.1, 1.0 - (perp_dist - 2.5) / 3.0)
        if dist_off_line > 7.0:
            pos_score *= max(0.1, 1.0 - (dist_off_line - 7.0) / 5.0)

        return dist_off_line, max(0.0, min(1.0, pos_score))

    def compute_xg(
        self,
        distance_m: float,
        angle_rad: float,
        defenders_in_cone: int,
        blockers_in_lane: int,
        gk_positioning_score: float,
        shot_speed_mps: float,
    ) -> float:
        """
        Calibrated Freeze-Frame Expected Goals (xG) logistic sigmoid model.
        """
        dist_factor = 0.14 * distance_m
        if distance_m > 16.5:
            dist_factor += 0.04 * (distance_m - 16.5)

        angle_factor = 2.75 * angle_rad
        cone_penalty = 0.32 * min(defenders_in_cone, 5)
        blocker_penalty = 0.45 * min(blockers_in_lane, 3)
        gk_penalty = 0.35 * gk_positioning_score

        speed_bonus = 0.0
        if shot_speed_mps > 22.0:
            speed_bonus = min(0.25, 0.02 * (shot_speed_mps - 22.0))

        logit = 1.32 + angle_factor - dist_factor - cone_penalty - blocker_penalty - gk_penalty + speed_bonus
        xg = 1.0 / (1.0 + math.exp(-logit))

        return max(0.01, min(0.98, xg))

    def detect_shots(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        teams: Dict[int, int],
        goalkeeper_ids: Optional[Dict[int, int]] = None,
        attacking_directions: Optional[Dict[int, str]] = None,
    ) -> List[ShotEvent]:
        """
        Spots kinematic shot events and computes freeze-frame Expected Goals (xG).
        """
        if not ball_trajectory or not player_trajectories:
            return []

        norm_ball, norm_players = self.normalize_coordinates(ball_trajectory, player_trajectories)

        if attacking_directions is None:
            attacking_directions = self.resolve_team_attacking_directions(norm_players, teams)

        frames = sorted(norm_ball.keys())
        if len(frames) < self.min_flight_frames:
            return []

        shots: List[ShotEvent] = []
        shot_id_counter = 1
        skip_until_frame = -1

        for idx, f in enumerate(frames):
            if f < skip_until_frame:
                continue

            bx, by = norm_ball[f]

            f_players = norm_players.get(f, {})
            if not f_players:
                continue

            closest_pid = None
            closest_dist = self.control_radius_m
            for pid, (px, py) in f_players.items():
                d = math.hypot(px - bx, py - by)
                if d < closest_dist:
                    closest_dist = d
                    closest_pid = pid

            if closest_pid is None:
                continue

            shooter_team = teams.get(closest_pid, 1)
            target_direction = attacking_directions.get(shooter_team, "right")
            target_goal_x = self.half_len if target_direction == "right" else -self.half_len

            is_attacking_territory = (
                (target_direction == "right" and bx >= 10.0)
                or (target_direction == "left" and bx <= -10.0)
            )
            if not is_attacking_territory:
                continue

            lookahead = min(len(frames) - 1, idx + 4)
            if lookahead - idx < 2:
                continue

            future_f = frames[lookahead]
            fbx, fby = norm_ball[future_f]
            dt = (future_f - f) / self.fps
            if dt <= 0:
                continue

            dist_traveled = math.hypot(fbx - bx, fby - by)
            initial_speed_mps = dist_traveled / dt

            if initial_speed_mps < self.min_shot_speed_mps:
                continue

            vx = fbx - bx
            vy = fby - by
            dx_to_goal = target_goal_x - bx
            dy_to_goal = 0.0 - by

            v_dot_g = vx * dx_to_goal + vy * dy_to_goal
            mag_v = math.hypot(vx, vy)
            mag_g = math.hypot(dx_to_goal, dy_to_goal)
            if mag_v < 1e-4 or mag_g < 1e-4:
                continue

            cos_aim = v_dot_g / (mag_v * mag_g)
            if cos_aim < math.cos(math.radians(40.0)):
                continue

            shot_start_frame = f
            shot_start_xy = (bx, by)
            shot_end_frame = future_f
            shot_end_xy = (fbx, fby)
            peak_speed = initial_speed_mps

            trace_idx = lookahead
            outcome = "off_target"

            while trace_idx < len(frames) and (frames[trace_idx] - shot_start_frame) <= self.max_flight_frames:
                cur_f = frames[trace_idx]
                cbx, cby = norm_ball[cur_f]
                shot_end_frame = cur_f
                shot_end_xy = (cbx, cby)

                crossed_goal_line = (
                    (target_direction == "right" and cbx >= self.half_len - 0.2)
                    or (target_direction == "left" and cbx <= -self.half_len + 0.2)
                )

                if crossed_goal_line:
                    if abs(cby) <= self.goal_half_w:
                        outcome = "goal"
                    else:
                        outcome = "off_target"
                    break

                cur_players = norm_players.get(cur_f, {})
                intercepted = False
                for opid, (opx, opy) in cur_players.items():
                    op_team = teams.get(opid)
                    if op_team is not None and op_team != shooter_team:
                        op_dist = math.hypot(opx - cbx, opy - cby)
                        if op_dist <= 1.2:
                            is_gk = (goalkeeper_ids is not None and goalkeeper_ids.get(op_team) == opid)
                            in_gk_area = (
                                (target_direction == "right" and cbx >= self.half_len - 16.5)
                                or (target_direction == "left" and cbx <= -self.half_len + 16.5)
                            )
                            if is_gk or in_gk_area:
                                outcome = "saved_by_keeper"
                            else:
                                outcome = "blocked_by_defender"
                            intercepted = True
                            break
                if intercepted:
                    break

                trace_idx += 1

            distance_to_goal = math.hypot(target_goal_x - bx, 0.0 - by)
            angle_rad, angle_deg = self.compute_subtended_goal_angle(
                shot_xy=(bx, by),
                goal_x=target_goal_x,
                post1_y=-self.goal_half_w,
                post2_y=self.goal_half_w,
            )

            opponent_defenders_xy: List[Tuple[float, float]] = []
            opp_gk_xy: Optional[Tuple[float, float]] = None
            opp_team_id = 2 if shooter_team == 1 else 1

            for pid, ppos in f_players.items():
                p_team = teams.get(pid)
                if p_team == opp_team_id:
                    is_gk = (goalkeeper_ids is not None and goalkeeper_ids.get(opp_team_id) == pid)
                    if is_gk:
                        opp_gk_xy = ppos
                    else:
                        opponent_defenders_xy.append(ppos)

            if opp_gk_xy is None and opponent_defenders_xy:
                closest_gk_idx = min(
                    range(len(opponent_defenders_xy)),
                    key=lambda i: math.hypot(target_goal_x - opponent_defenders_xy[i][0], opponent_defenders_xy[i][1])
                )
                opp_candidate = opponent_defenders_xy[closest_gk_idx]
                if math.hypot(target_goal_x - opp_candidate[0], opp_candidate[1]) <= 11.0:
                    opp_gk_xy = opp_candidate
                    opponent_defenders_xy.pop(closest_gk_idx)

            cone_defs, blocker_defs = self.count_defenders_in_shot_cone(
                shot_xy=(bx, by),
                target_goal_x=target_goal_x,
                post1_y=-self.goal_half_w,
                post2_y=self.goal_half_w,
                defenders_xy=opponent_defenders_xy,
            )

            gk_dist, gk_score = self.evaluate_goalkeeper_positioning(
                shot_xy=(bx, by),
                target_goal_x=target_goal_x,
                gk_xy=opp_gk_xy,
            )

            xg_val = self.compute_xg(
                distance_m=distance_to_goal,
                angle_rad=angle_rad,
                defenders_in_cone=cone_defs,
                blockers_in_lane=blocker_defs,
                gk_positioning_score=gk_score,
                shot_speed_mps=peak_speed,
            )

            is_goal = (outcome == "goal")
            is_on_target = (outcome in ("goal", "saved_by_keeper"))

            event = ShotEvent(
                shot_id=shot_id_counter,
                shooter_id=closest_pid,
                shooter_team=shooter_team,
                start_frame=shot_start_frame,
                end_frame=shot_end_frame,
                start_xy=shot_start_xy,
                end_xy=shot_end_xy,
                target_goal=target_direction,
                distance_m=distance_to_goal,
                angle_rad=angle_rad,
                angle_deg=angle_deg,
                xg=xg_val,
                outcome=outcome,
                is_goal=is_goal,
                is_on_target=is_on_target,
                defenders_in_cone=cone_defs,
                blockers_in_lane=blocker_defs,
                gk_distance_to_goal=gk_dist,
                gk_positioning_score=gk_score,
                shot_speed_mps=peak_speed,
                shot_speed_kmh=peak_speed * 3.6,
                timestamp_sec=shot_start_frame / self.fps,
            )
            shots.append(event)
            shot_id_counter += 1

            skip_until_frame = shot_end_frame + int(self.fps * 1.5)

        return shots

    def summarize_shooting_intelligence(
        self,
        shots: List[ShotEvent],
    ) -> Dict[str, Any]:
        """
        Aggregates match shooting metrics, clinical finishing efficiency (Goals - xG),
        and top shooter leaderboard.
        """
        total_shots = len(shots)
        if total_shots == 0:
            return {
                "total_shots": 0,
                "shots_on_target": 0,
                "goals": 0,
                "conversion_rate_pct": 0.0,
                "total_xg": 0.0,
                "xg_per_shot": 0.0,
                "goals_minus_xg": 0.0,
                "team_breakdown": {
                    "team1": {"shots": 0, "on_target": 0, "goals": 0, "total_xg": 0.0},
                    "team2": {"shots": 0, "on_target": 0, "goals": 0, "total_xg": 0.0},
                },
                "top_shooters": [],
                "shots": [],
            }

        goals = sum(1 for s in shots if s.is_goal)
        on_target = sum(1 for s in shots if s.is_on_target)
        total_xg = sum(s.xg for s in shots)
        conversion_pct = (goals / total_shots * 100.0) if total_shots else 0.0
        xg_per_shot = total_xg / total_shots if total_shots else 0.0
        goals_minus_xg = goals - total_xg

        team_stats: Dict[str, Dict[str, Any]] = {
            "team1": {"shots": 0, "on_target": 0, "goals": 0, "total_xg": 0.0},
            "team2": {"shots": 0, "on_target": 0, "goals": 0, "total_xg": 0.0},
        }
        player_stats: Dict[int, Dict[str, Any]] = {}

        for s in shots:
            t_key = f"team{s.shooter_team}"
            if t_key not in team_stats:
                team_stats[t_key] = {"shots": 0, "on_target": 0, "goals": 0, "total_xg": 0.0}
            team_stats[t_key]["shots"] += 1
            if s.is_on_target:
                team_stats[t_key]["on_target"] += 1
            if s.is_goal:
                team_stats[t_key]["goals"] += 1
            team_stats[t_key]["total_xg"] = round(team_stats[t_key]["total_xg"] + s.xg, 2)

            if s.shooter_id not in player_stats:
                player_stats[s.shooter_id] = {
                    "player_id": s.shooter_id,
                    "team": s.shooter_team,
                    "shots": 0,
                    "on_target": 0,
                    "goals": 0,
                    "total_xg": 0.0,
                }
            player_stats[s.shooter_id]["shots"] += 1
            if s.is_on_target:
                player_stats[s.shooter_id]["on_target"] += 1
            if s.is_goal:
                player_stats[s.shooter_id]["goals"] += 1
            player_stats[s.shooter_id]["total_xg"] = round(player_stats[s.shooter_id]["total_xg"] + s.xg, 2)

        top_shooters = sorted(
            player_stats.values(),
            key=lambda x: (x["goals"], x["total_xg"], x["shots"]),
            reverse=True,
        )

        return {
            "total_shots": total_shots,
            "shots_on_target": on_target,
            "goals": goals,
            "conversion_rate_pct": round(conversion_pct, 1),
            "total_xg": round(total_xg, 2),
            "xg_per_shot": round(xg_per_shot, 2),
            "goals_minus_xg": round(goals_minus_xg, 2),
            "team_breakdown": team_stats,
            "top_shooters": top_shooters,
            "shots": [s.to_dict() for s in shots],
        }

    def render_shot_map(
        self,
        shots: List[ShotEvent],
        output_path: Path,
        title: Optional[str] = None,
    ) -> None:
        """
        Renders a football pitch shot map visualizing shot origin locations,
        xG magnitude (bubble size), and outcome classification.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")

        # Pitch Boundary and markings (metric scale [-52.5, 52.5] x [-34, 34])
        ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="#475569", lw=2)
        ax.plot([0, 0], [-34, 34], color="#475569", lw=1.5, linestyle="--")
        circle = plt.Circle((0, 0), 9.15, color="#475569", fill=False, lw=1.5)
        ax.add_patch(circle)
        ax.scatter(0, 0, color="#475569", s=20)

        # Right Penalty Box (18-yard: 16.5m deep, 40.32m wide)
        rect_right_box = plt.Rectangle((52.5 - 16.5, -20.16), 16.5, 40.32, fill=False, ec="#475569", lw=1.5)
        ax.add_patch(rect_right_box)
        rect_right_6 = plt.Rectangle((52.5 - 5.5, -9.16), 5.5, 18.32, fill=False, ec="#475569", lw=1.2)
        ax.add_patch(rect_right_6)
        rect_right_goal = plt.Rectangle((52.5, -3.66), 2.0, 7.32, fill=False, ec="#38bdf8", lw=2.5)
        ax.add_patch(rect_right_goal)
        ax.scatter(52.5 - 11.0, 0, color="#475569", s=25)

        # Left Penalty Box
        rect_left_box = plt.Rectangle((-52.5, -20.16), 16.5, 40.32, fill=False, ec="#475569", lw=1.5)
        ax.add_patch(rect_left_box)
        rect_left_6 = plt.Rectangle((-52.5, -9.16), 5.5, 18.32, fill=False, ec="#475569", lw=1.2)
        ax.add_patch(rect_left_6)
        rect_left_goal = plt.Rectangle((-54.5, -3.66), 2.0, 7.32, fill=False, ec="#38bdf8", lw=2.5)
        ax.add_patch(rect_left_goal)
        ax.scatter(-52.5 + 11.0, 0, color="#475569", s=25)

        outcome_styles = {
            "goal": {"marker": "*", "color": "#facc15", "edge": "#eab308", "label": "Goal", "zorder": 10},
            "saved_by_keeper": {"marker": "o", "color": "#38bdf8", "edge": "#0284c7", "label": "Saved by GK", "zorder": 8},
            "blocked_by_defender": {"marker": "^", "color": "#fb923c", "edge": "#ea580c", "label": "Blocked", "zorder": 7},
            "off_target": {"marker": "x", "color": "#94a3b8", "edge": "#64748b", "label": "Off Target", "zorder": 6},
        }

        plotted_labels = set()

        for s in shots:
            style = outcome_styles.get(s.outcome, outcome_styles["off_target"])
            size = max(80, int(80 + s.xg * 550))
            lbl = style["label"] if style["label"] not in plotted_labels else ""
            if lbl:
                plotted_labels.add(lbl)

            arrow_color = style["color"]
            ax.annotate(
                "",
                xy=s.end_xy,
                xytext=s.start_xy,
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    lw=max(1.2, s.xg * 3.5),
                    alpha=0.65,
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=style["zorder"] - 1,
            )

            if style["marker"] == "x":
                ax.scatter(
                    s.start_xy[0],
                    s.start_xy[1],
                    s=size,
                    marker=style["marker"],
                    color=style["color"],
                    linewidths=2,
                    label=lbl,
                    zorder=style["zorder"],
                )
            else:
                ax.scatter(
                    s.start_xy[0],
                    s.start_xy[1],
                    s=size,
                    marker=style["marker"],
                    facecolors=style["color"],
                    edgecolors=style["edge"],
                    linewidths=1.8,
                    label=lbl,
                    zorder=style["zorder"],
                )

            label_text = f"#{s.shooter_id}\n({s.xg:.2f})"
            ax.annotate(
                label_text,
                xy=s.start_xy,
                xytext=(0, 10 if s.start_xy[1] >= 0 else -16),
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=style["zorder"] + 1,
            )

        ax.set_xlim(-56, 56)
        ax.set_ylim(-37, 37)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        tot_shots = len(shots)
        tot_goals = sum(1 for s in shots if s.is_goal)
        tot_xg = sum(s.xg for s in shots)

        if title is None:
            title = f"Tactical Shot Map & xG Analysis — Shots: {tot_shots} | Goals: {tot_goals} | Total xG: {tot_xg:.2f}"

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            facecolor="#1e293b",
            edgecolor="#475569",
            labelcolor="white",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
