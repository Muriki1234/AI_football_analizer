"""
pass_event_detector.py — Ball-Action Pass Event Spotting & Passing Network Analyzer

Inspired by SoccerNet Ball Action Spotting (CVPR 2025/2026) and Kloppy/StatsBomb event models.
Implements a 3-state kinematic automaton (Possession -> Flight -> Reception/Interception):
1. Detects pass release, flight duration, reception, and turnover events from 2D tracking data.
2. Classifies pass outcomes: 'completed', 'intercepted', 'incomplete_out_of_bounds'.
3. Computes advanced tactical metrics: progressive passes, Zone 14 entries, penalty box entries.
4. Builds team-level Passing Network graphs (player node centroids and directed pass edges).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PassEvent:
    pass_id: int
    passer_id: int
    passer_team: int
    receiver_id: Optional[int]
    receiver_team: Optional[int]
    start_frame: int
    end_frame: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    outcome: str  # "completed", "intercepted", "incomplete_out_of_bounds"
    pass_distance: float  # meters
    flight_time_sec: float
    avg_speed_mps: float
    is_progressive: bool
    is_zone14_entry: bool
    is_box_entry: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pass_id"] = int(self.pass_id)
        d["passer_id"] = int(self.passer_id) if self.passer_id is not None else None
        d["receiver_id"] = int(self.receiver_id) if self.receiver_id is not None else None
        d["team"] = int(self.team)
        d["start_frame"] = int(self.start_frame)
        d["end_frame"] = int(self.end_frame)
        d["start_xy"] = [float(round(v, 2)) for v in self.start_xy]
        d["end_xy"] = [float(round(v, 2)) for v in self.end_xy]
        d["pass_distance"] = float(round(self.pass_distance, 2))
        d["flight_time_sec"] = float(round(self.flight_time_sec, 2))
        d["avg_speed_mps"] = float(round(self.avg_speed_mps, 2))
        d["is_progressive"] = bool(self.is_progressive)
        d["is_zone14_entry"] = bool(self.is_zone14_entry)
        d["is_box_entry"] = bool(self.is_box_entry)
        return d


class PassEventDetector:
    """
    Kinematic ball action spotter for pass events and passing networks.
    Operates on metric coordinates (length: 105m, width: 68m, center: (0, 0)).
    Provides global trajectory coordinate normalization (auto-detects cm/m and corner/center origins).
    """

    def __init__(
        self,
        fps: float = 25.0,
        control_radius_m: float = 2.8,
        min_pass_distance_m: float = 3.5,
        min_flight_frames: int = 3,
        max_flight_frames: int = 150,
        progressive_distance_m: float = 9.15,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.control_radius_m = float(control_radius_m)
        self.min_pass_distance_m = float(min_pass_distance_m)
        self.min_flight_frames = int(min_flight_frames)
        self.max_flight_frames = int(max_flight_frames)
        self.progressive_distance_m = float(progressive_distance_m)
        self.half_len = pitch_length / 2.0
        self.half_wid = pitch_width / 2.0

    def normalize_trajectories(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[int, Tuple[float, float]]]]:
        """
        Global consistent coordinate normalization:
        1. Checks global extent across all points.
        2. Detects if cm (scales by 0.01).
        3. Detects if origin is at pitch corner [0, 105] (shifts to center [-52.5, 52.5]).
        Applies identical transformation to both ball and all players.
        """
        sample_x = [p[0] for p in ball_trajectory.values()]
        sample_y = [p[1] for p in ball_trajectory.values()]
        if not sample_x:
            return ball_trajectory, player_trajectories

        min_x, max_x = min(sample_x), max(sample_x)
        min_y, max_y = min(sample_y), max(sample_y)

        scale = 1.0
        if max_x > 200.0 or min_x < -200.0 or max_y > 200.0 or min_y < -200.0:
            scale = 0.01
            min_x *= scale
            max_x *= scale
            min_y *= scale
            max_y *= scale

        shift_x = 0.0
        shift_y = 0.0
        # If all points are strictly positive and reach > 50m, it's corner-based [0, 105] x [0, 68]
        if min_x >= -2.0 and max_x > 50.0:
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

    def _euclidean_dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _is_in_bounds(self, xy: Tuple[float, float], margin: float = 1.5) -> bool:
        x, y = xy
        return (-self.half_len - margin <= x <= self.half_len + margin) and (
            -self.half_wid - margin <= y <= self.half_wid + margin
        )

    def _is_zone14(self, xy: Tuple[float, float], attacking_direction: int = 1) -> bool:
        """Zone 14: Central attacking area (x in [17.5, 35.0], y in [-10.1, 10.1])."""
        x, y = xy
        if attacking_direction > 0:
            return 17.5 <= x <= 35.0 and -10.1 <= y <= 10.1
        else:
            return -35.0 <= x <= -17.5 and -10.1 <= y <= 10.1

    def _is_penalty_box(self, xy: Tuple[float, float], attacking_direction: int = 1) -> bool:
        """Penalty box: 16.5m from goal line, width 40.3m (y in [-20.15, 20.15])."""
        x, y = xy
        box_x_min = self.half_len - 16.5
        box_y_max = 20.15
        if attacking_direction > 0:
            return x >= box_x_min and abs(y) <= box_y_max
        else:
            return x <= -box_x_min and abs(y) <= box_y_max

    def _find_closest_player(
        self,
        ball_xy: Tuple[float, float],
        frame_players: Dict[int, Tuple[float, float]],
    ) -> Tuple[Optional[int], float]:
        closest_id = None
        min_dist = float("inf")
        for pid, p_xy in frame_players.items():
            d = self._euclidean_dist(ball_xy, p_xy)
            if d < min_dist:
                min_dist = d
                closest_id = pid
        return closest_id, min_dist

    def detect_passes(
        self,
        ball_trajectory: Dict[int, Tuple[float, float]],
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]],
        player_teams: Dict[int, int],
        team_attacking_direction: Optional[Dict[int, int]] = None,
    ) -> List[PassEvent]:
        """
        Detects pass events across frames.
        ball_trajectory: {frame_idx: (x, y)}
        player_trajectories: {frame_idx: {player_id: (x, y)}}
        player_teams: {player_id: team_id}
        """
        if not ball_trajectory or not player_trajectories:
            return []

        norm_ball, norm_players = self.normalize_trajectories(ball_trajectory, player_trajectories)
        team_dirs = team_attacking_direction or {1: 1, 2: -1}
        sorted_frames = sorted(norm_ball.keys())
        passes: List[PassEvent] = []

        current_possessor: Optional[int] = None
        consecutive_control_frames = 0
        pass_in_flight = False
        candidate_passer_id: Optional[int] = None
        candidate_start_frame: int = 0
        candidate_start_xy: Tuple[float, float] = (0.0, 0.0)

        for f_idx in sorted_frames:
            b_xy = norm_ball[f_idx]
            frame_players = norm_players.get(f_idx, {})

            closest_pid, closest_dist = self._find_closest_player(b_xy, frame_players)
            has_controller = closest_pid is not None and closest_dist <= self.control_radius_m

            if not pass_in_flight:
                if has_controller:
                    if closest_pid == current_possessor:
                        consecutive_control_frames += 1
                    else:
                        current_possessor = closest_pid
                        consecutive_control_frames = 1
                else:
                    if current_possessor is not None and consecutive_control_frames >= 2:
                        pass_in_flight = True
                        candidate_passer_id = current_possessor
                        candidate_start_frame = f_idx - 1
                        last_f = candidate_start_frame if candidate_start_frame in norm_ball else f_idx
                        candidate_start_xy = norm_ball[last_f]
                    current_possessor = None
                    consecutive_control_frames = 0
            else:
                flight_frames = f_idx - candidate_start_frame

                if not self._is_in_bounds(b_xy):
                    dist = self._euclidean_dist(candidate_start_xy, b_xy)
                    if dist >= self.min_pass_distance_m:
                        passer_team = player_teams.get(candidate_passer_id, 0)
                        passes.append(
                            PassEvent(
                                pass_id=len(passes) + 1,
                                passer_id=candidate_passer_id,
                                passer_team=passer_team,
                                receiver_id=None,
                                receiver_team=None,
                                start_frame=candidate_start_frame,
                                end_frame=f_idx,
                                start_xy=candidate_start_xy,
                                end_xy=b_xy,
                                outcome="incomplete_out_of_bounds",
                                pass_distance=dist,
                                flight_time_sec=flight_frames / self.fps,
                                avg_speed_mps=dist / max(1e-3, (flight_frames / self.fps)),
                                is_progressive=False,
                                is_zone14_entry=False,
                                is_box_entry=False,
                            )
                        )
                    pass_in_flight = False
                    current_possessor = None
                    consecutive_control_frames = 0
                    continue

                if has_controller and closest_pid != candidate_passer_id:
                    if flight_frames >= self.min_flight_frames:
                        dist = self._euclidean_dist(candidate_start_xy, b_xy)
                        if dist >= self.min_pass_distance_m:
                            passer_team = player_teams.get(candidate_passer_id, 0)
                            receiver_team = player_teams.get(closest_pid, 0)
                            is_completed = (passer_team != 0 and passer_team == receiver_team)
                            outcome = "completed" if is_completed else "intercepted"

                            atk_dir = team_dirs.get(passer_team, 1)
                            dx_prog = (b_xy[0] - candidate_start_xy[0]) * atk_dir
                            is_prog = is_completed and (dx_prog >= self.progressive_distance_m)

                            is_z14 = is_completed and (not self._is_zone14(candidate_start_xy, atk_dir)) and self._is_zone14(b_xy, atk_dir)
                            is_box = is_completed and (not self._is_penalty_box(candidate_start_xy, atk_dir)) and self._is_penalty_box(b_xy, atk_dir)

                            passes.append(
                                PassEvent(
                                    pass_id=len(passes) + 1,
                                    passer_id=candidate_passer_id,
                                    passer_team=passer_team,
                                    receiver_id=closest_pid,
                                    receiver_team=receiver_team,
                                    start_frame=candidate_start_frame,
                                    end_frame=f_idx,
                                    start_xy=candidate_start_xy,
                                    end_xy=b_xy,
                                    outcome=outcome,
                                    pass_distance=dist,
                                    flight_time_sec=flight_frames / self.fps,
                                    avg_speed_mps=dist / max(1e-3, (flight_frames / self.fps)),
                                    is_progressive=is_prog,
                                    is_zone14_entry=is_z14,
                                    is_box_entry=is_box,
                                )
                            )

                    pass_in_flight = False
                    current_possessor = closest_pid
                    consecutive_control_frames = 1
                elif flight_frames > self.max_flight_frames:
                    pass_in_flight = False
                    current_possessor = None
                    consecutive_control_frames = 0

        return passes

    def build_pass_network(
        self,
        passes: List[PassEvent],
        team_id: int,
    ) -> Dict[str, Any]:
        """
        Builds a passing network graph for a given team:
        - Nodes: players with average pass locations (centroid) and pass volume.
        - Edges: directed connections between passer and receiver with pass counts.
        """
        team_passes = [p for p in passes if p.passer_team == team_id]
        completed = [p for p in team_passes if p.outcome == "completed"]

        player_coords: Dict[int, List[Tuple[float, float]]] = {}
        pass_counts: Dict[int, int] = {}
        receive_counts: Dict[int, int] = {}

        for p in team_passes:
            pass_counts[p.passer_id] = pass_counts.get(p.passer_id, 0) + 1
            player_coords.setdefault(p.passer_id, []).append(p.start_xy)

        for p in completed:
            if p.receiver_id is not None:
                receive_counts[p.receiver_id] = receive_counts.get(p.receiver_id, 0) + 1
                player_coords.setdefault(p.receiver_id, []).append(p.end_xy)

        nodes: List[Dict[str, Any]] = []
        for pid, coords in player_coords.items():
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            nodes.append({
                "player_id": int(pid) if str(pid).isdigit() else str(pid),
                "centroid_xy": [float(round(avg_x, 2)), float(round(avg_y, 2))],
                "passes_made": int(pass_counts.get(pid, 0)),
                "passes_received": int(receive_counts.get(pid, 0)),
            })
        nodes.sort(key=lambda n: n["passes_made"], reverse=True)

        edge_counts: Dict[Tuple[int, int], int] = {}
        for p in completed:
            if p.receiver_id is not None and p.receiver_id != p.passer_id:
                key = (int(p.passer_id), int(p.receiver_id))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        edges: List[Dict[str, Any]] = []
        for (src, dst), count in edge_counts.items():
            edges.append({
                "source": int(src) if str(src).isdigit() else str(src),
                "target": int(dst) if str(dst).isdigit() else str(dst),
                "count": int(count),
            })
        edges.sort(key=lambda e: e["count"], reverse=True)

        total_p = int(len(team_passes))
        comp_p = int(len(completed))
        comp_rate = float(round((comp_p / total_p * 100.0) if total_p > 0 else 0.0, 1))
        prog_p = int(sum(1 for p in completed if p.is_progressive))
        z14_p = int(sum(1 for p in completed if p.is_zone14_entry))
        box_p = int(sum(1 for p in completed if p.is_box_entry))

        return {
            "team_id": int(team_id),
            "total_passes": total_p,
            "completed_passes": comp_p,
            "completion_rate_pct": comp_rate,
            "progressive_passes": prog_p,
            "zone14_entries": z14_p,
            "box_entries": box_p,
            "nodes": nodes,
            "edges": edges,
        }
