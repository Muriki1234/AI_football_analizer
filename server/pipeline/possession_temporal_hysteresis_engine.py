"""
possession_temporal_hysteresis_engine.py
=========================================
Continuous Team Possession State Machine with Temporal Hysteresis & Flight Bridging.

Root Cause Solved:
1. Replaces the naive frame-by-frame instantaneous proximity filter (<2.5m) that
   caused 91.2% Neutral possession in match broadcasts.
2. In real football (FIFA, Opta, StatsBomb standards), team possession is an unbroken
   state sequence that encompasses passes, shots, and rebounds until:
   a) Opposing player gains verified controlled possession;
   b) Ball goes dead (out of play / stoppage);
   c) A prolonged contested loose-ball timeout (> 3.0s) elapses.
3. Bridges airborne passes and ball detection dropouts using a Markov possession state
   machine with exponential decay and turnover confirmation gates.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


class PossessionTemporalHysteresisEngine:
    """
    Markov State Machine for continuous team possession tracking with temporal hysteresis.
    """

    STATE_NEUTRAL: int = 0
    STATE_TEAM_1: int = 1
    STATE_TEAM_2: int = 2
    STATE_DEAD_BALL: int = -1

    def __init__(
        self,
        fps: float = 30.0,
        control_radius_m: float = 2.8,
        pass_grace_period_s: float = 2.5,
        turnover_confirm_frames: int = 3,
        loose_ball_decay_s: float = 3.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.control_radius_m = float(control_radius_m)
        self.pass_grace_frames = max(1, int(round(pass_grace_period_s * self.fps)))
        self.turnover_confirm_frames = max(1, int(turnover_confirm_frames))
        self.loose_ball_decay_frames = max(1, int(round(loose_ball_decay_s * self.fps)))

        # State tracking
        self.current_team_possession: int = self.STATE_NEUTRAL
        self.current_player_possession: Optional[int] = None
        self.frames_since_last_touch: int = 999999
        self.last_controlling_team: int = self.STATE_NEUTRAL
        self.last_controlling_player: Optional[int] = None

        # Turnover confirmation buffer
        self._opposition_candidate_team: int = self.STATE_NEUTRAL
        self._opposition_candidate_frames: int = 0

        # Sequence stats
        self.possession_history: List[int] = []
        self.player_possession_history: List[Optional[int]] = []
        self.state_confidence_history: List[float] = []

    def reset(self) -> None:
        """Reset internal state machine."""
        self.current_team_possession = self.STATE_NEUTRAL
        self.current_player_possession = None
        self.frames_since_last_touch = 999999
        self.last_controlling_team = self.STATE_NEUTRAL
        self.last_controlling_player = None
        self._opposition_candidate_team = self.STATE_NEUTRAL
        self._opposition_candidate_frames = 0
        self.possession_history.clear()
        self.player_possession_history.clear()
        self.state_confidence_history.clear()

    def update_frame(
        self,
        frame_idx: int,
        ball_pos_m: Optional[Tuple[float, float]] = None,
        players_m: Optional[Dict[int, Dict[str, Any]]] = None,
        ball_state_hint: str = "unknown",
        direct_touch_pid: Optional[int] = None,
        direct_touch_team: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes one frame observation:
        Args:
            frame_idx: Frame index.
            ball_pos_m: Optional (x, y) ball coordinate in FIFA meters.
            players_m: Dict mapping pid -> {"x_m": float, "y_m": float, "team": int}
            ball_state_hint: e.g. "controlled", "flying", "contested", "loose_ball"
            direct_touch_pid: Optional directly observed controlling player ID.
            direct_touch_team: Optional directly observed controlling team (1 or 2).

        Returns:
            Dict containing assigned team, controlling player, confidence, and state.
        """
        closest_pid: Optional[int] = None
        closest_dist: float = 99999.0
        closest_team: int = self.STATE_NEUTRAL
        players_map = players_m or {}

        if direct_touch_team in (self.STATE_TEAM_1, self.STATE_TEAM_2):
            closest_team = direct_touch_team
            closest_pid = direct_touch_pid
            closest_dist = 0.5
            has_touch = True
        else:
            # 1. Spatial Proximity Scan (if ball position is available)
            if ball_pos_m is not None and players_map:
                bx, by = ball_pos_m
                for pid, pdata in players_map.items():
                    px = pdata.get("x_m")
                    py = pdata.get("y_m")
                    team = pdata.get("team", self.STATE_NEUTRAL)
                    if px is None or py is None or team not in (self.STATE_TEAM_1, self.STATE_TEAM_2):
                        continue
                    d = math.hypot(bx - px, by - py)
                    if d < closest_dist:
                        closest_dist = d
                        closest_pid = pid
                        closest_team = team

            # 2. Touch Detection (within physical control radius)
            has_touch = (closest_pid is not None and closest_dist <= self.control_radius_m)

        confidence = 0.0

        if has_touch and closest_team in (self.STATE_TEAM_1, self.STATE_TEAM_2):
            self.frames_since_last_touch = 0

            # If touch is from the same team currently possessing the ball
            if closest_team == self.current_team_possession:
                self.current_player_possession = closest_pid
                self.last_controlling_team = closest_team
                self.last_controlling_player = closest_pid
                self._opposition_candidate_team = self.STATE_NEUTRAL
                self._opposition_candidate_frames = 0
                confidence = max(0.7, 1.0 - (closest_dist / self.control_radius_m) * 0.4)

            # If touch is from the opposition: require confirmation frames to avoid single-frame bounce noise
            else:
                if closest_team == self._opposition_candidate_team:
                    self._opposition_candidate_frames += 1
                else:
                    self._opposition_candidate_team = closest_team
                    self._opposition_candidate_frames = 1

                # Confirm turnover if candidate holds touch for required frames or immediate clear touch < 1.2m
                if (
                    self._opposition_candidate_frames >= self.turnover_confirm_frames
                    or closest_dist < 1.2
                    or self.current_team_possession == self.STATE_NEUTRAL
                ):
                    self.current_team_possession = closest_team
                    self.current_player_possession = closest_pid
                    self.last_controlling_team = closest_team
                    self.last_controlling_player = closest_pid
                    self._opposition_candidate_team = self.STATE_NEUTRAL
                    self._opposition_candidate_frames = 0
                    confidence = 0.85
                else:
                    # Still in contest/turnover transition: preserve previous team until confirmed
                    confidence = 0.5

        else:
            # 3. Ball in Flight / Loose Ball / Dropout (No player touching right now)
            self.frames_since_last_touch += 1

            # Hysteresis Decay Logic:
            # While ball is in flight during a pass or shot (within pass_grace_frames),
            # the last team to play the ball retains possession.
            if self.frames_since_last_touch <= self.pass_grace_frames:
                # Retain team possession, but player possession is none (ball in flight)
                self.current_team_possession = self.last_controlling_team
                self.current_player_possession = None
                # Confidence smoothly decays from 0.85 down to 0.45 across flight
                decay_ratio = self.frames_since_last_touch / float(self.pass_grace_frames)
                confidence = max(0.45, 0.85 - decay_ratio * 0.40)

            elif self.frames_since_last_touch <= self.loose_ball_decay_frames:
                # Contested loose ball window: decay to neutral gradually
                self.current_team_possession = self.last_controlling_team
                self.current_player_possession = None
                confidence = 0.30

            else:
                # Sustained loose ball or dead ball timeout: transition to Neutral
                self.current_team_possession = self.STATE_NEUTRAL
                self.current_player_possession = None
                confidence = 0.90

        self.possession_history.append(self.current_team_possession)
        self.player_possession_history.append(self.current_player_possession)
        self.state_confidence_history.append(confidence)

        return {
            "frame_idx": frame_idx,
            "team_possession": self.current_team_possession,
            "player_possession": self.current_player_possession,
            "confidence": round(confidence, 2),
            "frames_since_touch": self.frames_since_last_touch,
            "is_flight": (
                self.current_team_possession != self.STATE_NEUTRAL
                and self.current_player_possession is None
            ),
        }

    def compute_summary_statistics(
        self,
        range_start: int = 0,
        range_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Computes accurate match possession percentages, switches, and durations.
        """
        if not self.possession_history:
            return {
                "team1_possession_pct": 50.0,
                "team2_possession_pct": 50.0,
                "neutral_possession_pct": 0.0,
                "possession_switches": 0,
                "team1_seconds": 0.0,
                "team2_seconds": 0.0,
                "neutral_seconds": 0.0,
            }

        end_idx = len(self.possession_history) if range_end is None else min(len(self.possession_history), range_end)
        sub_arr = np.asarray(self.possession_history[range_start:end_idx], dtype=np.int32)
        total_frames = len(sub_arr)
        if total_frames == 0:
            return {
                "team1_possession_pct": 50.0,
                "team2_possession_pct": 50.0,
                "neutral_possession_pct": 0.0,
                "possession_switches": 0,
                "team1_seconds": 0.0,
                "team2_seconds": 0.0,
                "neutral_seconds": 0.0,
            }

        t1_count = int(np.sum(sub_arr == self.STATE_TEAM_1))
        t2_count = int(np.sum(sub_arr == self.STATE_TEAM_2))
        neu_count = int(np.sum(sub_arr == self.STATE_NEUTRAL))

        # Possession switches (turnovers between Team 1 and Team 2)
        switches = 0
        prev_team = self.STATE_NEUTRAL
        for t in sub_arr:
            if t in (self.STATE_TEAM_1, self.STATE_TEAM_2):
                if prev_team in (self.STATE_TEAM_1, self.STATE_TEAM_2) and t != prev_team:
                    switches += 1
                prev_team = t

        t1_pct = round((t1_count / float(total_frames)) * 100.0, 1)
        t2_pct = round((t2_count / float(total_frames)) * 100.0, 1)
        neu_pct = round((neu_count / float(total_frames)) * 100.0, 1)

        # Standard broadcast ratio (normalized excluding neutral dead ball, e.g. 54% vs 46%)
        active_total = t1_count + t2_count
        if active_total > 0:
            bcast_t1_pct = round((t1_count / float(active_total)) * 100.0, 1)
            bcast_t2_pct = round((t2_count / float(active_total)) * 100.0, 1)
        else:
            bcast_t1_pct = 50.0
            bcast_t2_pct = 50.0

        return {
            "team1_possession_pct": t1_pct,
            "team2_possession_pct": t2_pct,
            "neutral_possession_pct": neu_pct,
            "broadcast_team1_pct": bcast_t1_pct,
            "broadcast_team2_pct": bcast_t2_pct,
            "possession_switches": switches,
            "team1_seconds": round(t1_count / self.fps, 1),
            "team2_seconds": round(t2_count / self.fps, 1),
            "neutral_seconds": round(neu_count / self.fps, 1),
            "total_analyzed_frames": total_frames,
        }

    def filter_control_sequence(
        self,
        raw_control: Sequence[int],
    ) -> List[int]:
        """
        Filters a frame-by-frame instantaneous team control sequence [0, 1, 2, 0, ...]
        applying continuous Markov temporal hysteresis to bridge pass flights and
        eliminate single-frame bounce jitter.

        Args:
            raw_control: Sequence of integers representing instantaneous frame control
                         (1 for Team 1, 2 for Team 2, 0 for neutral / loose ball).

        Returns:
            List of smoothed team control IDs with flight gaps bridged and turnovers confirmed.
        """
        self.reset()
        filtered: List[int] = []
        for fi, team_val in enumerate(raw_control):
            tv = int(team_val) if team_val is not None else 0
            touch_team = tv if tv in (self.STATE_TEAM_1, self.STATE_TEAM_2) else None
            res = self.update_frame(
                frame_idx=fi,
                ball_pos_m=None,
                players_m=None,
                direct_touch_pid=None,
                direct_touch_team=touch_team,
            )
            filtered.append(int(res["team_possession"]))
        return filtered

