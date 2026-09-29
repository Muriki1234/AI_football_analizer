"""
benchmark_possession_pixel_sweep.py
===================================
P0-7: Systematic Pixel-Space Proximity Threshold Sweep for Dual-Mode Possession

Sweeps d_px in [20, 30, 40, 45, 50, 60] px.
Evaluates:
- Possession Precision (%)
- Possession Recall (%)
- False Acquisition Count (Opponent falsely awarded turnover)
- Missed Possession Frames (Ball carrier's legitimate control dropped to Neutral)
- State Flip Stability (Number of false state transitions / ping-pong toggles)

Goal: Find the empirical Pareto-optimal threshold d_px that maximizes true possession
retention without introducing spurious turnover acquisitions.
"""

import json
import math
import numpy as np


class DualModePossessionHysteresisEngine:
    """
    Experimental Dual-Mode Possession Engine supporting both metric (m)
    and pixel-space (px) proximity matching.
    """

    STATE_NEUTRAL: int = 0
    STATE_TEAM_1: int = 1
    STATE_TEAM_2: int = 2

    def __init__(
        self,
        fps: float = 25.0,
        control_radius_m: float = 2.8,
        control_radius_px: float = 45.0,
        pass_grace_period_s: float = 2.0,
        turnover_confirm_frames: int = 3,
        loose_ball_decay_s: float = 3.0,
    ):
        self.fps = fps
        self.control_radius_m = control_radius_m
        self.control_radius_px = control_radius_px
        self.pass_grace_frames = int(round(pass_grace_period_s * fps))
        self.turnover_confirm_frames = turnover_confirm_frames
        self.loose_ball_decay_frames = int(round(loose_ball_decay_s * fps))

        self.current_team = self.STATE_NEUTRAL
        self.current_player = None
        self.last_team = self.STATE_NEUTRAL
        self.last_player = None
        self.frames_since_touch = 999999
        self._opp_candidate_team = self.STATE_NEUTRAL
        self._opp_candidate_frames = 0

    def update_frame(
        self,
        frame_idx: int,
        ball_pos_m=None,
        players_m=None,
        ball_pos_px=None,
        players_px=None,
    ):
        closest_pid = None
        closest_dist = 999999.0
        closest_team = self.STATE_NEUTRAL
        has_touch = False

        # Mode A: Metric Space (when Homography is valid)
        if ball_pos_m is not None and players_m:
            bx, by = ball_pos_m
            for pid, pdata in players_m.items():
                px, py = pdata.get("x_m"), pdata.get("y_m")
                team = pdata.get("team", self.STATE_NEUTRAL)
                if px is None or py is None or team not in (self.STATE_TEAM_1, self.STATE_TEAM_2):
                    continue
                d = math.hypot(bx - px, by - py)
                if d < closest_dist:
                    closest_dist = d
                    closest_pid = pid
                    closest_team = team
            has_touch = (closest_pid is not None and closest_dist <= self.control_radius_m)

        # Mode B: Image-Space Fallback (when Homography is unavailable)
        elif ball_pos_px is not None and players_px:
            bx, by = ball_pos_px
            for pid, pdata in players_px.items():
                px, py = pdata.get("px_x"), pdata.get("px_y")
                team = pdata.get("team", self.STATE_NEUTRAL)
                if px is None or py is None or team not in (self.STATE_TEAM_1, self.STATE_TEAM_2):
                    continue
                d = math.hypot(bx - px, by - py)
                if d < closest_dist:
                    closest_dist = d
                    closest_pid = pid
                    closest_team = team
            has_touch = (closest_pid is not None and closest_dist <= self.control_radius_px)

        if has_touch and closest_team in (self.STATE_TEAM_1, self.STATE_TEAM_2):
            self.frames_since_touch = 0
            if closest_team == self.current_team:
                self.current_player = closest_pid
                self.last_team = closest_team
                self.last_player = closest_pid
                self._opp_candidate_team = self.STATE_NEUTRAL
                self._opp_candidate_frames = 0
            else:
                if closest_team == self._opp_candidate_team:
                    self._opp_candidate_frames += 1
                else:
                    self._opp_candidate_team = closest_team
                    self._opp_candidate_frames = 1

                # Turnover confirmation gate
                if (
                    self._opp_candidate_frames >= self.turnover_confirm_frames
                    or (closest_dist < self.control_radius_px * 0.45 if ball_pos_m is None else closest_dist < 1.2)
                    or self.current_team == self.STATE_NEUTRAL
                ):
                    self.current_team = closest_team
                    self.current_player = closest_pid
                    self.last_team = closest_team
                    self.last_player = closest_pid
                    self._opp_candidate_team = self.STATE_NEUTRAL
                    self._opp_candidate_frames = 0
        else:
            self.frames_since_touch += 1
            if self.frames_since_touch <= self.pass_grace_frames:
                self.current_team = self.last_team
                self.current_player = None
            elif self.frames_since_touch > self.loose_ball_decay_frames:
                self.current_team = self.STATE_NEUTRAL
                self.current_player = None

        return {
            "team_possession": self.current_team,
            "player_possession": self.current_player,
            "has_touch": has_touch,
            "closest_dist": closest_dist if closest_pid is not None else None
        }


def run_threshold_sweep():
    thresholds = [20.0, 30.0, 40.0, 45.0, 50.0, 60.0]
    sweep_results = []

    print("\n" + "=" * 90)
    print("P0-7: EMPIRICAL PARAMETER SWEEP FOR PIXEL-SPACE FALLBACK THRESHOLD (d_px)")
    print("=" * 90)

    # 300-Frame Comprehensive Multi-Scenario Testbench:
    # Segment 1 (Frames 0..99): Normal Dribble by Team 1 (ID 10).
    #            Distance oscillates between 16px and 38px (natural stride variation).
    # Segment 2 (Frames 100..159): Contested Tackle.
    #            Team 1 retains ball at 20px, but Opponent (Team 2, ID 20) hovers at 44px for 20 frames!
    #            A bad threshold (>44px) will falsely flip possession to Team 2!
    # Segment 3 (Frames 160..199): High Pass in flight across field.
    #            Distance to any player is 110px. Hysteresis should preserve Team 1.
    # Segment 4 (Frames 200..299): True Turnover to Team 2 (ID 20).
    #            Team 2 tackles and takes ball to 15px; Team 1 drops back to 80px.

    for d_thresh in thresholds:
        engine = DualModePossessionHysteresisEngine(fps=25.0, control_radius_px=d_thresh)
        correct_frames = 0
        false_acquisitions = 0
        missed_possessions = 0
        state_switches = 0
        prev_team = 0

        for fi in range(300):
            # Ground truth determination
            if fi < 100:
                gt_team = 1
                # Dribble stride: distance oscillates 16..38px
                carrier_d = 16.0 + 22.0 * abs(math.sin(fi * 0.25))
                ball_px = (500.0, 500.0)
                players_px = {
                    10: {"px_x": 500.0 + carrier_d, "px_y": 500.0, "team": 1},
                    20: {"px_x": 650.0, "px_y": 500.0, "team": 2}
                }
            elif 100 <= fi < 160:
                gt_team = 1  # Team 1 holds ball, opponent pressing close
                ball_px = (500.0, 500.0)
                # Carrier at 20px, Opponent at 44px
                # In frames 125..135, carrier is occluded by opponent!
                players_px = {
                    20: {"px_x": 544.0, "px_y": 500.0, "team": 2}
                }
                if not (125 <= fi < 135):
                    players_px[10] = {"px_x": 520.0, "px_y": 500.0, "team": 1}
            elif 160 <= fi < 200:
                gt_team = 1  # Pass in flight (grace period)
                ball_px = (500.0 + (fi - 160) * 10.0, 500.0)
                players_px = {
                    10: {"px_x": 500.0, "px_y": 500.0, "team": 1},
                    11: {"px_x": 900.0, "px_y": 500.0, "team": 1}
                }
            else:
                gt_team = 2  # True turnover to Team 2
                ball_px = (900.0, 500.0)
                players_px = {
                    10: {"px_x": 800.0, "px_y": 500.0, "team": 1},
                    20: {"px_x": 915.0, "px_y": 500.0, "team": 2}
                }

            res = engine.update_frame(fi, ball_pos_px=ball_px, players_px=players_px)
            pred_team = res["team_possession"]

            if pred_team == gt_team:
                correct_frames += 1
            else:
                if pred_team == 0 and gt_team != 0:
                    missed_possessions += 1
                elif pred_team != gt_team and pred_team != 0:
                    false_acquisitions += 1

            if pred_team != prev_team and fi > 0:
                state_switches += 1
            prev_team = pred_team

        accuracy_pct = round((correct_frames / 300.0) * 100.0, 1)
        sweep_results.append({
            "d_px": int(d_thresh),
            "accuracy_pct": accuracy_pct,
            "false_acquisitions": false_acquisitions,
            "missed_possessions": missed_possessions,
            "state_switches": state_switches,
            "verdict": "Optimal" if (false_acquisitions == 0 and accuracy_pct >= 98.0) else "Suboptimal"
        })

    header = f"{'Threshold d_px':<16} | {'Accuracy':<10} | {'False Acq':<11} | {'Missed Poss':<13} | {'Switches':<10} | {'Verdict':<10}"
    print(header)
    print("-" * 90)
    for r in sweep_results:
        row = f"{r['d_px']:>8} px       | {r['accuracy_pct']:>8.1f}% | {r['false_acquisitions']:>9} | {r['missed_possessions']:>11} | {r['state_switches']:>8} | {r['verdict']:<10}"
        print(row)
    print("=" * 90)

    with open(".agents/memory/possession_pixel_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


if __name__ == '__main__':
    run_threshold_sweep()
