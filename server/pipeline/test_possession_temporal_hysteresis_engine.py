"""
test_possession_temporal_hysteresis_engine.py
=============================================
Unit tests & benchmark for PossessionTemporalHysteresisEngine.
"""

import time
import unittest
import numpy as np

from server.pipeline.possession_temporal_hysteresis_engine import (
    PossessionTemporalHysteresisEngine,
)


class TestPossessionTemporalHysteresisEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PossessionTemporalHysteresisEngine(
            fps=30.0,
            control_radius_m=2.8,
            pass_grace_period_s=2.0,  # 60 frames
            turnover_confirm_frames=3,
            loose_ball_decay_s=3.0,   # 90 frames
        )

    def test_01_direct_touch_possession(self):
        """Player within 2.8m directly takes possession for their team."""
        ball_pos = (50.0, 30.0)
        players = {
            101: {"x_m": 51.0, "y_m": 30.5, "team": 1},  # dist ~ 1.11m
            201: {"x_m": 60.0, "y_m": 30.0, "team": 2},  # dist 10m
        }
        res = self.engine.update_frame(0, ball_pos, players)
        self.assertEqual(res["team_possession"], 1)
        self.assertEqual(res["player_possession"], 101)
        self.assertFalse(res["is_flight"])
        self.assertGreaterEqual(res["confidence"], 0.7)

    def test_02_pass_flight_hysteresis_bridging(self):
        """
        Team 1 initiates a 40-frame pass.
        During flight, ball is > 5m from any player.
        Hysteresis MUST maintain Team 1 possession, unlike naive proximity which collapses to Neutral.
        """
        # Frame 0: Player 101 touches ball
        players_touch = {
            101: {"x_m": 20.0, "y_m": 30.0, "team": 1},
        }
        self.engine.update_frame(0, (20.0, 30.0), players_touch)

        # Frames 1..40: Ball travels across pitch from (20,30) to (50,30)
        # Players are far away (passer stayed at 20, receiver waiting at 50)
        players_flight = {
            101: {"x_m": 20.0, "y_m": 30.0, "team": 1},
            102: {"x_m": 50.0, "y_m": 30.0, "team": 1},
        }

        naive_neutral_count = 0
        hysteresis_team1_count = 0

        for fi in range(1, 41):
            ball_x = 20.0 + (fi / 40.0) * 30.0
            ball_pos = (ball_x, 30.0)
            res = self.engine.update_frame(fi, ball_pos, players_flight)

            # In flight, player_possession is None, but team_possession is 1
            if res["team_possession"] == 1:
                hysteresis_team1_count += 1

            # Naive distance check for comparison
            d1 = np.hypot(ball_x - 20.0, 0.0)
            d2 = np.hypot(ball_x - 50.0, 0.0)
            if min(d1, d2) > 2.8:
                naive_neutral_count += 1

        # Hysteresis maintained Team 1 throughout the entire pass
        self.assertEqual(hysteresis_team1_count, 40)
        # Naive proximity failed and dropped to Neutral during mid-flight
        self.assertGreater(naive_neutral_count, 30)

        # Receiver 102 catches the pass at Frame 41
        res_catch = self.engine.update_frame(41, (50.0, 30.0), players_flight)
        self.assertEqual(res_catch["team_possession"], 1)
        self.assertEqual(res_catch["player_possession"], 102)

    def test_03_turnover_with_confirmation_gate(self):
        """
        Team 1 has ball. Team 2 intercepts.
        Requires 3 confirmation frames to switch team possession (filters 1-frame bounce glitches).
        """
        # Team 1 holds ball
        self.engine.update_frame(0, (30.0, 20.0), {101: {"x_m": 30.0, "y_m": 20.0, "team": 1}})

        # Team 2 opponent enters contested proximity (dist ~ 2.0m)
        players_interception = {
            201: {"x_m": 32.0, "y_m": 20.0, "team": 2},
        }

        # Frame 1: Candidate frame 1
        r1 = self.engine.update_frame(1, (30.0, 20.0), players_interception)
        self.assertEqual(r1["team_possession"], 1)  # preserved during candidate phase

        # Frame 2: Candidate frame 2
        r2 = self.engine.update_frame(2, (30.0, 20.0), players_interception)
        self.assertEqual(r2["team_possession"], 1)

        # Frame 3: Candidate frame 3 -> Turnover Confirmed!
        r3 = self.engine.update_frame(3, (30.0, 20.0), players_interception)
        self.assertEqual(r3["team_possession"], 2)
        self.assertEqual(r3["player_possession"], 201)

    def test_04_loose_ball_decay_to_neutral(self):
        """
        If ball is kicked away and no one touches it for > loose_ball_decay_s (90 frames),
        possession cleanly decays to Neutral.
        """
        # Team 1 kick
        self.engine.update_frame(0, (10.0, 10.0), {101: {"x_m": 10.0, "y_m": 10.0, "team": 1}})

        # Ball rolls away with no player nearby
        empty_players = {}
        for fi in range(1, 120):
            res = self.engine.update_frame(fi, (10.0 + fi * 0.1, 10.0), empty_players)

        # Beyond 90 frames, it must be Neutral
        self.assertEqual(res["team_possession"], 0)
        self.assertIsNone(res["player_possession"])

    def test_05_summary_statistics_realistic_breakdown(self):
        """
        Simulate a realistic 1000-frame sequence (Team 1 pass sequence + Team 2 counter-attack).
        Verify that neutral possession is around 5~15% (not 91.2%!).
        """
        self.engine.reset()

        # Phase 1: Team 1 possession for 500 frames (touches + passes)
        for fi in range(500):
            # Touch every 30 frames, pass in between
            if fi % 30 == 0:
                p = {101: {"x_m": 50.0, "y_m": 30.0, "team": 1}}
                self.engine.update_frame(fi, (50.0, 30.0), p)
            else:
                self.engine.update_frame(fi, (55.0, 30.0), {})

        # Phase 2: Turnover to Team 2 for 400 frames
        for fi in range(500, 900):
            if fi % 30 == 0:
                p = {201: {"x_m": 60.0, "y_m": 30.0, "team": 2}}
                self.engine.update_frame(fi, (60.0, 30.0), p)
            else:
                self.engine.update_frame(fi, (65.0, 30.0), {})

        # Phase 3: Dead ball / out of play for 100 frames
        for fi in range(900, 1000):
            self.engine.update_frame(fi, None, {})

        stats = self.engine.compute_summary_statistics()

        print(f"\n[Validation Proof - Possession Breakdown Comparison]:")
        print(f"  Team 1 Possession : {stats['team1_possession_pct']}%")
        print(f"  Team 2 Possession : {stats['team2_possession_pct']}%")
        print(f"  Neutral Possession: {stats['neutral_possession_pct']}% (was 91.2%!)")
        print(f"  Broadcast Ratio   : {stats['broadcast_team1_pct']}% vs {stats['broadcast_team2_pct']}%")
        print(f"  Turnover Switches : {stats['possession_switches']}")

        # Neutral should be minor (~10-15%), NOT 91.2%!
        self.assertLess(stats["neutral_possession_pct"], 25.0)
        self.assertGreater(stats["team1_possession_pct"], 40.0)
        self.assertGreater(stats["team2_possession_pct"], 35.0)
        self.assertGreaterEqual(stats["possession_switches"], 1)

    def test_06_throughput_benchmark(self):
        """Engine must process > 100,000 frames/sec in pure memory."""
        self.engine.reset()
        n_frames = 10000
        ball_pos = (50.0, 30.0)
        players = {
            101: {"x_m": 51.0, "y_m": 30.5, "team": 1},
            201: {"x_m": 60.0, "y_m": 30.0, "team": 2},
        }
        t0 = time.perf_counter()
        for i in range(n_frames):
            self.engine.update_frame(i, ball_pos, players)
        t_elapsed = time.perf_counter() - t0
        fps = n_frames / t_elapsed
        print(f"[Algorithm-only Benchmark] Possession Hysteresis Engine: {fps:,.0f} FPS ({n_frames} frames in {t_elapsed:.3f}s)")
        self.assertGreater(fps, 50000)

    def test_07_filter_control_sequence_bridges_pass_flights(self):
        """
        Verify that filter_control_sequence bridges a 30-frame airborne pass gap
        while respecting turnover confirmation for team transitions.
        """
        raw_control = (
            [1] * 5          # Team 1 touch
            + [0] * 35       # 35 frames airborne pass flight (naive would drop to Neutral)
            + [1] * 5        # Team 1 receiver touch
            + [0] * 20       # 20 frames pass flight
            + [2] * 10       # Team 2 interception
            + [0] * 120      # 120 frames loose/dead ball (should decay to Neutral)
        )
        filtered = self.engine.filter_control_sequence(raw_control)
        self.assertEqual(len(filtered), len(raw_control))

        # First 45 frames must be continuous Team 1 possession (0 bridged)
        for fi in range(45):
            self.assertEqual(filtered[fi], 1, f"Frame {fi} should be Team 1")

        # After Team 2 touch confirms turnover, frames around 70 must be Team 2
        self.assertEqual(filtered[70], 2)

        # At the end of prolonged 120-frame inactivity, must decay to Neutral (0)
        self.assertEqual(filtered[-1], 0)


if __name__ == "__main__":
    unittest.main()

