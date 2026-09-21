"""
test_offside_var_evaluator.py - Comprehensive Unit Tests and Benchmarks for VAR Offside Evaluator
"""

import os
import time
import unittest
import numpy as np

from server.pipeline.offside_var_evaluator import (
    VAROffsideEvaluator,
    OffsideEvaluationResult,
    PlayerOffsideStatus,
)


class TestVAROffsideEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = VAROffsideEvaluator(pitch_length=105.0, pitch_width=68.0, tolerance_meters=0.05)

    def test_01_clear_offside_right_attack(self):
        """Attacker receiver is 1.5m beyond second-last defender in opponent half."""
        ball_xy = (20.0, 5.0)
        # Passer: #10 (team 1) at (18.0, 4.0)
        # Receiver: #9 (team 1) at (36.5, 0.0)
        # Defenders (team 2):
        #   #1 (GK) at (48.0, 0.0) -> Deepest defender
        #   #4 at (35.0, -10.0)    -> Second-last defender (offside line x = 35.0m)
        #   #5 at (32.0, 10.0)
        positions = {
            10: (18.0, 4.0),
            9: (36.5, 0.0),
            1: (48.0, 0.0),
            4: (35.0, -10.0),
            5: (32.0, 10.0),
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2, 5: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=100,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertTrue(res.is_offside)
        self.assertEqual(res.decision, "OFFSIDE")
        self.assertAlmostEqual(res.offside_line_x, 35.0, places=2)
        self.assertAlmostEqual(res.margin_meters, 1.5, places=2)
        self.assertEqual(res.second_last_defender_id, 4)
        self.assertEqual(res.deepest_defender_id, 1)

    def test_02_clear_onside_right_attack(self):
        """Attacker receiver is 1.0m behind second-last defender."""
        ball_xy = (20.0, 5.0)
        positions = {
            10: (18.0, 4.0),
            9: (34.0, 0.0),      # Attacker at 34.0m
            1: (48.0, 0.0),      # Deepest defender at 48.0m
            4: (35.0, -10.0),    # 2nd-last defender at 35.0m (line x = 35.0m)
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=101,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertFalse(res.is_offside)
        self.assertEqual(res.decision, "ONSIDE")
        self.assertAlmostEqual(res.margin_meters, -1.0, places=2)

    def test_03_own_half_exemption(self):
        """Attacker is ahead of defenders, but inside own defensive half (x < 0)."""
        ball_xy = (-15.0, 0.0)
        positions = {
            10: (-20.0, 0.0),
            9: (-5.0, 5.0),      # Receiver at -5.0m (own half!)
            1: (40.0, 0.0),      # GK
            4: (-10.0, 0.0),     # High defensive line at -10.0m
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=102,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertFalse(res.is_offside)
        self.assertEqual(res.decision, "OWN_HALF")
        # Under FIFA rules, offside line in opponent half is at least the halfway line (x=0)
        self.assertGreaterEqual(res.offside_line_x, 0.0)

    def test_04_behind_ball_exemption(self):
        """Attacker is beyond second-last defender, but behind the ball."""
        ball_xy = (40.0, 0.0)   # Cutback pass from 40.0m
        positions = {
            10: (39.5, 0.0),    # Passer
            9: (37.0, 5.0),     # Receiver at 37.0m (behind ball!)
            1: (45.0, 0.0),     # GK
            4: (35.0, 0.0),     # 2nd-last defender at 35.0m
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=103,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertFalse(res.is_offside)
        self.assertEqual(res.decision, "BEHIND_BALL")

    def test_05_left_attack_inversion(self):
        """Team attacks toward left goal (-X direction, goal at -52.5m)."""
        ball_xy = (-20.0, 5.0)
        positions = {
            10: (-18.0, 4.0),
            9: (-38.0, 0.0),    # Receiver at -38.0m (closer to -52.5m than -35.0m)
            1: (-48.0, 0.0),    # GK (deepest, min x = -48.0m)
            4: (-35.0, 0.0),    # 2nd-last defender (line at -35.0m)
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=104,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=-1,
        )

        self.assertTrue(res.is_offside)
        self.assertEqual(res.decision, "OFFSIDE")
        self.assertAlmostEqual(res.offside_line_x, -35.0, places=2)
        self.assertAlmostEqual(res.margin_meters, 3.0, places=2)
        self.assertEqual(res.second_last_defender_id, 4)
        self.assertEqual(res.deepest_defender_id, 1)

    def test_06_tolerance_margin_level_decision(self):
        """Attacker is 0.03m ahead of line (within 0.05m tolerance) -> ONSIDE."""
        ball_xy = (20.0, 0.0)
        positions = {
            10: (18.0, 0.0),
            9: (35.03, 0.0),    # 3cm ahead of 35.0m line
            1: (48.0, 0.0),
            4: (35.0, 0.0),
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=105,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertFalse(res.is_offside)
        self.assertEqual(res.decision, "ONSIDE")
        self.assertAlmostEqual(res.margin_meters, 0.03, places=2)

    def test_07_goalkeeper_ahead_swap(self):
        """Goalkeeper is caught out of position (x=10.0m) while outfield defender covers goal line."""
        ball_xy = (25.0, 0.0)
        positions = {
            10: (20.0, 0.0),
            9: (36.0, 0.0),
            1: (10.0, 0.0),     # GK at 10.0m (ahead of defenders!)
            4: (45.0, 0.0),     # Last outfield defender on goal line (deepest)
            5: (35.0, 0.0),     # Second outfield defender (second-last defender at 35.0m)
        }
        teams = {10: 1, 9: 1, 1: 2, 4: 2, 5: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=106,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        self.assertTrue(res.is_offside)
        self.assertEqual(res.deepest_defender_id, 4)
        self.assertEqual(res.second_last_defender_id, 5)
        self.assertAlmostEqual(res.offside_line_x, 35.0, places=2)
        self.assertAlmostEqual(res.margin_meters, 1.0, places=2)

    def test_08_render_var_pitch_graphic(self):
        """Renders 2D VAR pitch visualization without error and saves test artifact."""
        ball_xy = (20.0, 5.0)
        positions = {
            10: (18.0, 4.0),
            9: (36.5, 0.0),
            11: (30.0, 15.0),
            1: (48.0, 0.0),
            4: (35.0, -10.0),
            5: (32.0, 10.0),
        }
        teams = {10: 1, 9: 1, 11: 1, 1: 2, 4: 2, 5: 2}

        res = self.evaluator.evaluate_pass_release(
            frame_idx=250,
            passer_id=10,
            passer_team=1,
            ball_xy=ball_xy,
            player_positions=positions,
            player_teams=teams,
            receiver_id=9,
            attacking_direction=1,
        )

        out_path = "/tmp/test_var_offside_pitch.png"
        self.evaluator.render_var_freeze_frame(res, output_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)
        if os.path.exists(out_path):
            os.remove(out_path)

    def test_09_algorithm_only_benchmark(self):
        """Evaluates 10,000 kick-point offside evaluations in memory."""
        positions = {
            10: (18.0, 4.0),
            9: (36.5, 0.0),
            11: (30.0, 15.0),
            1: (48.0, 0.0),
            4: (35.0, -10.0),
            5: (32.0, 10.0),
        }
        teams = {10: 1, 9: 1, 11: 1, 1: 2, 4: 2, 5: 2}
        ball_xy = (20.0, 5.0)

        n_evals = 10000
        t0 = time.perf_counter()
        for i in range(n_evals):
            self.evaluator.evaluate_pass_release(
                frame_idx=i,
                passer_id=10,
                passer_team=1,
                ball_xy=ball_xy,
                player_positions=positions,
                player_teams=teams,
                receiver_id=9,
                attacking_direction=1,
            )
        elapsed = time.perf_counter() - t0
        fps = n_evals / elapsed
        print(f"\n[Algorithm-only Benchmark] VAROffsideEvaluator: {fps:,.0f} evaluations/sec ({elapsed*1000:.2f} ms for {n_evals} evals)")
        self.assertGreater(fps, 10000.0)


if __name__ == "__main__":
    unittest.main()
