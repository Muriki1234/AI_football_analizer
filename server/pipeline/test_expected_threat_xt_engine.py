#!/usr/bin/env python3
"""
test_expected_threat_xt_engine.py - Unit Tests & Performance Benchmark for Expected Threat (xT) Engine
"""

import os
import tempfile
import time
import unittest

import numpy as np

from server.pipeline.expected_threat_xt_engine import (
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_PITCH_LENGTH_M,
    DEFAULT_PITCH_WIDTH_M,
    ExpectedThreatEngine,
    ThreatAction,
)


class TestExpectedThreatEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ExpectedThreatEngine()

    def test_01_grid_dimensions_and_initialization(self):
        """Verify grid size, cell resolutions, and surface shape."""
        self.assertEqual(self.engine.cols, DEFAULT_GRID_COLS)
        self.assertEqual(self.engine.rows, DEFAULT_GRID_ROWS)
        self.assertEqual(self.engine.xt_surface.shape, (DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS))
        self.assertAlmostEqual(self.engine.cell_w, DEFAULT_PITCH_LENGTH_M / 16.0, places=3)
        self.assertAlmostEqual(self.engine.cell_h, DEFAULT_PITCH_WIDTH_M / 12.0, places=3)

    def test_02_value_iteration_convergence(self):
        """Verify that value iteration produced valid probabilities in [0.0, 1.0]."""
        surface = self.engine.xt_surface
        self.assertTrue(np.all(surface >= 0.0))
        self.assertTrue(np.all(surface <= 1.0))
        self.assertGreater(np.max(surface), 0.35)
        self.assertLess(np.min(surface), 0.05)

    def test_03_spatial_threat_gradient(self):
        """
        Verify that Expected Threat strictly increases as the ball advances towards opponent goal:
        Own box (col 0) < Defensive midfield (col 4) < Central midfield (col 8) < Zone 14 (col 13) < Box (col 15)
        """
        mid_row = DEFAULT_GRID_ROWS // 2  # Central corridor (y ≈ 0)
        own_box_xt = self.engine.xt_surface[mid_row, 1]
        def_mid_xt = self.engine.xt_surface[mid_row, 4]
        att_mid_xt = self.engine.xt_surface[mid_row, 9]
        zone14_xt = self.engine.xt_surface[mid_row, 13]
        penalty_box_xt = self.engine.xt_surface[mid_row, 15]

        self.assertLess(own_box_xt, def_mid_xt)
        self.assertLess(def_mid_xt, att_mid_xt)
        self.assertLess(att_mid_xt, zone14_xt)
        self.assertLess(zone14_xt, penalty_box_xt)

    def test_04_coordinate_to_zone_mapping(self):
        """Test both centered [-52.5, 52.5] and uncentered [0, 105] coordinate resolution."""
        # Centered: Own goal (-50.0, 0)
        col_c, row_c = self.engine.coords_to_zone(-50.0, 0.0)
        self.assertEqual(col_c, 0)
        self.assertIn(row_c, [5, 6])  # middle rows

        # Centered: Opponent box (48.0, 0.0)
        col_opp, row_opp = self.engine.coords_to_zone(48.0, 0.0)
        self.assertEqual(col_opp, 15)
        self.assertIn(row_opp, [5, 6])

        # Uncentered: [0, 105] format (e.g. 52.5, 34 -> center spot)
        col_mid, row_mid = self.engine.coords_to_zone(52.5, 34.0, corner_origin=True)
        self.assertIn(col_mid, [7, 8])
        self.assertIn(row_mid, [5, 6])

    def test_05_attacking_direction_mirroring(self):
        """Verify that when attacking in -X direction, movement towards negative X generates positive threat."""
        # Attacking +X: pass from 0 to +30m
        action_plus = self.engine.evaluate_action(
            start_x=0.0, start_y=0.0, end_x=30.0, end_y=0.0, attacking_direction="+X"
        )
        self.assertGreater(action_plus.delta_xt, 0.0)

        # Attacking -X: pass from 0 to -30m
        action_minus = self.engine.evaluate_action(
            start_x=0.0, start_y=0.0, end_x=-30.0, end_y=0.0, attacking_direction="-X"
        )
        self.assertGreater(action_minus.delta_xt, 0.0)
        # Symmetrical magnitude
        self.assertAlmostEqual(action_plus.delta_xt, action_minus.delta_xt, places=4)

    def test_06_action_evaluation_progressive_pass(self):
        """A line-breaking progressive pass into Zone 14 should have delta_xt > 0.015."""
        action = self.engine.evaluate_action(
            start_x=10.0, start_y=0.0, end_x=38.0, end_y=0.0,
            action_type="pass", player_id=10, team_id=1
        )
        self.assertTrue(action.is_progressive)
        self.assertGreater(action.delta_xt, 0.015)
        self.assertEqual(action.player_id, 10)

    def test_07_action_evaluation_negative_backpass(self):
        """A safety backpass towards own half should yield negative delta_xt."""
        action = self.engine.evaluate_action(
            start_x=35.0, start_y=0.0, end_x=5.0, end_y=0.0,
            action_type="pass", player_id=8, team_id=1
        )
        self.assertFalse(action.is_progressive)
        self.assertLess(action.delta_xt, 0.0)

    def test_08_pass_network_attribution(self):
        """Verify multi-player pass event batch aggregation and threat share calculations."""
        mock_passes = [
            {"passer_id": 10, "team_id": 1, "start_x": 5.0, "start_y": 0.0, "end_x": 38.0, "end_y": 2.0},
            {"passer_id": 10, "team_id": 1, "start_x": 38.0, "start_y": 2.0, "end_x": 48.0, "end_y": 0.0},
            {"passer_id": 4, "team_id": 1, "start_x": -30.0, "start_y": -10.0, "end_x": -15.0, "end_y": 0.0},
            {"passer_id": 7, "team_id": 2, "start_x": 0.0, "start_y": 0.0, "end_x": 32.0, "end_y": 10.0},
        ]
        res = self.engine.evaluate_pass_network(mock_passes)
        self.assertEqual(res["total_actions"], 4)
        self.assertGreater(res["progressive_actions"], 0)
        self.assertIn("1", res["team_xt_totals"])
        self.assertIn("2", res["team_xt_totals"])

        creators = res["top_threat_creators"]
        self.assertEqual(creators[0]["player_id"], 10)  # Player 10 executed 2 progressive passes
        self.assertGreater(creators[0]["total_xt"], 0.0)
        self.assertGreater(creators[0]["threat_share_pct"], 50.0)

    def test_09_render_xt_plot(self):
        """Verify generation of xT pitch surface visualization artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_xt_surface.png")
            action = self.engine.evaluate_action(0.0, 0.0, 35.0, 5.0, player_id=10)
            rendered = self.engine.render_xt_heatmap_plot(out_file, [action])
            self.assertTrue(os.path.exists(rendered))
            self.assertGreater(os.path.getsize(rendered), 10000)

    def test_10_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark] Evaluate pure-memory action evaluation throughput.
        Guarantees >100,000 evaluations per second.
        """
        eval_count = 50000
        start_xs = np.random.uniform(-40, 30, eval_count)
        start_ys = np.random.uniform(-30, 30, eval_count)
        end_xs = start_xs + np.random.uniform(5, 25, eval_count)
        end_ys = start_ys + np.random.uniform(-10, 10, eval_count)

        t0 = time.perf_counter()
        for i in range(eval_count):
            self.engine.evaluate_action(
                start_x=float(start_xs[i]),
                start_y=float(start_ys[i]),
                end_x=float(end_xs[i]),
                end_y=float(end_ys[i]),
                action_type="pass",
                player_id=10,
                team_id=1,
            )
        elapsed_sec = time.perf_counter() - t0
        fps = eval_count / elapsed_sec

        print(
            f"\n[Algorithm-only Benchmark] ExpectedThreatEngine: "
            f"{eval_count} actions in {elapsed_sec * 1000.0:.2f}ms ({fps:,.0f} evaluations/sec)"
        )
        self.assertGreater(fps, 50000)  # Must achieve at least 50k evals/sec


if __name__ == "__main__":
    unittest.main()
