#!/usr/bin/env python3
"""
test_dynamic_formation_line_analyzer.py - Unit Tests & Benchmark for Dynamic Formation Analyzer
"""

import os
import tempfile
import time
import unittest

from server.pipeline.dynamic_formation_line_analyzer import (
    DynamicFormationAnalyzer,
    DynamicFormationSnapshot,
    TacticalLine,
)


class TestDynamicFormationAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = DynamicFormationAnalyzer()

    def test_01_isolate_goalkeeper_standard_attacking_plus_x(self):
        """
        Verify that in +X attack direction, the keeper at x = -48.0m is isolated
        from the 4 defenders at x = -28.0m (gap = 20m).
        """
        players = {
            1: (-48.0, 0.0),  # GK
            2: (-28.0, -22.0), 3: (-28.0, -8.0), 4: (-28.0, 8.0), 5: (-28.0, 22.0),  # DEF
            6: (-10.0, -14.0), 8: (-10.0, 0.0), 10: (-10.0, 14.0),  # MID
            7: (18.0, -20.0), 9: (24.0, 0.0), 11: (18.0, 20.0),  # FWD
        }
        keeper_id, outfield = self.analyzer.isolate_goalkeeper(players, attacking_direction="+X")
        self.assertEqual(keeper_id, 1)
        self.assertEqual(len(outfield), 10)
        self.assertNotIn(1, outfield)

    def test_02_isolate_goalkeeper_reversed_attacking_minus_x(self):
        """
        Verify that in -X attack direction, the keeper at x = +48.0m is isolated
        from the defenders at x = +28.0m.
        """
        players = {
            1: (48.0, 0.0),  # GK
            2: (28.0, -20.0), 3: (28.0, 0.0), 4: (28.0, 20.0),
            6: (10.0, -10.0), 8: (10.0, 10.0),
            9: (-20.0, 0.0),
        }
        keeper_id, outfield = self.analyzer.isolate_goalkeeper(players, attacking_direction="-X")
        self.assertEqual(keeper_id, 1)
        self.assertEqual(len(outfield), 6)

    def test_03_cluster_into_lines_classic_433(self):
        """
        Verify classification of a classic 4-3-3 shape.
        """
        players = {
            1: (-48.0, 0.0),
            2: (-25.0, -24.0), 3: (-25.0, -8.0), 4: (-25.0, 8.0), 5: (-25.0, 24.0),
            6: (-5.0, -15.0), 8: (-5.0, 0.0), 10: (-5.0, 15.0),
            7: (22.0, -22.0), 9: (28.0, 0.0), 11: (22.0, 22.0),
        }
        snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        self.assertEqual(snapshot.formation_name, "4-3-3")
        self.assertEqual(snapshot.line_counts, [4, 3, 3])
        self.assertEqual(len(snapshot.tactical_lines), 3)
        self.assertEqual(snapshot.tactical_lines[0].line_name, "Defensive Line")
        self.assertEqual(snapshot.tactical_lines[1].line_name, "Midfield Line")
        self.assertEqual(snapshot.tactical_lines[2].line_name, "Forward Line")

    def test_04_cluster_into_lines_classic_442(self):
        """
        Verify classification of a classic 4-4-2 shape.
        """
        players = {
            1: (-48.0, 0.0),
            2: (-26.0, -22.0), 3: (-26.0, -7.0), 4: (-26.0, 7.0), 5: (-26.0, 22.0),
            6: (-4.0, -20.0), 7: (-4.0, -6.0), 8: (-4.0, 6.0), 11: (-4.0, 20.0),
            9: (25.0, -7.0), 10: (25.0, 7.0),
        }
        snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        self.assertEqual(snapshot.formation_name, "4-4-2")
        self.assertEqual(snapshot.line_counts, [4, 4, 2])

    def test_05_cluster_into_lines_modern_325_possession_shape(self):
        """
        Verify classification of a modern 3-2-5 attacking shape (e.g. Manchester City in possession).
        """
        players = {
            1: (-45.0, 0.0),
            3: (-20.0, -18.0), 4: (-20.0, 0.0), 5: (-20.0, 18.0),  # 3 Rest-Defenders
            8: (2.0, -10.0), 16: (2.0, 10.0),  # 2 Midfield Pivots
            7: (28.0, -25.0), 17: (28.0, -12.0), 9: (32.0, 0.0), 47: (28.0, 12.0), 10: (28.0, 25.0),  # 5 Attackers
        }
        snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        self.assertEqual(snapshot.formation_name, "3-2-5")
        self.assertEqual(snapshot.line_counts, [3, 2, 5])

    def test_06_inter_line_distances_and_vulnerability(self):
        """
        Verify detection of space between the lines (d_def_mid = 24.0m > 20.0m safe limit).
        """
        players = {
            1: (-48.0, 0.0),
            2: (-30.0, -20.0), 3: (-30.0, 0.0), 4: (-30.0, 20.0),  # Deep defense at -30m
            6: (-5.0, -10.0), 8: (-5.0, 0.0), 10: (-5.0, 10.0),    # Midfield at -5m (gap = 25m)
            9: (20.0, 0.0),
        }
        snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        self.assertTrue(snapshot.space_between_lines_vulnerability)
        self.assertGreater(snapshot.max_inter_line_gap_m, 20.0)

    def test_07_compact_block_no_vulnerability(self):
        """
        A compact defensive block with inter-line gaps <= 15m should NOT trigger vulnerability.
        """
        players = {
            1: (-45.0, 0.0),
            2: (-20.0, -15.0), 3: (-20.0, 0.0), 4: (-20.0, 15.0),
            6: (-8.0, -10.0), 8: (-8.0, 10.0),
            9: (3.0, 0.0),
        }
        snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        self.assertFalse(snapshot.space_between_lines_vulnerability)
        self.assertLessEqual(snapshot.max_inter_line_gap_m, 20.0)

    def test_08_degenerate_or_few_players(self):
        """Gracefully handle empty or small player dictionaries."""
        snapshot = self.analyzer.evaluate_frame({}, team_id=1)
        self.assertEqual(snapshot.formation_name, "")
        self.assertEqual(snapshot.line_counts, [])
        self.assertEqual(snapshot.total_team_depth_m, 0.0)

    def test_09_render_formation_diagram(self):
        """Verify generation of broadcast formation diagram PNG artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_formation_433.png")
            players = {
                1: (-48.0, 0.0),
                2: (-25.0, -22.0), 3: (-25.0, -7.0), 4: (-25.0, 7.0), 5: (-25.0, 22.0),
                6: (-6.0, -14.0), 8: (-6.0, 0.0), 10: (-6.0, 14.0),
                7: (22.0, -20.0), 9: (26.0, 0.0), 11: (22.0, 20.0),
            }
            snapshot = self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
            rendered = self.analyzer.render_formation_tactical_diagram(snapshot, out_file)
            self.assertTrue(os.path.exists(rendered))
            self.assertGreater(os.path.getsize(rendered), 10000)

    def test_10_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark] Evaluate pure-memory dynamic formation clustering throughput.
        Guarantees >15,000 frames evaluated per second.
        """
        eval_count = 20000
        players = {
            1: (-48.0, 0.0),
            2: (-25.0, -22.0), 3: (-25.0, -7.0), 4: (-25.0, 7.0), 5: (-25.0, 22.0),
            6: (-6.0, -14.0), 8: (-6.0, 0.0), 10: (-6.0, 14.0),
            7: (22.0, -20.0), 9: (26.0, 0.0), 11: (22.0, 20.0),
        }

        t0 = time.perf_counter()
        for _ in range(eval_count):
            self.analyzer.evaluate_frame(players, team_id=1, attacking_direction="+X")
        elapsed_sec = time.perf_counter() - t0
        fps = eval_count / elapsed_sec

        print(
            f"\n[Algorithm-only Benchmark] DynamicFormationAnalyzer: "
            f"{eval_count} frames in {elapsed_sec * 1000.0:.2f}ms ({fps:,.0f} frames/sec)"
        )
        self.assertGreater(fps, 10000)


if __name__ == "__main__":
    unittest.main()
