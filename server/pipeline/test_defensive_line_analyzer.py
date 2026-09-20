"""
test_defensive_line_analyzer.py - Unit Test & Benchmark Suite
for DefensiveLineAnalyzer.

Verifies:
1. Bidirectional attack resolution (left vs right attack).
2. Deepest defender selection relative to target goal.
3. Sustained line-break penetration event spotting.
4. Transient noise rejection.
5. High-throughput algorithm microbenchmark (> 100,000 frames/sec).
"""

from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from server.pipeline.defensive_line_analyzer import DefensiveLineAnalyzer


class TestDefensiveLineAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DefensiveLineAnalyzer(defender_count=4, min_consecutive_frames=3, fps=25.0)

    def test_attack_direction_resolution(self):
        # Case 1: Team 1 has mean x = 35.0 (left), Team 2 has mean x = 70.0 (right)
        # Team 1 attacking direction must be 'right' (+x)
        tracks_right = {
            "players": [
                {
                    1: {"team": 1, "position_minimap": [30.0, 30.0]},
                    2: {"team": 1, "position_minimap": [40.0, 40.0]},
                    3: {"team": 2, "position_minimap": [65.0, 30.0]},
                    4: {"team": 2, "position_minimap": [75.0, 40.0]},
                }
            ]
        }
        direction_right = self.analyzer.resolve_attack_direction(tracks_right, tracked_team=1, opponent_team=2)
        self.assertEqual(direction_right, "right")

        # Case 2: Team 1 has mean x = 75.0 (right), Team 2 has mean x = 30.0 (left)
        # Team 1 attacking direction must be 'left' (-x)
        tracks_left = {
            "players": [
                {
                    1: {"team": 1, "position_minimap": [70.0, 30.0]},
                    2: {"team": 1, "position_minimap": [80.0, 40.0]},
                    3: {"team": 2, "position_minimap": [25.0, 30.0]},
                    4: {"team": 2, "position_minimap": [35.0, 40.0]},
                }
            ]
        }
        direction_left = self.analyzer.resolve_attack_direction(tracks_left, tracked_team=1, opponent_team=2)
        self.assertEqual(direction_left, "left")

    def test_defensive_line_deepest_defenders_selection(self):
        # Opponent defenders at x = [10, 15, 20, 25, 60, 70, 80, 85, 90, 95]
        opp_xs = [10.0, 15.0, 20.0, 25.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0]
        players_dict = {}
        for idx, x in enumerate(opp_xs):
            players_dict[idx + 1] = {
                "team": 2,
                "position_minimap": [x, 34.0],
                "bbox": [0, 0, 10, 10],
            }
        # Add tracked player (team 1)
        players_dict[99] = {
            "team": 1,
            "position_minimap": [50.0, 34.0],
            "bbox": [100, 100, 20, 40],
        }

        tracks = {"players": [players_dict]}
        tracked_bboxes = {0: (100, 100, 20, 40)}

        # Attacking RIGHT: Defense is deepest defenders near x=105 -> [80, 85, 90, 95] -> mean 87.5
        series_right = self.analyzer.compute_defensive_line_series(
            tracks, tracked_bboxes, tracked_team=1, opponent_team=2, attacking_direction="right"
        )
        self.assertEqual(len(series_right), 1)
        tx, dx, tp = series_right[0]
        self.assertEqual(tx, 50.0)
        self.assertAlmostEqual(dx, 87.5, places=2)

        # Attacking LEFT: Defense is deepest defenders near x=0 -> [10, 15, 20, 25] -> mean 17.5
        series_left = self.analyzer.compute_defensive_line_series(
            tracks, tracked_bboxes, tracked_team=1, opponent_team=2, attacking_direction="left"
        )
        self.assertEqual(len(series_left), 1)
        tx_l, dx_l, tp_l = series_left[0]
        self.assertEqual(tx_l, 50.0)
        self.assertAlmostEqual(dx_l, 17.5, places=2)

    def test_penetration_events_right_attack(self):
        # Attacking right: line is at x = 70.0. Attacker moves: 65, 68, 72, 73, 74, 69
        # Frames 2, 3, 4 are > 70.0 (sustained for 3 frames) -> 1 verified penetration event!
        frame_data = [
            (65.0, 70.0, (65.0, 30.0)),
            (68.0, 70.0, (68.0, 30.0)),
            (72.0, 70.0, (72.0, 30.0)), # beyond 1
            (73.0, 70.0, (73.0, 30.0)), # beyond 2
            (74.0, 70.0, (74.0, 30.0)), # beyond 3 -> TRIGGER EVENT
            (69.0, 70.0, (69.0, 30.0)), # retreats behind
        ]

        result = self.analyzer.detect_penetration_events(
            frame_data, attacking_direction="right", tracked_team=1, opponent_team=2
        )
        self.assertEqual(result["penetration_count"], 1)
        self.assertEqual(result["attacking_direction"], "right")
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["frame_idx"], 2)
        self.assertAlmostEqual(event["time_sec"], 2 / 25.0, places=2)
        self.assertEqual(event["position"], [72.0, 30.0])
        self.assertEqual(result["max_penetration_depth_m"], 4.0)  # max(|74 - 70|)

    def test_penetration_events_left_attack(self):
        # Attacking left: line is at x = 30.0. Attacker moves: 35, 32, 28, 27, 26, 31
        # Frames 2, 3, 4 are < 30.0 (sustained for 3 frames forward) -> 1 verified penetration event!
        frame_data = [
            (35.0, 30.0, (35.0, 30.0)),
            (32.0, 30.0, (32.0, 30.0)),
            (28.0, 30.0, (28.0, 30.0)), # beyond 1
            (27.0, 30.0, (27.0, 30.0)), # beyond 2
            (26.0, 30.0, (26.0, 30.0)), # beyond 3 -> TRIGGER EVENT
            (31.0, 30.0, (31.0, 30.0)), # retreats behind
        ]

        result = self.analyzer.detect_penetration_events(
            frame_data, attacking_direction="left", tracked_team=1, opponent_team=2
        )
        self.assertEqual(result["penetration_count"], 1)
        self.assertEqual(result["attacking_direction"], "left")
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["frame_idx"], 2)
        self.assertEqual(event["position"], [28.0, 30.0])
        self.assertEqual(result["max_penetration_depth_m"], 4.0)  # max(|26 - 30|)

    def test_transient_noise_rejection(self):
        # Attacker flickers beyond line for only 2 frames (consecutive < 3)
        frame_data = [
            (68.0, 70.0, (68.0, 30.0)),
            (71.0, 70.0, (71.0, 30.0)), # 1 frame
            (72.0, 70.0, (72.0, 30.0)), # 2 frames
            (68.0, 70.0, (68.0, 30.0)), # back behind -> rejected
        ]
        result = self.analyzer.detect_penetration_events(
            frame_data, attacking_direction="right", tracked_team=1, opponent_team=2
        )
        self.assertEqual(result["penetration_count"], 0)
        self.assertEqual(len(result["events"]), 0)

    def test_render_visualization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "defensive_line.png"
            frame_data = [
                (50.0, 75.0, (50.0, 30.0)),
                (65.0, 75.0, (65.0, 32.0)),
                (80.0, 75.0, (80.0, 34.0)),
                (82.0, 75.0, (82.0, 35.0)),
                (83.0, 75.0, (83.0, 35.0)),
            ]
            result = self.analyzer.detect_penetration_events(
                frame_data, attacking_direction="right", tracked_team=1, opponent_team=2
            )
            self.analyzer.render_visualization(frame_data, result, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 5000)

    def test_defensive_line_throughput_benchmark(self):
        n_frames = 20000
        dx = 70.0
        # Generate alternating penetration simulation
        frame_data = []
        for i in range(n_frames):
            phase = (i // 10) % 2
            tx = (dx + 5.0) if phase == 1 else (dx - 5.0)
            frame_data.append((tx, dx, (tx, 34.0)))

        t0 = time.perf_counter()
        result = self.analyzer.detect_penetration_events(
            frame_data, attacking_direction="right", tracked_team=1, opponent_team=2
        )
        elapsed = time.perf_counter() - t0
        fps = n_frames / max(elapsed, 1e-6)

        print(f"\n[Algorithm-only Benchmark] DefensiveLineAnalyzer: {n_frames} frames in {elapsed*1000:.1f}ms ({fps:,.0f} frames/sec)")
        self.assertGreater(fps, 100000.0)
        self.assertGreater(result["penetration_count"], 500)


if __name__ == "__main__":
    unittest.main()
