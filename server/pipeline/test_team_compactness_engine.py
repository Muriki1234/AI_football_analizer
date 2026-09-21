"""
test_team_compactness_engine.py - Comprehensive Unit Tests and Benchmarks for Team Compactness Engine
"""

import os
import time
import unittest
import numpy as np

from server.pipeline.team_compactness_engine import (
    TeamCompactnessEngine,
    TeamFrameMetrics,
    FrameCompactnessSnapshot,
)


class TestTeamCompactnessEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TeamCompactnessEngine(fps=25.0, exclude_goalkeeper=True)

    def test_01_compact_defensive_block(self):
        """Outfield players arranged in a tight 20m x 20m defensive square (400 m^2)."""
        # 8 outfield players at the vertices and edges of a 20m x 20m square centered at (-20, 0)
        # Coordinates: x in [-30, -10], y in [-10, 10]
        # Area should be 20 * 20 = 400 m^2
        positions = {
            1: (-48.0, 0.0),    # GK (should be excluded)
            2: (-30.0, -10.0),
            3: (-30.0, 10.0),
            4: (-10.0, -10.0),
            5: (-10.0, 10.0),
            6: (-20.0, -10.0),
            7: (-20.0, 10.0),
            8: (-30.0, 0.0),
            9: (-10.0, 0.0),
        }
        teams = {pid: 1 for pid in positions}

        metrics = self.engine.compute_team_shape(100, 1, positions, teams)
        self.assertAlmostEqual(metrics.hull_area_m2, 400.0, delta=1.0)
        self.assertAlmostEqual(metrics.centroid_xy[0], -20.0, delta=0.5)
        self.assertAlmostEqual(metrics.centroid_xy[1], 0.0, delta=0.5)
        self.assertAlmostEqual(metrics.length_m, 20.0, delta=0.5)
        self.assertAlmostEqual(metrics.width_m, 20.0, delta=0.5)
        self.assertGreater(metrics.stretch_index_m, 5.0)

    def test_02_expanded_attacking_shape(self):
        """Team expands into 50m x 40m attacking shape (area ~2000 m^2)."""
        positions = {
            1: (-40.0, 0.0),    # GK
            2: (0.0, -20.0),
            3: (0.0, 20.0),
            4: (50.0, -20.0),
            5: (50.0, 20.0),
            6: (25.0, 0.0),
        }
        teams = {pid: 1 for pid in positions}

        metrics = self.engine.compute_team_shape(101, 1, positions, teams)
        # Expected area = 50 * 40 = 2000 m^2
        self.assertAlmostEqual(metrics.hull_area_m2, 2000.0, delta=5.0)
        self.assertGreater(metrics.stretch_index_m, 15.0)

    def test_03_goalkeeper_exclusion(self):
        """Verify goalkeeper is excluded when exclude_goalkeeper=True."""
        # 6 players: GK far back at -50m, 5 outfield players clustered near 0m
        positions = {
            1: (-50.0, 0.0),    # Deep GK
            2: (0.0, -5.0),
            3: (0.0, 5.0),
            4: (10.0, -5.0),
            5: (10.0, 5.0),
            6: (5.0, 0.0),
        }
        teams = {pid: 1 for pid in positions}

        metrics_with_gk_excl = self.engine.compute_team_shape(102, 1, positions, teams)
        # With GK excluded: length is 10.0m (from 0 to 10), area is 10 * 10 = 100 m^2
        self.assertAlmostEqual(metrics_with_gk_excl.length_m, 10.0, delta=0.5)
        self.assertAlmostEqual(metrics_with_gk_excl.hull_area_m2, 100.0, delta=2.0)

        # Disable GK exclusion
        engine_raw = TeamCompactnessEngine(fps=25.0, exclude_goalkeeper=False)
        metrics_raw = engine_raw.compute_team_shape(102, 1, positions, teams)
        # With GK included: length is 60.0m (from -50 to 10)
        self.assertAlmostEqual(metrics_raw.length_m, 60.0, delta=0.5)
        self.assertGreater(metrics_raw.hull_area_m2, 250.0)

    def test_04_graham_scan_fallback_consistency(self):
        """Verify pure-Python Graham scan produces identical area and vertices."""
        pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]
        area, perimeter, hull = self.engine._graham_scan_convex_hull(pts)
        self.assertAlmostEqual(area, 100.0, places=2)
        self.assertAlmostEqual(perimeter, 40.0, places=2)
        self.assertEqual(len(hull), 4)

    def test_05_inter_centroid_distance(self):
        """Verify inter-team centroid distance and snapshot calculation."""
        p_traj = {
            100: {
                # Team 1 around (-10, 0)
                1: (-10.0, -5.0), 2: (-10.0, 5.0), 3: (-5.0, 0.0),
                # Team 2 around (20, 0)
                10: (20.0, -5.0), 11: (20.0, 5.0), 12: (15.0, 0.0),
            }
        }
        teams = {1: 1, 2: 1, 3: 1, 10: 2, 11: 2, 12: 2}
        snapshots = self.engine.analyze_match_timeline(p_traj, teams)

        self.assertEqual(len(snapshots), 1)
        snap = snapshots[0]
        # Distance between (-8.33, 0) and (18.33, 0) ~ 26.67m
        self.assertAlmostEqual(snap.inter_centroid_dist_m, 26.67, delta=1.0)

    def test_06_match_timeline_and_summary(self):
        """Processes 50 frames with alternating possession phases and verifies summary KPIs."""
        p_traj = {}
        poss_map = {}
        for f in range(50):
            poss = 1 if f < 25 else 2
            poss_map[f] = poss
            # Team 1 expands during possession, compresses during defense
            t1_scale = 1.5 if poss == 1 else 0.8
            t2_scale = 1.5 if poss == 2 else 0.8
            p_traj[f] = {
                1: (-20.0 * t1_scale, -10.0 * t1_scale),
                2: (-20.0 * t1_scale, 10.0 * t1_scale),
                3: (0.0 * t1_scale, 0.0),
                10: (20.0 * t2_scale, -10.0 * t2_scale),
                11: (20.0 * t2_scale, 10.0 * t2_scale),
                12: (0.0 * t2_scale, 0.0),
            }
        teams = {1: 1, 2: 1, 3: 1, 10: 2, 11: 2, 12: 2}

        snapshots = self.engine.analyze_match_timeline(p_traj, teams, ball_possession_timeline=poss_map)
        summary = self.engine.generate_compactness_summary(snapshots)

        self.assertEqual(summary["total_frames_analyzed"], 50)
        self.assertGreater(summary["team1"]["possession_phase_area_m2"], summary["team1"]["defensive_phase_area_m2"])
        self.assertGreater(summary["team1"]["tactical_expansion_ratio"], 1.2)

    def test_07_render_dashboard_artifact(self):
        """Renders dual-panel compactness dashboard to PNG without errors."""
        p_traj = {
            f: {
                1: (-25.0, -15.0), 2: (-25.0, 15.0), 3: (-5.0, -10.0), 4: (-5.0, 10.0), 5: (-45.0, 0.0),
                10: (25.0, -15.0), 11: (25.0, 15.0), 12: (5.0, -10.0), 13: (5.0, 10.0), 14: (45.0, 0.0),
            }
            for f in range(20)
        }
        teams = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2}
        snapshots = self.engine.analyze_match_timeline(p_traj, teams)

        out_path = "/tmp/test_compactness_dashboard.png"
        self.engine.render_compactness_dashboard(snapshots, sample_frame_idx=10, output_path=out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)
        if os.path.exists(out_path):
            os.remove(out_path)

    def test_08_algorithm_only_benchmark(self):
        """Evaluates 10,000 frames of 22-player 2D convex hull & stretch index calculation."""
        p_dict = {
            i: (-20.0 + (i * 2.0), -15.0 + (i * 1.5)) for i in range(1, 12)
        }
        p_dict.update({
            i: (20.0 - ((i - 11) * 2.0), -15.0 + ((i - 11) * 1.5)) for i in range(12, 23)
        })
        teams = {i: 1 if i <= 11 else 2 for i in range(1, 23)}

        n_frames = 10000
        t0 = time.perf_counter()
        for i in range(n_frames):
            self.engine.compute_team_shape(i, 1, p_dict, teams)
            self.engine.compute_team_shape(i, 2, p_dict, teams)
        elapsed = time.perf_counter() - t0
        fps = n_frames / elapsed
        print(f"\n[Algorithm-only Benchmark] TeamCompactnessEngine: {fps:,.0f} frames/sec ({elapsed*1000:.2f} ms for {n_frames} frames)")
        self.assertGreater(fps, 5000.0)


if __name__ == "__main__":
    unittest.main()
