"""
test_pass_event_detector.py — Unit Tests & Algorithm-Only Benchmarks for PassEventDetector
"""

import time
import unittest
from server.pipeline.pass_event_detector import PassEventDetector, PassEvent


class TestPassEventDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PassEventDetector(fps=25.0, control_radius_m=2.5, min_pass_distance_m=3.5)
        self.teams = {
            1: 1, 2: 1, 3: 1, 4: 1,  # Team 1 (attacking right +1)
            8: 2, 9: 2, 10: 2,        # Team 2 (attacking left -1)
        }

    def test_01_successful_forward_pass(self):
        """Player 1 controls ball, releases, and Player 2 receives it 18m away."""
        ball_traj = {}
        player_traj = {}

        # Frames 0-4: Player 1 controls ball at (-10.0, 0.0)
        for f in range(5):
            ball_traj[f] = (-10.0, 0.0)
            player_traj[f] = {1: (-10.0, 0.0), 2: (8.0, 2.0)}

        # Frames 5-11: Ball flies from (-10.0, 0.0) to (8.0, 2.0)
        for f in range(5, 12):
            alpha = (f - 4) / 7.0
            bx = -10.0 + alpha * (8.0 - (-10.0))
            by = 0.0 + alpha * 2.0
            ball_traj[f] = (bx, by)
            player_traj[f] = {1: (-10.0, 0.0), 2: (8.0, 2.0)}

        # Frames 12-15: Player 2 receives and controls ball at (8.0, 2.0)
        for f in range(12, 16):
            ball_traj[f] = (8.0, 2.0)
            player_traj[f] = {1: (-10.0, 0.0), 2: (8.0, 2.0)}

        passes = self.detector.detect_passes(ball_traj, player_traj, self.teams)
        self.assertEqual(len(passes), 1)
        p = passes[0]
        self.assertEqual(p.passer_id, 1)
        self.assertEqual(p.receiver_id, 2)
        self.assertEqual(p.passer_team, 1)
        self.assertEqual(p.receiver_team, 1)
        self.assertEqual(p.outcome, "completed")
        self.assertAlmostEqual(p.pass_distance, 18.11, delta=0.5)
        self.assertTrue(p.is_progressive)  # dx = 18m > 9.15m

    def test_02_intercepted_turnover_pass(self):
        """Player 1 passes, but Player 9 of Team 2 intercepts it."""
        ball_traj = {}
        player_traj = {}

        for f in range(5):
            ball_traj[f] = (0.0, 0.0)
            player_traj[f] = {1: (0.0, 0.0), 9: (15.0, 0.0)}

        for f in range(5, 12):
            alpha = (f - 4) / 7.0
            ball_traj[f] = (alpha * 15.0, 0.0)
            player_traj[f] = {1: (0.0, 0.0), 9: (15.0, 0.0)}

        for f in range(12, 16):
            ball_traj[f] = (15.0, 0.0)
            player_traj[f] = {1: (0.0, 0.0), 9: (15.0, 0.0)}

        passes = self.detector.detect_passes(ball_traj, player_traj, self.teams)
        self.assertEqual(len(passes), 1)
        p = passes[0]
        self.assertEqual(p.passer_id, 1)
        self.assertEqual(p.receiver_id, 9)
        self.assertEqual(p.outcome, "intercepted")
        self.assertFalse(p.is_progressive)

    def test_03_out_of_bounds_pass(self):
        """Pass over the sideline (> 34m width) is classified as incomplete_out_of_bounds."""
        ball_traj = {}
        player_traj = {}

        for f in range(5):
            ball_traj[f] = (0.0, 20.0)
            player_traj[f] = {1: (0.0, 20.0)}

        for f in range(5, 12):
            alpha = (f - 4) / 7.0
            ball_traj[f] = (alpha * 5.0, 20.0 + alpha * 18.0)  # y reaches 38.0m (out of bounds)
            player_traj[f] = {1: (0.0, 20.0)}

        passes = self.detector.detect_passes(ball_traj, player_traj, self.teams)
        self.assertEqual(len(passes), 1)
        p = passes[0]
        self.assertEqual(p.outcome, "incomplete_out_of_bounds")
        self.assertIsNone(p.receiver_id)

    def test_04_short_dribble_touches_ignored(self):
        """Micro-touches (< min_pass_distance_m) are treated as dribbling, not passes."""
        ball_traj = {}
        player_traj = {}

        for f in range(10):
            ball_traj[f] = (f * 0.1, 0.0)  # Total displacement = 0.9m
            player_traj[f] = {1: (f * 0.1, 0.0)}

        passes = self.detector.detect_passes(ball_traj, player_traj, self.teams)
        self.assertEqual(len(passes), 0)

    def test_05_zone14_and_box_entry_detection(self):
        """Verifies tactical entry into Zone 14 and the opponent penalty box."""
        # Zone 14 entry: Pass from (10.0, 0.0) -> (25.0, 2.0)
        ball_z14 = {}
        players_z14 = {}
        for f in range(4):
            ball_z14[f] = (10.0, 0.0)
            players_z14[f] = {1: (10.0, 0.0), 3: (25.0, 2.0)}
        for f in range(4, 10):
            a = (f - 3) / 6.0
            ball_z14[f] = (10.0 + a * 15.0, a * 2.0)
            players_z14[f] = {1: (10.0, 0.0), 3: (25.0, 2.0)}
        for f in range(10, 14):
            ball_z14[f] = (25.0, 2.0)
            players_z14[f] = {1: (10.0, 0.0), 3: (25.0, 2.0)}

        passes_z14 = self.detector.detect_passes(ball_z14, players_z14, self.teams)
        self.assertEqual(len(passes_z14), 1)
        self.assertTrue(passes_z14[0].is_zone14_entry)
        self.assertFalse(passes_z14[0].is_box_entry)

        # Box entry: Pass from (25.0, 2.0) -> (42.0, 1.0)
        ball_box = {}
        players_box = {}
        for f in range(4):
            ball_box[f] = (25.0, 2.0)
            players_box[f] = {3: (25.0, 2.0), 4: (42.0, 1.0)}
        for f in range(4, 10):
            a = (f - 3) / 6.0
            ball_box[f] = (25.0 + a * 17.0, 2.0 - a * 1.0)
            players_box[f] = {3: (25.0, 2.0), 4: (42.0, 1.0)}
        for f in range(10, 14):
            ball_box[f] = (42.0, 1.0)
            players_box[f] = {3: (25.0, 2.0), 4: (42.0, 1.0)}

        passes_box = self.detector.detect_passes(ball_box, players_box, self.teams)
        self.assertEqual(len(passes_box), 1)
        self.assertTrue(passes_box[0].is_box_entry)

    def test_06_pass_network_graph_construction(self):
        """Builds directed passing graph with nodes (player centroids) and edges (pass weights)."""
        passes = [
            PassEvent(1, 1, 1, 2, 1, 0, 10, (-10.0, 0.0), (5.0, 5.0), "completed", 15.8, 0.4, 39.5, True, False, False),
            PassEvent(2, 1, 1, 2, 1, 20, 30, (-8.0, 2.0), (6.0, 4.0), "completed", 14.1, 0.4, 35.3, True, False, False),
            PassEvent(3, 2, 1, 3, 1, 40, 50, (6.0, 4.0), (25.0, 0.0), "completed", 19.4, 0.4, 48.5, True, True, False),
            PassEvent(4, 3, 1, 9, 2, 60, 70, (25.0, 0.0), (28.0, -5.0), "intercepted", 5.8, 0.4, 14.5, False, False, False),
        ]
        net = self.detector.build_pass_network(passes, team_id=1)

        self.assertEqual(net["team_id"], 1)
        self.assertEqual(net["total_passes"], 4)
        self.assertEqual(net["completed_passes"], 3)
        self.assertEqual(net["completion_rate_pct"], 75.0)
        self.assertEqual(net["progressive_passes"], 3)
        self.assertEqual(net["zone14_entries"], 1)

        # Verify edge counts: 1 -> 2 has count 2, 2 -> 3 has count 1
        edges = { (e["source"], e["target"]): e["count"] for e in net["edges"] }
        self.assertEqual(edges[(1, 2)], 2)
        self.assertEqual(edges[(2, 3)], 1)

        # Verify nodes exist
        node_ids = {n["player_id"] for n in net["nodes"]}
        self.assertIn(1, node_ids)
        self.assertIn(2, node_ids)
        self.assertIn(3, node_ids)

    def test_07_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark: In-Memory Kinematic Vector Analysis]:
        Evaluates pass spotting throughput over 10,000 frames with 22 players.
        Measures pure algorithmic state-machine latency (excluding video decode and vision inference).
        """
        n_frames = 10000
        ball_traj = {}
        player_traj = {}
        teams = {i: (1 if i <= 11 else 2) for i in range(1, 23)}

        # Simulate 10k frames: passes every 50 frames
        for f in range(n_frames):
            pass_cycle = f % 50
            passer = (f // 50) % 11 + 1
            receiver = ((f // 50) + 1) % 11 + 1
            if pass_cycle < 10:
                ball_traj[f] = (float(passer * 4 - 25), 0.0)
            elif pass_cycle < 40:
                a = (pass_cycle - 9) / 31.0
                x = (passer * 4 - 25) + a * ((receiver * 4 - 25) - (passer * 4 - 25))
                ball_traj[f] = (x, 0.0)
            else:
                ball_traj[f] = (float(receiver * 4 - 25), 0.0)

            frame_players = {
                pid: (float(pid * 4 - 25), -10.0 if pid % 2 == 0 else 10.0)
                for pid in range(1, 23)
            }
            # Keep current controller near ball
            if pass_cycle < 10:
                frame_players[passer] = ball_traj[f]
            elif pass_cycle >= 40:
                frame_players[receiver] = ball_traj[f]
            player_traj[f] = frame_players

        t0 = time.perf_counter()
        passes = self.detector.detect_passes(ball_traj, player_traj, teams)
        t_detect = time.perf_counter() - t0

        fps = n_frames / max(1e-5, t_detect)
        us_per_frame = (t_detect / n_frames) * 1e6

        print(f"\n[Algorithm-only Benchmark: In-Memory Kinematic Vector Analysis] ({n_frames} frames, 22 players):")
        print(f"  Total processing time : {t_detect*1000:.2f}ms")
        print(f"  Throughput            : {fps:,.0f} FPS ({us_per_frame:.2f} μs/frame)")
        print(f"  Detected passes count : {len(passes)}")

        self.assertGreater(len(passes), 100)
        self.assertGreater(fps, 30000.0, "Throughput must exceed 30,000 FPS for in-memory kinematic analysis")


if __name__ == "__main__":
    unittest.main()
