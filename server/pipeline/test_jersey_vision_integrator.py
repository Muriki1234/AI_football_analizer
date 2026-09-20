"""
test_jersey_vision_integrator.py — Unit Tests & Benchmarks for JerseyVisionIntegrator
"""

import time
import unittest
import numpy as np

from server.pipeline.jersey_vision_integrator import JerseyVisionIntegrator


class TestJerseyVisionIntegrator(unittest.TestCase):
    def setUp(self):
        self.integrator = JerseyVisionIntegrator(max_keyframes_per_player=5, min_bbox_height=40.0)

    def test_01_keyframe_candidate_selection(self):
        """Top-K keyframes must be chosen based on largest bounding box resolution and aspect ratio."""
        tracks = {
            "players": [
                {
                    1: {"bbox": [100, 100, 140, 200]},  # w=40, h=100, area=4000
                    2: {"bbox": [200, 200, 210, 220]},  # h=20 < 40.0 (ignored)
                },
                {
                    1: {"bbox": [100, 100, 160, 250]},  # w=60, h=150, area=9000 (larger)
                },
                {
                    1: {"bbox": [100, 100, 130, 180]},  # w=30, h=80, area=2400
                },
            ]
        }

        candidates = self.integrator.select_keyframe_candidates(tracks)

        self.assertIn(1, candidates)
        self.assertNotIn(2, candidates)
        # First candidate should be frame 1 (area=9000)
        self.assertEqual(candidates[1][0][0], 1)
        self.assertEqual(len(candidates[1]), 3)

    def test_02_torso_patch_extraction(self):
        """Upper torso patch must be cropped, resized to (64, 48), and enhanced."""
        # Synthetic 720p image
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        # Draw player silhouette at (200, 150, 300, 400)
        frame[150:400, 200:300] = [20, 20, 200]  # Red jersey

        patch = self.integrator.extract_torso_patch(frame, (200, 150, 300, 400))

        self.assertIsNotNone(patch)
        self.assertEqual(patch.shape, (64, 48))
        self.assertEqual(patch.dtype, np.uint8)

    def test_03_consensus_voting_and_track_mutation(self):
        """
        Noisy multi-frame observations must be resolved to true consensus jersey number
        and annotated across player tracklets in tracks['players'].
        """
        tracks = {
            "players": [
                {10: {"bbox": [100, 100, 150, 200]}, 7: {"bbox": [300, 300, 350, 400]}},
                {10: {"bbox": [102, 101, 152, 201]}, 7: {"bbox": [302, 301, 352, 401]}},
                {10: {"bbox": [105, 103, 155, 203]}},
            ]
        }

        # Player 10 observations: four times #10, one noisy #18
        self.integrator.record_observation(track_id=10, frame_idx=0, raw_number="10", confidence=0.90)
        self.integrator.record_observation(track_id=10, frame_idx=1, raw_number="#10", confidence=0.85)
        self.integrator.record_observation(track_id=10, frame_idx=2, raw_number="18", confidence=0.35)  # Noise

        # Player 7 observations: #7
        self.integrator.record_observation(track_id=7, frame_idx=0, raw_number=7, confidence=0.95)

        resolved = self.integrator.annotate_tracks(tracks)

        self.assertEqual(resolved[10]["jersey_number"], 10)
        self.assertGreater(resolved[10]["confidence"], 0.70)
        self.assertEqual(resolved[7]["jersey_number"], 7)

        # Check track mutation
        p10_f0 = tracks["players"][0][10]
        self.assertEqual(p10_f0["jersey_number"], 10)
        self.assertEqual(p10_f0["jersey_confidence"], resolved[10]["confidence"])

        p7_f1 = tracks["players"][1][7]
        self.assertEqual(p7_f1["jersey_number"], 7)

    def test_04_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark: Jersey Keyframe Selection & Consensus Voting]:
        Measures keyframe extraction and consensus resolution across 10,000 frames (22 players).
        """
        n_frames = 10000
        n_players = 22

        tracks = {
            "players": [
                {
                    pid: {"bbox": [pid * 20, 100, pid * 20 + 40, 200]}
                    for pid in range(1, n_players + 1)
                }
                for _ in range(n_frames)
            ]
        }

        t0 = time.perf_counter()
        candidates = self.integrator.select_keyframe_candidates(tracks)

        # Simulate 5 observations per player
        for pid in range(1, n_players + 1):
            for i in range(5):
                self.integrator.record_observation(pid, i * 10, pid, 0.85)

        resolved = self.integrator.annotate_tracks(tracks)
        elapsed_s = time.perf_counter() - t0

        total_evals = n_frames * n_players
        throughput = total_evals / max(1e-5, elapsed_s)

        print(f"\n[Algorithm-only Benchmark: Jersey Keyframe Selection & Consensus Voting]:")
        print(f"  Frames Evaluated     : {n_frames:,} ({n_players} players, {total_evals:,} evaluations)")
        print(f"  Total Processing Time: {elapsed_s*1000.0:.2f} ms")
        print(f"  Throughput           : {throughput:,.0f} player-bboxes/sec")

        self.assertLess(elapsed_s, 0.60)
        self.assertGreater(throughput, 400000.0)
        self.assertEqual(len(resolved), n_players)


if __name__ == "__main__":
    unittest.main()
