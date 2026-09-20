"""
test_camera_motion_ransac.py — Unit Tests & Benchmarks for RobustCameraMovementEstimator
"""

import time
import unittest
import numpy as np

from server.pipeline.camera_motion_ransac import (
    estimate_motion_ransac,
    RobustCameraMovementEstimator,
)


class TestCameraMotionRansac(unittest.TestCase):
    def test_01_pure_camera_translation_ransac(self):
        """Uniform camera pan of (dx=10.0, dy=-4.0) must be recovered with high accuracy."""
        np.random.seed(42)
        n_pts = 60
        old_pts = np.random.uniform(100.0, 1000.0, (n_pts, 2))
        true_cx, true_cy = 10.0, -4.0
        # old - new = true_c => new = old - true_c
        new_pts = old_pts - np.array([true_cx, true_cy])

        cx, cy, inlier_ratio, inlier_cnt = estimate_motion_ransac(old_pts, new_pts)

        self.assertAlmostEqual(cx, true_cx, delta=0.2)
        self.assertAlmostEqual(cy, true_cy, delta=0.2)
        self.assertGreaterEqual(inlier_ratio, 0.95)
        self.assertEqual(inlier_cnt, n_pts)

    def test_02_moving_player_outlier_rejection_proof(self):
        """
        Validation Proof: 80 static pitch points move 2.0px, while 20 running player
        points move 35.0px.
        Naive max-displacement reports ~35.0px (falsely triggering pan & keypoint detector).
        RANSAC estimator must report ~2.0px and isolate inliers.
        """
        np.random.seed(123)
        n_bg = 80
        n_fg = 20

        # Background pitch turf & lines
        bg_old = np.random.uniform(100.0, 1000.0, (n_bg, 2))
        bg_new = bg_old - np.array([2.0, 0.0])  # Camera panning 2px right

        # Foreground sprinting players & referee
        fg_old = np.random.uniform(400.0, 600.0, (n_fg, 2))
        fg_new = fg_old - np.array([35.0, -12.0])  # Player running fast

        all_old = np.vstack([bg_old, fg_old])
        all_new = np.vstack([bg_new, fg_new])

        # Naive approach: max displacement
        naive_disps = [np.linalg.norm(o - n) for o, n in zip(all_old, all_new)]
        naive_max_d = max(naive_disps)

        # Robust RANSAC approach
        cx, cy, inlier_ratio, inlier_cnt = estimate_motion_ransac(all_old, all_new)

        print(f"\n[Validation Proof - Player Motion Bleed-through Fix]:")
        print(f"  True Camera Pan    : 2.0 px")
        print(f"  Naive Max-d Method : {naive_max_d:.1f} px (FAILS: triggers false _PAN_TRIGGER_PX > 15px)")
        print(f"  Robust RANSAC      : cx={cx:.2f} px, cy={cy:.2f} px (Inliers: {inlier_cnt}/100, Ratio: {inlier_ratio:.2f})")

        self.assertGreater(naive_max_d, 30.0)
        self.assertAlmostEqual(cx, 2.0, delta=0.5)
        self.assertAlmostEqual(cy, 0.0, delta=0.5)
        self.assertGreaterEqual(inlier_cnt, 75)

    def test_03_static_camera_deadband_zero_drift(self):
        """Subpixel optical noise on a locked static camera must produce exact [0.0, 0.0]."""
        np.random.seed(42)
        n_pts = 50
        old_pts = np.random.uniform(200.0, 800.0, (n_pts, 2))
        noise = np.random.normal(0.0, 0.3, (n_pts, 2))  # +/- 0.3px noise
        new_pts = old_pts + noise

        cx, cy, _, _ = estimate_motion_ransac(old_pts, new_pts, min_distance=1.2)

        self.assertEqual(cx, 0.0)
        self.assertEqual(cy, 0.0)

    def test_04_interpolation_and_track_adjustment(self):
        """Verifies full interpolation and tracks dictionary compensation."""
        estimator = RobustCameraMovementEstimator(None, min_distance=1.0)
        sampled = {0: [0.0, 0.0], 10: [10.0, 0.0], 20: [20.0, 0.0]}
        movement = estimator._interpolate_movement(sampled, 21)

        self.assertEqual(len(movement), 21)
        self.assertAlmostEqual(movement[0][0], 0.0, delta=1.0)
        self.assertAlmostEqual(movement[10][0], 10.0, delta=1.0)
        self.assertAlmostEqual(movement[20][0], 20.0, delta=1.0)

        tracks = {
            "players": [
                {1: {"position": (100.0, 100.0)}}
                for _ in range(21)
            ]
        }
        estimator.add_adjust_positions_to_tracks(tracks, movement)
        adj_10 = tracks["players"][10][1]["position_adjusted"]
        # pos - mv = (100 - ~10, 100 - 0) = (90, 100)
        self.assertAlmostEqual(adj_10[0], 90.0, delta=1.5)

    def test_05_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark: RANSAC Background Camera Motion Estimation]:
        Evaluates throughput over 2,000 consecutive video frames (100 KLT points per frame).
        """
        np.random.seed(99)
        n_frames = 2000
        n_pts = 100

        old_pts = np.random.uniform(50.0, 1200.0, (n_pts, 2)).astype(np.float32)
        # 80 inliers + 20 outliers per frame
        displacements = np.zeros((n_pts, 2), dtype=np.float32)
        displacements[:80] = [3.5, -1.2]
        displacements[80:] = np.random.uniform(-30.0, 30.0, (20, 2))
        new_pts = old_pts - displacements

        t0 = time.perf_counter()
        for _ in range(n_frames):
            estimate_motion_ransac(old_pts, new_pts)
        elapsed_s = time.perf_counter() - t0

        fps = n_frames / max(1e-5, elapsed_s)
        print(f"\n[Algorithm-only Benchmark: RANSAC Background Camera Motion Estimation]:")
        print(f"  Frames Processed : {n_frames:,}")
        print(f"  Total Time       : {elapsed_s*1000.0:.2f} ms")
        print(f"  Throughput       : {fps:,.0f} frames/sec")

        self.assertGreater(fps, 3000.0)


if __name__ == "__main__":
    unittest.main()
