"""
test_homography_stabilizer.py - Unit Test & Benchmark Suite
for HomographyConditioner, CanonicalAnchorSmoother, and TemporalHomographyStabilizer.
"""

import time
import unittest
import numpy as np

from server.pipeline.homography_stabilizer import (
    HomographyConditioner,
    CanonicalAnchorSmoother,
    TemporalHomographyStabilizer,
)


class TestHomographyConditioner(unittest.TestCase):
    def setUp(self):
        self.conditioner = HomographyConditioner(
            min_points=6,
            min_x_span=3000.0,
            min_y_span=1500.0,
            ransac_thresh=400.0,
            min_inlier_ratio=0.5,
        )
        # Canonical soccer pitch keypoints
        self.canonical_dst = np.array([
            [0.0, 0.0],
            [0.0, 7000.0],
            [6000.0, 0.0],
            [6000.0, 7000.0],
            [12000.0, 0.0],
            [12000.0, 7000.0],
            [2015.0, 1450.0],
            [2015.0, 5550.0],
        ], dtype=np.float32)

        # Synthetic camera projection matrix (approx 1920x1080 view of field)
        # Perspective transform from pitch (12000x7000) to image (1920x1080)
        pts1 = np.float32([[0, 0], [12000, 0], [12000, 7000], [0, 7000]])
        pts2 = np.float32([[200, 200], [1720, 200], [1820, 950], [100, 950]])
        import cv2
        self.H_true_inv = cv2.getPerspectiveTransform(pts1, pts2)
        self.H_true = np.linalg.inv(self.H_true_inv)

        # Generate ground truth image points
        dst_homo = np.hstack([self.canonical_dst, np.ones((len(self.canonical_dst), 1), dtype=np.float32)])
        proj = (self.H_true_inv @ dst_homo.T).T
        self.canonical_src = proj[:, :2] / proj[:, 2:3]

    def test_insufficient_points_rejected(self):
        H, diag = self.conditioner.estimate_conditioned_homography(
            self.canonical_src[:4], self.canonical_dst[:4]
        )
        self.assertIsNone(H)
        self.assertEqual(diag["status"], "REJECTED")
        self.assertIn("Insufficient points", diag["reason"])

    def test_narrow_span_rejected(self):
        narrow_dst = np.array([[100, 100], [150, 120], [200, 100], [120, 180], [180, 180], [150, 150]], dtype=np.float32)
        narrow_src = narrow_dst * 0.1
        H, diag = self.conditioner.estimate_conditioned_homography(narrow_src, narrow_dst)
        self.assertIsNone(H)
        self.assertEqual(diag["status"], "REJECTED")
        self.assertIn("Spatial span too small", diag["reason"])

    def test_clean_points_accepted(self):
        H, diag = self.conditioner.estimate_conditioned_homography(
            self.canonical_src, self.canonical_dst
        )
        self.assertIsNotNone(H)
        self.assertEqual(diag["status"], "ACCEPTED")
        self.assertEqual(diag["inliers_count"], len(self.canonical_src))
        self.assertGreaterEqual(diag["inlier_ratio"], 0.9)

    def test_outlier_rejection_and_purging(self):
        noisy_src = self.canonical_src.copy()
        # Corrupt two points with huge pixel errors
        noisy_src[2] += np.array([300.0, -400.0])
        noisy_src[6] += np.array([-500.0, 200.0])

        H, diag = self.conditioner.estimate_conditioned_homography(
            noisy_src, self.canonical_dst
        )
        self.assertIsNotNone(H)
        self.assertEqual(diag["status"], "ACCEPTED")
        # Outliers must be rejected
        self.assertEqual(diag["inliers_count"], 6)
        self.assertEqual(diag["inlier_ratio"], 0.75)


class TestCanonicalAnchorSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = CanonicalAnchorSmoother(alpha=0.20)
        pts1 = np.float32([[0, 0], [12000, 0], [12000, 7000], [0, 7000]])
        pts2 = np.float32([[200, 200], [1720, 200], [1820, 950], [100, 950]])
        import cv2
        H_inv = cv2.getPerspectiveTransform(pts1, pts2)
        self.H_base = np.linalg.inv(H_inv)

    def test_initialization_sets_baseline(self):
        H_smooth = self.smoother.update(self.H_base)
        self.assertIsNotNone(H_smooth)
        np.testing.assert_allclose(H_smooth, self.H_base, atol=1e-3)

    def test_temporal_jitter_damping(self):
        # Simulate 20 frames of camera jitter: static camera with noise
        np.random.seed(42)
        raw_transformed_x = []
        smooth_transformed_x = []

        test_player_pixel = np.array([[960.0, 600.0]], dtype=np.float32)

        for i in range(30):
            # Inject noise into homography matrix
            noise = np.random.normal(0, 0.0001, size=(3, 3))
            H_noisy = self.H_base + noise
            H_smooth = self.smoother.update(H_noisy)

            # Transform test player
            raw_pt = (H_noisy @ np.array([960.0, 600.0, 1.0]))
            raw_transformed_x.append(raw_pt[0] / raw_pt[2])

            smooth_pt = (H_smooth @ np.array([960.0, 600.0, 1.0]))
            smooth_transformed_x.append(smooth_pt[0] / smooth_pt[2])

        # Ignore first 5 warmup frames
        raw_std = np.std(raw_transformed_x[5:])
        smooth_std = np.std(smooth_transformed_x[5:])

        self.assertLess(smooth_std, raw_std)
        jitter_reduction = (1.0 - smooth_std / max(raw_std, 1e-6)) * 100
        print(f"\n[Algorithm-only Benchmark] Homography Jitter Reduction: {jitter_reduction:.1f}% (raw_std={raw_std:.2f}m, smooth_std={smooth_std:.2f}m)")
        self.assertGreater(jitter_reduction, 40.0, "Temporal anchor smoothing must damp jitter by >= 40%")

    def test_missing_frame_fallback(self):
        self.smoother.update(self.H_base)
        # Next frame detection dropped (None)
        H_fallback = self.smoother.update(None)
        self.assertIsNotNone(H_fallback)
        np.testing.assert_allclose(H_fallback, self.H_base, atol=1e-3)


class TestTemporalHomographyStabilizer(unittest.TestCase):
    def setUp(self):
        self.stabilizer = TemporalHomographyStabilizer(alpha=0.25)

    def test_end_to_end_transformation_and_clamping(self):
        pts1 = np.float32([[0, 0], [12000, 0], [12000, 7000], [0, 7000]])
        pts2 = np.float32([[200, 200], [1720, 200], [1820, 950], [100, 950]])
        import cv2
        H_inv = cv2.getPerspectiveTransform(pts1, pts2)
        H_true = np.linalg.inv(H_inv)

        # Canonical points
        dst = np.array([[0, 0], [0, 7000], [6000, 0], [6000, 7000], [12000, 0], [12000, 7000]], dtype=np.float32)
        dst_h = np.hstack([dst, np.ones((len(dst), 1), dtype=np.float32)])
        proj = (H_inv @ dst_h.T).T
        src = proj[:, :2] / proj[:, 2:3]

        H_stab, diag = self.stabilizer.process_frame(src, dst)
        self.assertIsNotNone(H_stab)
        self.assertEqual(diag["status"], "ACCEPTED")

        # Test player points
        players = np.array([
            [960.0, 540.0],   # center field
            [-500.0, -500.0], # wild out of bounds point
        ], dtype=np.float32)

        coords = self.stabilizer.transform_points(H_stab, players, clamp_to_pitch=True)
        self.assertEqual(len(coords), 2)
        # Center field should be roughly 6000, 3500
        self.assertAlmostEqual(coords[0, 0], 6000.0, delta=500.0)
        self.assertAlmostEqual(coords[0, 1], 3500.0, delta=500.0)
        # Wild point clamped to boundaries
        self.assertGreaterEqual(coords[1, 0], -500.0)
        self.assertLessEqual(coords[1, 0], 12500.0)

    def test_algorithm_only_benchmark_throughput(self):
        pts1 = np.float32([[0, 0], [12000, 0], [12000, 7000], [0, 7000]])
        pts2 = np.float32([[200, 200], [1720, 200], [1820, 950], [100, 950]])
        import cv2
        H_inv = cv2.getPerspectiveTransform(pts1, pts2)

        dst = np.array([[0, 0], [0, 7000], [6000, 0], [6000, 7000], [12000, 0], [12000, 7000]], dtype=np.float32)
        dst_h = np.hstack([dst, np.ones((len(dst), 1), dtype=np.float32)])
        proj = (H_inv @ dst_h.T).T
        src = proj[:, :2] / proj[:, 2:3]

        # Warmup
        for _ in range(20):
            self.stabilizer.process_frame(src, dst)

        t0 = time.perf_counter()
        n_frames = 1000
        for _ in range(n_frames):
            self.stabilizer.process_frame(src, dst)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_frame_ms = elapsed_ms / n_frames

        print(f"\n[Algorithm-only Benchmark] TemporalHomographyStabilizer: {n_frames} frames in {elapsed_ms:.1f}ms "
              f"({per_frame_ms:.4f}ms/frame, throughput: {n_frames / (elapsed_ms / 1000):.0f} FPS)")
        self.assertLess(per_frame_ms, 0.5, "Homography stabilization must execute in < 0.5ms per frame")


if __name__ == "__main__":
    unittest.main()
