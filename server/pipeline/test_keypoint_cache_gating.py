"""
test_keypoint_cache_gating.py - Unit Test & Benchmark Suite
for HomographyCache and KeypointCacheGater.

Verifies:
1. Sub-pixel homography caching and RANSAC bypass.
2. Static camera keypoint detection gating and 60%+ inference reduction.
3. Rapid camera pan trigger resilience.
4. High-throughput algorithm microbenchmarks.
"""

import time
import unittest
import numpy as np

from server.pipeline.keypoint_gating import (
    HomographyCache,
    KeypointCacheGater,
)
from server.pipeline.homography_stabilizer import (
    TemporalHomographyStabilizer,
)


class TestHomographyCache(unittest.TestCase):
    def setUp(self):
        self.cache = HomographyCache(tolerance_px=0.85, max_age_frames=100)
        self.canonical_src = np.array([
            [200.0, 150.0],
            [1700.0, 150.0],
            [1800.0, 900.0],
            [150.0, 900.0],
            [950.0, 520.0],
            [950.0, 150.0],
        ], dtype=np.float32)
        self.mock_H = np.array([
            [1.2, 0.05, 10.0],
            [-0.02, 1.15, 20.0],
            [0.0001, 0.0002, 1.0],
        ], dtype=np.float32)
        self.mock_diag = {"inliers_count": 6, "status": "ACCEPTED"}

    def test_cache_miss_on_empty(self):
        res = self.cache.get(self.canonical_src)
        self.assertIsNone(res)
        stats = self.cache.get_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 0)

    def test_cache_hit_on_subpixel_shift(self):
        self.cache.put(self.canonical_src, None, self.mock_H, self.mock_diag)
        # Shift keypoints by tiny sub-pixel amount (0.2 px < 0.85 px)
        shifted_src = self.canonical_src + 0.2
        res = self.cache.get(shifted_src)
        self.assertIsNotNone(res)
        H_cached, diag_cached = res
        np.testing.assert_allclose(H_cached, self.mock_H, rtol=1e-5)
        self.assertTrue(diag_cached["cache_hit"])
        self.assertLess(diag_cached["max_shift_px"], 0.85)

        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["lookups"], 1)

    def test_cache_miss_on_large_shift(self):
        self.cache.put(self.canonical_src, None, self.mock_H, self.mock_diag)
        # Shift keypoints by 2.0 px (> 0.85 px tolerance)
        shifted_src = self.canonical_src + 2.0
        res = self.cache.get(shifted_src)
        self.assertIsNone(res)
        stats = self.cache.get_stats()
        self.assertEqual(stats["misses"], 1)

    def test_cache_miss_on_shape_mismatch(self):
        self.cache.put(self.canonical_src, None, self.mock_H, self.mock_diag)
        # Drop one keypoint (shape changes from (6, 2) to (5, 2))
        res = self.cache.get(self.canonical_src[:5])
        self.assertIsNone(res)

    def test_cache_age_expiration(self):
        cache_short = HomographyCache(tolerance_px=0.85, max_age_frames=5)
        cache_short.put(self.canonical_src, None, self.mock_H, self.mock_diag)

        # First 5 lookups hit
        for i in range(5):
            res = cache_short.get(self.canonical_src)
            self.assertIsNotNone(res)

        # 6th lookup must expire
        res_expired = cache_short.get(self.canonical_src)
        self.assertIsNone(res_expired)

    def test_cache_throughput_benchmark(self):
        self.cache.put(self.canonical_src, None, self.mock_H, self.mock_diag)
        n_iters = 50000
        shifted_src = self.canonical_src + 0.1

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = self.cache.get(shifted_src)
        elapsed = time.perf_counter() - t0

        fps = n_iters / max(elapsed, 1e-6)
        print(f"\n[Algorithm-only Benchmark] HomographyCache throughput: {n_iters} lookups in {elapsed*1000:.1f}ms ({fps:,.0f} lookups/sec)")
        self.assertGreater(fps, 100000.0)


class TestKeypointCacheGater(unittest.TestCase):
    def setUp(self):
        self.gater = KeypointCacheGater(
            pan_trigger_px=15.0,
            drift_threshold_px=2.5,
            max_static_interval=60,
            keypoint_stride=20,
            static_deadband_px=0.8,
        )

    def test_initial_frame_always_sampled(self):
        sampled, reason = self.gater.should_sample(0, (0.0, 0.0))
        self.assertTrue(sampled)
        self.assertEqual(reason, "INITIAL_FRAME")

    def test_stationary_camera_stride_gating(self):
        # Frame 0: anchor
        self.gater.should_sample(0, (0.0, 0.0))

        # Stationary camera (sub-pixel sensor noise 0.3px < 0.8px deadband)
        # Frame 20 (stride 1): must be gated skip
        sampled_20, reason_20 = self.gater.should_sample(20, (0.3, 0.2))
        self.assertFalse(sampled_20)
        self.assertEqual(reason_20, "GATED_STATIC_SKIP")

        # Frame 40 (stride 2): must be gated skip
        sampled_40, reason_40 = self.gater.should_sample(40, (0.1, 0.1))
        self.assertFalse(sampled_40)
        self.assertEqual(reason_40, "GATED_STATIC_SKIP")

        # Frame 60 (stride 3, 60 frames since frame 0): guard anchor must trigger!
        sampled_60, reason_60 = self.gater.should_sample(60, (0.2, 0.1))
        self.assertTrue(sampled_60)
        self.assertEqual(reason_60, "MAX_INTERVAL_REACHED")

    def test_pan_trigger(self):
        self.gater.should_sample(0, (0.0, 0.0))
        # Frame 12: camera makes sudden sweep of 18.0px (> 15.0px)
        sampled_12, reason_12 = self.gater.should_sample(12, (15.0, 10.0))
        self.assertTrue(sampled_12)
        self.assertEqual(reason_12, "PAN_TRIGGER")

    def test_cumulative_drift_trigger(self):
        self.gater.should_sample(0, (0.0, 0.0))
        # Small camera drifts of 1.2px per frame (> 0.8px deadband)
        # In 3 frames, accumulated drift reaches 3.6px (> 2.5px threshold)
        self.gater.should_sample(1, (1.2, 0.0))
        self.gater.should_sample(2, (1.2, 0.0))
        self.gater.should_sample(3, (1.2, 0.0))

        # Next stride frame (frame 20) must trigger due to DRIFT_EXCEEDED
        sampled_20, reason_20 = self.gater.should_sample(20, (0.0, 0.0))
        self.assertTrue(sampled_20)
        self.assertEqual(reason_20, "DRIFT_EXCEEDED")

    def test_no_flow_fallback(self):
        # When optical flow is disabled (cam_movement is None)
        sampled_0, _ = self.gater.should_sample(0, None)
        self.assertTrue(sampled_0)

        sampled_10, reason_10 = self.gater.should_sample(10, None)
        self.assertFalse(sampled_10)
        self.assertEqual(reason_10, "NON_STRIDE_SKIP")

        sampled_20, reason_20 = self.gater.should_sample(20, None)
        self.assertTrue(sampled_20)
        self.assertEqual(reason_20, "STRIDE_NO_FLOW")

    def test_plan_chunk_sampling_reduction(self):
        # 120-frame chunk with stationary camera
        chunk_len = 120
        movements = [(0.2, 0.1) for _ in range(chunk_len)]
        samples, diag = self.gater.plan_chunk_sampling(0, chunk_len, movements)

        # Standard stride=20 without gating would sample: 0, 20, 40, 60, 80, 100 (6 frames)
        # With gating, frames 20, 40, 80, 100 are pruned; only 0 and 60 are sampled (2 frames)
        self.assertEqual(samples, [0, 60])
        self.assertEqual(diag["sampled_count"], 2)
        self.assertGreaterEqual(diag["skip_percentage"], 98.0)
        stats = self.gater.get_stats()
        self.assertEqual(stats["gated_count"], 4)  # 4 stride frames skipped


class TestStabilizerCacheIntegration(unittest.TestCase):
    def setUp(self):
        self.stabilizer = TemporalHomographyStabilizer(
            min_points=6,
            alpha=0.25,
            enable_cache=True,
            cache_tolerance=0.85,
        )
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

        pts1 = np.float32([[0, 0], [12000, 0], [12000, 7000], [0, 7000]])
        pts2 = np.float32([[200, 200], [1720, 200], [1820, 950], [100, 950]])
        import cv2
        H_true_inv = cv2.getPerspectiveTransform(pts1, pts2)
        H_true = np.linalg.inv(H_true_inv)
        dst_homo = np.hstack([self.canonical_dst, np.ones((len(self.canonical_dst), 1), dtype=np.float32)])
        proj = (H_true_inv @ dst_homo.T).T
        self.canonical_src = proj[:, :2] / proj[:, 2:3]

    def test_stabilizer_cache_hits_on_stationary_sequence(self):
        # Frame 0: cold start (RANSAC + smoother)
        H0, diag0 = self.stabilizer.process_frame(self.canonical_src, self.canonical_dst)
        self.assertIsNotNone(H0)
        self.assertFalse(diag0.get("cache_hit", False))

        # Frames 1..50: sub-pixel jitter (< 0.5px) -> all must hit cache!
        for i in range(1, 51):
            jittered_src = self.canonical_src + np.random.uniform(-0.3, 0.3, size=self.canonical_src.shape).astype(np.float32)
            H_i, diag_i = self.stabilizer.process_frame(jittered_src, self.canonical_dst)
            self.assertIsNotNone(H_i)
            self.assertTrue(diag_i["cache_hit"])
            np.testing.assert_allclose(H_i, H0, rtol=1e-5)

        stats = self.stabilizer.get_cache_stats()
        self.assertEqual(stats["hits"], 50)
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hit_ratio"], 0.98)


if __name__ == "__main__":
    unittest.main()
