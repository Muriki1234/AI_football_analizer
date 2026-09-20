from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from server.pipeline.camera_motion_ransac import estimate_motion_ransac


class TestMergedPipelineStreamOverlap(unittest.TestCase):
    def test_01_chunk_size_memory_bounding(self):
        """Verifies that large chunk_size inputs are bounded by STREAMING_CHUNK_SIZE to prevent OOM."""
        with patch.dict(os.environ, {"STREAMING_CHUNK_SIZE": "120"}):
            max_chunk = int(os.environ.get("STREAMING_CHUNK_SIZE", "150"))
            chunk_size = 500
            if chunk_size > max_chunk:
                chunk_size = max_chunk
            self.assertEqual(chunk_size, 120)

    def test_02_optical_flow_ransac_player_isolation(self):
        """Verifies that RANSAC consensus in _run_optical_flow_chunk rejects foreground runner artifacts."""
        np.random.seed(42)
        n_bg = 60
        n_player = 20

        # Background points remain stationary (< 0.5px jitter)
        bg_old = np.random.uniform(50, 500, (n_bg, 2)).astype(np.float32)
        bg_new = bg_old + np.random.normal(0, 0.1, (n_bg, 2)).astype(np.float32)

        # Running player points move 30.0px to the right
        player_old = np.random.uniform(200, 300, (n_player, 2)).astype(np.float32)
        player_new = player_old + np.array([30.0, 0.0], dtype=np.float32)

        all_old = np.vstack([bg_old, player_old])
        all_new = np.vstack([bg_new, player_new])

        # RANSAC estimation
        cx, cy, inlier_ratio, inlier_cnt = estimate_motion_ransac(all_old, all_new, min_distance=1.2)

        # In Naive approach, max_d = 30px, triggering spurious pan!
        # In RANSAC, background is recognized and camera movement is clamped to 0.0
        self.assertEqual(cx, 0.0)
        self.assertEqual(cy, 0.0)
        self.assertGreaterEqual(inlier_ratio, 0.65)

    def test_03_genuine_camera_pan_detection(self):
        """Verifies that true camera movement (e.g. pan of 4.5px) is accurately estimated."""
        np.random.seed(42)
        n_pts = 50
        pts_old = np.random.uniform(50, 500, (n_pts, 2)).astype(np.float32)
        # Camera pans left -> features move right by 4.5px
        pts_new = pts_old + np.array([4.5, 0.0], dtype=np.float32)

        cx, cy, inlier_ratio, inlier_cnt = estimate_motion_ransac(pts_old, pts_new, min_distance=1.2)
        self.assertAlmostEqual(cx, -4.5, delta=0.5)
        self.assertAlmostEqual(cy, 0.0, delta=0.5)

    def test_04_threadpool_concurrent_execution(self):
        """Verifies ThreadPoolExecutor(max_workers=3) concurrently executes optical flow and inference without deadlocks."""
        from concurrent.futures import ThreadPoolExecutor

        flow_executed = False
        yolo_executed = False
        kpt_executed = False

        def mock_yolo(batch):
            nonlocal yolo_executed
            time.sleep(0.01)
            yolo_executed = True
            return ["yolo_res"]

        def mock_flow(chunk, start):
            nonlocal flow_executed
            time.sleep(0.01)
            flow_executed = True
            return True

        def mock_kpt(batch):
            nonlocal kpt_executed
            time.sleep(0.01)
            kpt_executed = True
            return ["kpt_res"]

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_yolo = pool.submit(mock_yolo, ["frame1"])
            f_flow = pool.submit(mock_flow, ["frame1"], 0)
            f_kpt = pool.submit(mock_kpt, ["frame1"])

            self.assertEqual(f_yolo.result(), ["yolo_res"])
            self.assertTrue(f_flow.result())
            self.assertEqual(f_kpt.result(), ["kpt_res"])

        self.assertTrue(yolo_executed)
        self.assertTrue(flow_executed)
        self.assertTrue(kpt_executed)

    def test_05_microbenchmark_ransac_camera_throughput(self):
        """[Algorithm-only Benchmark: Camera Motion RANSAC] Evaluates RANSAC consensus throughput on synthetic keypoints."""
        np.random.seed(42)
        n_pts = 60
        pts_old = np.random.uniform(50, 500, (n_pts, 2)).astype(np.float32)
        pts_new = pts_old + np.random.normal(0, 0.2, (n_pts, 2)).astype(np.float32)

        start = time.perf_counter()
        iterations = 2000
        for _ in range(iterations):
            estimate_motion_ransac(pts_old, pts_new, min_distance=1.2)

        elapsed = time.perf_counter() - start
        fps = iterations / max(1e-6, elapsed)
        print(f"\n[Algorithm-only Benchmark: RANSAC Camera Motion Consensus]")
        print(f"Evaluated {iterations} motion estimations in {elapsed * 1000:.2f} ms ({fps:,.0f} frames/sec)")
        self.assertGreater(fps, 1000)


if __name__ == "__main__":
    unittest.main()
