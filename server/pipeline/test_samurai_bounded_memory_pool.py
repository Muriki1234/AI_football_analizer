"""
test_samurai_bounded_memory_pool.py — Unit tests for SamuraiBoundedWorkerPool
"""

import threading
import time
import unittest
from unittest.mock import patch

from server.pipeline.samurai_bounded_memory_pool import SamuraiBoundedWorkerPool


class TestSamuraiBoundedWorkerPool(unittest.TestCase):
    def setUp(self):
        self.pool = SamuraiBoundedWorkerPool(max_total_ram_gb=24.0, ram_per_1080p_worker_gb=7.5)

    def test_estimate_worker_ram(self):
        # 1080p should be ~7.5 GB
        ram_1080 = self.pool.estimate_worker_ram_gb(1920, 1080)
        self.assertAlmostEqual(ram_1080, 7.5, delta=0.1)

        # 720p should require less RAM
        ram_720 = self.pool.estimate_worker_ram_gb(1280, 720)
        self.assertLess(ram_720, ram_1080)

        # 4K should scale reasonably
        ram_4k = self.pool.estimate_worker_ram_gb(3840, 2160)
        self.assertGreater(ram_4k, ram_1080)

    @patch.object(SamuraiBoundedWorkerPool, "get_system_ram_gb", return_value=(64.0, 128.0))
    def test_safe_concurrency_concurrent_mode(self, mock_ram):
        # 11 segments in concurrent mode with YOLO
        plan = self.pool.calculate_safe_worker_concurrency(
            n_segments=11, orig_w=1920, orig_h=1080, is_concurrent_with_yolo=True
        )
        # Should be capped at 3 workers to prevent GPU contention
        self.assertEqual(plan["max_workers"], 3)
        self.assertLessEqual(plan["estimated_peak_ram_gb"], 24.0)
        self.assertEqual(plan["batches_required"], 4)

    @patch.object(SamuraiBoundedWorkerPool, "get_system_ram_gb", return_value=(64.0, 128.0))
    def test_safe_concurrency_serial_mode(self, mock_ram):
        # In serial mode, can use full 24GB budget (24 / 7.5 = 3 workers)
        plan = self.pool.calculate_safe_worker_concurrency(
            n_segments=11, orig_w=1920, orig_h=1080, is_concurrent_with_yolo=False
        )
        self.assertEqual(plan["max_workers"], 3)
        self.assertLessEqual(plan["estimated_peak_ram_gb"], 24.0)

    @patch.object(SamuraiBoundedWorkerPool, "get_system_ram_gb", return_value=(8.0, 16.0))
    def test_low_ram_environment_clamps_to_single_worker(self, mock_ram):
        # Available RAM is only 8GB -> budget is 6.4GB -> 0 full 7.5GB workers -> clamps to min 1
        plan = self.pool.calculate_safe_worker_concurrency(
            n_segments=5, orig_w=1920, orig_h=1080, is_concurrent_with_yolo=True
        )
        self.assertEqual(plan["max_workers"], 1)

    def test_execute_bounded_tracking_success(self):
        segments = [{"seg_id": i, "frames": 100} for i in range(7)]
        active_workers = []
        max_active_observed = [0]
        lock = threading.Lock()

        def mock_worker(idx, seg):
            with lock:
                active_workers.append(idx)
                if len(active_workers) > max_active_observed[0]:
                    max_active_observed[0] = len(active_workers)
            time.sleep(0.02)
            with lock:
                active_workers.remove(idx)
            return {"seg_idx": idx, "bboxes": {0: [10, 20, 30, 40]}}

        progress_calls = []

        res = self.pool.execute_bounded_tracking(
            segments=segments,
            worker_fn=mock_worker,
            orig_w=1920,
            orig_h=1080,
            is_concurrent_with_yolo=True,
            progress_cb=lambda done, total: progress_calls.append((done, total)),
        )

        self.assertEqual(res["completed_segments"], 7)
        self.assertEqual(len(res["results"]), 7)
        self.assertLessEqual(max_active_observed[0], res["workers_used"])
        self.assertEqual(progress_calls[-1], (7, 7))

    def test_execute_bounded_tracking_kill_event(self):
        segments = [{"seg_id": i} for i in range(5)]
        kill_event = threading.Event()

        def mock_worker(idx, seg):
            time.sleep(0.05)
            return idx

        # Trigger kill event quickly
        threading.Timer(0.01, kill_event.set).start()

        with self.assertRaises(RuntimeError):
            self.pool.execute_bounded_tracking(
                segments=segments,
                worker_fn=mock_worker,
                kill_event=kill_event,
            )


if __name__ == "__main__":
    unittest.main()
