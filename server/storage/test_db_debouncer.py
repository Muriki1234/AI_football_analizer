"""
test_db_debouncer.py — Unit Tests & Benchmark Suite for DebouncedStatusUpdater
"""

import threading
import time
import unittest
from typing import Any, Dict, List

from server.storage.db_debouncer import DebouncedStatusUpdater


class TestDebouncedStatusUpdater(unittest.TestCase):
    def setUp(self):
        self.call_log: List[Dict[str, Any]] = []

    def mock_db_update(
        self, session_id: str, status: str, progress: int = None, stage: str = None, error: str = None, **extra
    ):
        self.call_log.append(
            {
                "session_id": session_id,
                "status": status,
                "progress": progress,
                "stage": stage,
                "error": error,
                "extra": dict(extra),
            }
        )

    def test_critical_bypass_on_stage_and_status(self):
        updater = DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=5.0)

        # Initial call: status "analyzing", stage "init"
        updater.update("sess_1", "analyzing", progress=0, stage="init")
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(self.call_log[-1]["stage"], "init")

        # Second call immediately after: stage change "yolo_detection" -> must bypass throttle
        updater.update("sess_1", "analyzing", progress=5, stage="yolo_detection")
        self.assertEqual(len(self.call_log), 2)
        self.assertEqual(self.call_log[-1]["stage"], "yolo_detection")

        # Third call: error occurs -> must bypass throttle
        updater.update("sess_1", "failed", progress=5, error="OOM")
        self.assertEqual(len(self.call_log), 3)
        self.assertEqual(self.call_log[-1]["error"], "OOM")

    def test_throttling_and_extra_consolidation(self):
        updater = DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=2.0, min_progress_delta=10)

        # First call flushes
        updater.update("sess_1", "analyzing", progress=10, stage="tracking", metric_a=1)
        self.assertEqual(len(self.call_log), 1)

        # Next 10 fast updates with small progress increments and different extra keys
        for i in range(1, 10):
            updater.update("sess_1", "analyzing", progress=10 + (i % 3), stage="tracking", **{f"metric_{i}": i * 10})

        # All 9 fast intermediate updates should have been throttled
        self.assertEqual(len(self.call_log), 1)
        self.assertEqual(updater.throttled_calls, 9)

        # Now explicit flush
        updater.flush()
        self.assertEqual(len(self.call_log), 2)

        # Verify extra consolidation: the second flushed call has all merged metric keys
        last_extra = self.call_log[-1]["extra"]
        for i in range(1, 10):
            self.assertIn(f"metric_{i}", last_extra)
            self.assertEqual(last_extra[f"metric_{i}"], i * 10)

    def test_progress_jump_bypass(self):
        updater = DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=10.0, min_progress_delta=15)
        updater.update("sess_1", "analyzing", progress=10, stage="yolo")
        self.assertEqual(len(self.call_log), 1)

        # Jump by +20% (>= 15%) -> must trigger flush even though min_interval_sec hasn't elapsed
        updater.update("sess_1", "analyzing", progress=30, stage="yolo")
        self.assertEqual(len(self.call_log), 2)
        self.assertEqual(self.call_log[-1]["progress"], 30)

    def test_context_manager_auto_flush(self):
        with DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=10.0) as updater:
            updater.update("sess_1", "analyzing", progress=10, stage="yolo")
            # This second call is buffered
            updater.update("sess_1", "analyzing", progress=11, stage="yolo", extra_flag=True)
            self.assertEqual(len(self.call_log), 1)

        # Context manager exit must have flushed the buffered state
        self.assertEqual(len(self.call_log), 2)
        self.assertEqual(self.call_log[-1]["progress"], 11)
        self.assertEqual(self.call_log[-1]["extra"].get("extra_flag"), True)

    def test_multithreaded_concurrency(self):
        updater = DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=0.01)
        threads = []

        def worker(w_id):
            for step in range(20):
                updater.update("sess_1", "analyzing", progress=step, stage="multi", worker_id=w_id)
                time.sleep(0.001)

        for w in range(5):
            t = threading.Thread(target=worker, args=(w,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        updater.flush()
        self.assertGreater(len(self.call_log), 0)
        self.assertGreater(updater.total_calls, len(self.call_log))

    def test_algorithm_benchmark_network_savings(self):
        # Benchmark simulating 40 progress calls with simulated network latency
        # Without debouncer: 40 calls
        # With debouncer: throttled to milestone flushes
        simulated_network_latency_ms = 50.0  # 50ms per round-trip

        updater = DebouncedStatusUpdater(self.mock_db_update, min_interval_sec=0.05, min_progress_delta=10)

        t0 = time.perf_counter()
        # Simulate 40 rapid progress updates across 4 stages
        stages = ["yolo", "camera", "keypoints", "summary"]
        for i in range(40):
            stage = stages[i // 10]
            updater.update("sess_bench", "analyzing", progress=i * 2, stage=stage, tick=i)
            time.sleep(0.002)
        updater.flush()
        elapsed_actual_ms = (time.perf_counter() - t0) * 1000.0

        num_flushes = len(self.call_log)
        total_calls = 40
        unoptimized_simulated_network_ms = total_calls * simulated_network_latency_ms
        optimized_simulated_network_ms = num_flushes * simulated_network_latency_ms
        latency_reduction_pct = (
            (unoptimized_simulated_network_ms - optimized_simulated_network_ms)
            / unoptimized_simulated_network_ms
        ) * 100.0

        print(f"\n[Algorithm-only Benchmark] DebouncedStatusUpdater Network Saving:")
        print(f"  Total update calls: {total_calls} -> Database flushes: {num_flushes}")
        print(f"  Throttled calls: {updater.throttled_calls} ({updater.throttled_calls/total_calls*100:.1f}% reduction)")
        print(f"  Simulated Network Wait (50ms RTT): {unoptimized_simulated_network_ms:.0f}ms -> {optimized_simulated_network_ms:.0f}ms")
        print(f"  Network Latency Saved: {latency_reduction_pct:.1f}% ({unoptimized_simulated_network_ms - optimized_simulated_network_ms:.0f}ms saved)")

        self.assertGreaterEqual(latency_reduction_pct, 60.0)


if __name__ == "__main__":
    unittest.main()
