"""
test_throttled_progress_reporter.py — Unit tests for ThrottledProgressReporter
"""

import time
import unittest
from server.pipeline.throttled_progress_reporter import ThrottledProgressReporter


class TestThrottledProgressReporter(unittest.TestCase):
    def test_temporal_and_delta_throttling(self):
        dispatched_updates = []

        def mock_sink(prog, stage, extra):
            dispatched_updates.append((prog, stage))

        reporter = ThrottledProgressReporter(
            update_fn=mock_sink,
            min_interval_sec=0.10,  # 100ms
            min_delta_pct=5,        # 5% progress delta
        )

        try:
            # Emit 50 rapid calls with sub-threshold deltas (0% -> 1% -> 2% -> 3% -> 4%)
            for i in range(50):
                pct = 10 + (i % 4)  # fluctuates 10, 11, 12, 13
                reporter.update(pct, stage=f"chunk_{i}")
                time.sleep(0.001)

            # Delta is within 4% and time < 100ms, so only the initial call should be dispatched
            time.sleep(0.05)
            # Now trigger a large delta jump (10% -> 20%)
            reporter.update(20, stage="chunk_jump")
            time.sleep(0.05)

            # Now wait for time interval to expire (100ms) and emit same progress
            time.sleep(0.12)
            reporter.update(20, stage="chunk_time_tick")
            time.sleep(0.05)

            telemetry = reporter.get_telemetry()
            self.assertGreater(telemetry["throttled_skips"], 40)
            self.assertGreater(telemetry["reduction_percentage"], 80.0)
            self.assertLessEqual(telemetry["dispatched_network_calls"], 5)
        finally:
            reporter.flush_and_close()

    def test_terminal_flush_is_synchronous(self):
        dispatched_updates = []

        def mock_sink(prog, stage, extra):
            dispatched_updates.append((prog, stage))

        reporter = ThrottledProgressReporter(
            update_fn=mock_sink,
            min_interval_sec=10.0,
            min_delta_pct=20,
        )

        try:
            # Emit normal update
            reporter.update(10, "start")

            # Terminal update must be delivered immediately even if delta < 20%
            reporter.update(100, "done", is_terminal=True)
            self.assertEqual(dispatched_updates[-1], (100, "done"))
        finally:
            reporter.flush_and_close()

    def test_simulated_runpod_workload_reduction(self):
        # Simulates 169 chunks of a 14-minute video
        dispatched = []

        def mock_sink(prog, stage, extra):
            time.sleep(0.005)  # simulate brief network call
            dispatched.append(prog)

        reporter = ThrottledProgressReporter(
            update_fn=mock_sink,
            min_interval_sec=0.05,  # fast test interval
            min_delta_pct=3,
        )

        try:
            # 169 chunks advancing progress from 10% to 55%
            for chunk_idx in range(169):
                ratio = (chunk_idx + 1) / 169.0
                pct = int(10 + ratio * 45)
                reporter.update(pct, stage=f"chunk_{chunk_idx}/169")
                time.sleep(0.001)

            reporter.update(100, stage="complete", is_terminal=True)
            time.sleep(0.1)

            t = reporter.get_telemetry()
            # 170 calls should be reduced to ~15-25 calls (>80% reduction)
            self.assertGreater(t["reduction_percentage"], 75.0)
            self.assertLess(t["dispatched_network_calls"], 35)
            self.assertGreater(t["estimated_saved_latency_sec"], 30.0)
        finally:
            reporter.flush_and_close()


if __name__ == "__main__":
    unittest.main()
