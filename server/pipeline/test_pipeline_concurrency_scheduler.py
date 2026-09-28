"""
test_pipeline_concurrency_scheduler.py — Unit Tests and Overlap Benchmarks for PipelineConcurrencyScheduler
"""

import time
import unittest
from unittest.mock import patch
from server.pipeline.pipeline_concurrency_scheduler import (
    PipelineConcurrencyScheduler,
    compute_samurai_concurrency_cap,
)


class TestPipelineConcurrencyScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = PipelineConcurrencyScheduler(min_free_vram_gb=6.0, default_mode="auto")

    def test_01_mode_force_serial(self):
        """When forced serial, always returns False regardless of VRAM."""
        run_c, reason = self.scheduler.should_run_concurrently("force_serial")
        self.assertFalse(run_c)
        self.assertIn("forced to sequential", reason)

    def test_02_mode_force_concurrent(self):
        """When forced concurrent, always returns True regardless of VRAM."""
        run_c, reason = self.scheduler.should_run_concurrently("force_concurrent")
        self.assertTrue(run_c)
        self.assertIn("forced to concurrent", reason)

    @patch.object(PipelineConcurrencyScheduler, "get_vram_headroom")
    def test_03_vram_headroom_check_low_memory(self, mock_vram):
        """Under auto mode, VRAM < 6GB automatically falls back to serial."""
        mock_vram.return_value = (3.5, 24.0, True)  # 3.5 GB free < 6.0 GB
        run_c, reason = self.scheduler.should_run_concurrently("auto")
        self.assertFalse(run_c)
        self.assertIn("Insufficient VRAM headroom", reason)

    @patch.object(PipelineConcurrencyScheduler, "get_vram_headroom")
    def test_04_vram_headroom_check_ample_memory(self, mock_vram):
        """Under auto mode, VRAM >= 6GB activates concurrency."""
        mock_vram.return_value = (16.2, 24.0, True)  # 16.2 GB free >= 6.0 GB
        run_c, reason = self.scheduler.should_run_concurrently("auto")
        self.assertTrue(run_c)
        self.assertIn("VRAM headroom verified", reason)

    def test_05_concurrent_execution_overlap_timing(self):
        """
        Simulates SAMURAI (0.12s) and YOLO (0.16s).
        Serial: 0.12s + 0.16s = 0.28s.
        Concurrent: max(0.12, 0.16) ~= 0.16s.
        """
        def mock_samurai(sess_id, s, segs, sm):
            time.sleep(0.12)
            return {"bboxes_count": 120}

        def mock_yolo(sess_id, s, sm):
            # Wait for samurai event if attached
            event = s.get("_samurai_done_event")
            time.sleep(0.16)
            if event:
                event.wait(timeout=1.0)
            return {"yolo_tracks_count": 500}

        result = self.scheduler.execute_pipeline(
            session_id="test_sess_01",
            session={"video_path": "dummy.mp4"},
            segments=[{"start_frame": 0, "end_frame": 100}],
            sm=None,
            samurai_runner=mock_samurai,
            yolo_runner=mock_yolo,
            override_mode="force_concurrent",
        )

        self.assertEqual(result["mode"], "concurrent")
        self.assertGreater(result["overlap_saved_sec"], 0.05)
        self.assertLess(result["wall_clock_sec"], 0.24)
        self.assertEqual(result["samurai_result"]["bboxes_count"], 120)
        self.assertEqual(result["yolo_result"]["yolo_tracks_count"], 500)

    def test_06_samurai_exception_propagates_cleanly(self):
        """Exceptions in the asynchronous worker thread are safely captured and propagated."""
        def mock_failing_samurai(sess_id, s, segs, sm):
            raise ValueError("Corrupt segment weights")

        def mock_yolo(sess_id, s, sm):
            time.sleep(0.05)
            return {"ok": True}

        with self.assertRaises(RuntimeError) as ctx:
            self.scheduler.execute_pipeline(
                session_id="test_sess_err",
                session={"video_path": "dummy.mp4"},
                segments=[],
                sm=None,
                samurai_runner=mock_failing_samurai,
                yolo_runner=mock_yolo,
                override_mode="force_concurrent",
            )
        self.assertIn("Concurrent SAMURAI task failed", str(ctx.exception))

    def test_07_algorithm_only_benchmark_concurrency_savings(self):
        """
        [Algorithm-only Benchmark: Pipeline Concurrency Overlap Simulation]:
        Simulates scaled 14-minute match profile (SAMURAI ratio: 96ms, YOLO ratio: 333ms).
        Measures real wall-clock savings from thread concurrency.
        """
        samurai_cost = 0.060  # 60ms simulated
        yolo_cost = 0.120     # 120ms simulated

        def mock_samurai(sess_id, s, segs, sm):
            time.sleep(samurai_cost)
            return True

        def mock_yolo(sess_id, s, sm):
            time.sleep(yolo_cost)
            return True

        # Serial Baseline
        res_serial = self.scheduler.execute_pipeline(
            session_id="bench_sess", session={}, segments=[], sm=None,
            samurai_runner=mock_samurai, yolo_runner=mock_yolo,
            override_mode="force_serial",
        )

        # Concurrent Execution
        res_concurrent = self.scheduler.execute_pipeline(
            session_id="bench_sess", session={}, segments=[], sm=None,
            samurai_runner=mock_samurai, yolo_runner=mock_yolo,
            override_mode="force_concurrent",
        )

        time_saved = res_serial["wall_clock_sec"] - res_concurrent["wall_clock_sec"]
        pct_saved = (time_saved / res_serial["wall_clock_sec"]) * 100.0

        print(f"\n[Algorithm-only Benchmark: Pipeline Concurrency Overlap Simulation]:")
        print(f"  Serial Wall-Clock     : {res_serial['wall_clock_sec']:.3f}s (SAMURAI: {res_serial['samurai_sec']:.3f}s + YOLO: {res_serial['yolo_sec']:.3f}s)")
        print(f"  Concurrent Wall-Clock : {res_concurrent['wall_clock_sec']:.3f}s (Overlapped)")
        print(f"  Wall-Clock Saved      : {time_saved:.3f}s ({pct_saved:.1f}% reduction)")

        self.assertGreater(res_serial["wall_clock_sec"], res_concurrent["wall_clock_sec"])
        self.assertGreater(pct_saved, 25.0)

    def test_08_compute_samurai_concurrency_cap_rounding_fix(self):
        """
        Verify that 1926x1080 (which previously caused int(10.9828) -> 10) properly rounds to 11
        under cap=11, preventing the 11th segment from being isolated and queued.
        """
        cap = compute_samurai_concurrency_cap(1926, 1080, env_cap_override=11)
        self.assertEqual(cap, 11)

    def test_09_compute_samurai_concurrency_cap_safe_default(self):
        """
        Verify that default concurrency cap is bounded to 4 (4-parallel execution),
        while still supporting explicit overrides via env_cap_override.
        """
        # Default with no override is 4
        self.assertEqual(compute_samurai_concurrency_cap(1920, 1080), 4)
        # Explicit overrides work as intended
        self.assertEqual(compute_samurai_concurrency_cap(1920, 1080, env_cap_override=6), 6)
        self.assertEqual(compute_samurai_concurrency_cap(1920, 1080, env_cap_override=8), 8)

    def test_10_compute_samurai_concurrency_cap_high_res_scaling(self):
        """
        Verify that 4K (3840x2160 = 4x pixels) scales down safely via sqrt:
        4 / sqrt(4) = 2
        """
        cap_4k = compute_samurai_concurrency_cap(3840, 2160)
        self.assertEqual(cap_4k, 2)

    def test_11_schedule_balanced_waves(self):
        from server.pipeline.pipeline_concurrency_scheduler import schedule_balanced_waves
        # 10 segments with cap 5 -> 5 + 5
        self.assertEqual(schedule_balanced_waves(10, 5), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        # 8 segments with cap 4 -> 4 + 4
        self.assertEqual(schedule_balanced_waves(8, 4), [[0, 1, 2, 3], [4, 5, 6, 7]])
        # 11 segments with cap 4 -> 4 + 4 + 3
        self.assertEqual(schedule_balanced_waves(11, 4), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]])
        # Edge cases
        self.assertEqual(schedule_balanced_waves(0, 5), [])
        self.assertEqual(schedule_balanced_waves(3, 5), [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
