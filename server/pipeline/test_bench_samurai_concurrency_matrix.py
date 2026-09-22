"""
test_bench_samurai_concurrency_matrix.py — Unit Tests for SAMURAI Matrix Harness
"""

import os
import tempfile
import unittest
from pathlib import Path

from server.pipeline.bench_samurai_concurrency_matrix import (
    BackgroundTelemetryMonitor,
    ConcurrencyRunResult,
    SamuraiConcurrencyBenchmarkHarness,
)


class TestSamuraiConcurrencyMatrixHarness(unittest.TestCase):
    def test_01_telemetry_monitor_start_stop(self):
        monitor = BackgroundTelemetryMonitor(polling_interval_sec=0.1)
        monitor.start()
        # Allow at least one poll
        import time
        time.sleep(0.25)
        gpu_avg, gpu_peak, vram_peak, ram_peak, cpu_avg = monitor.stop()

        self.assertGreaterEqual(gpu_avg, 0.0)
        self.assertGreaterEqual(gpu_peak, 0.0)
        self.assertGreaterEqual(vram_peak, 0.0)
        self.assertGreaterEqual(ram_peak, 0.0)
        self.assertGreaterEqual(cpu_avg, 0.0)

    def test_02_harness_dry_run_matrix_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "test_results.json"
            harness = SamuraiConcurrencyBenchmarkHarness(
                video_path="dummy_video.mp4",
                concurrency_values=[1, 2, 4],
                runs_per_concurrency=1,
                dry_run=True,
                output_json=str(out_json),
            )

            results = harness.run_matrix()

            self.assertEqual(len(results), 3)
            self.assertTrue(out_json.exists())

            # Verify dataclass fields
            for r in results:
                self.assertIsInstance(r, ConcurrencyRunResult)
                self.assertIn(r.concurrency, [1, 2, 4])
                self.assertGreater(r.full_e2e_wall_clock, 0.0)
                self.assertGreater(r.samurai_wall_clock, 0.0)
                self.assertEqual(r.tracking_checksum, "dryrun_simulated_checksum")
                self.assertEqual(r.samurai_coverage_pct, 95.7)

    def test_03_cold_start_cleanup(self):
        harness = SamuraiConcurrencyBenchmarkHarness(
            video_path="dummy.mp4",
            concurrency_values=[1],
            dry_run=True,
        )
        # Should not raise even if files do not exist
        harness.perform_cold_start_cleanup("test_session_cleanup_123")


if __name__ == "__main__":
    unittest.main()
