"""
test_cuda_stream_priority_concurrency_governor.py
=================================================
Unit tests for CudaStreamPriorityConcurrencyGovernor.
"""

import unittest
from server.pipeline.cuda_stream_priority_concurrency_governor import (
    CudaStreamPriorityConcurrencyGovernor,
)


class TestCudaStreamPriorityConcurrencyGovernor(unittest.TestCase):
    def setUp(self):
        self.governor = CudaStreamPriorityConcurrencyGovernor(
            vram_budget_mb=24000.0,
            ram_budget_mb=120000.0,
            max_slices=5,
        )

    def test_stream_priorities_assignment(self):
        """
        Verify stream priorities: YOLO detector must have high priority (-1).
        """
        summary = self.governor.get_telemetry_summary()
        priorities = summary["stream_priorities"]
        self.assertEqual(priorities["yolo_detector"], -1, "YOLO detector must be high priority (priority=-1)")
        self.assertEqual(priorities["samurai_tracker"], 0, "SAMURAI tracker must be normal priority (priority=0)")
        self.assertEqual(priorities["background_io"], 1, "Background IO must be low priority (priority=1)")

    def test_calculate_concurrency_cap_rtx_a5000(self):
        """
        Verify dynamic concurrency cap calculation under different VRAM/RAM regimes.
        """
        # RTX A5000: 24GB VRAM, 128GB RAM -> Cap = 5 (configured max)
        cap_a5000 = self.governor.calculate_concurrency_cap(
            available_vram_mb=24000.0,
            available_ram_mb=128000.0,
        )
        self.assertEqual(cap_a5000, 5)

        # Constrained GPU (e.g. RTX 4070 12GB VRAM): VRAM headroom limits cap
        cap_constrained = self.governor.calculate_concurrency_cap(
            available_vram_mb=8000.0,
            available_ram_mb=64000.0,
        )
        self.assertLessEqual(cap_constrained, 2)

    def test_schedule_waves_balanced_partition(self):
        """
        Verify balanced wave scheduling across segments.
        """
        # 10 segments with cap 5 -> exactly 2 waves of 5
        waves_10 = self.governor.schedule_waves(total_segments=10, concurrency_cap=5)
        self.assertEqual(waves_10, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # 8 segments with cap 4 -> exactly 2 waves of 4
        waves_8 = self.governor.schedule_waves(total_segments=8, concurrency_cap=4)
        self.assertEqual(waves_8, [[0, 1, 2, 3], [4, 5, 6, 7]])

        # 7 segments with cap 4 -> 4 + 3 balanced
        waves_7 = self.governor.schedule_waves(total_segments=7, concurrency_cap=4)
        self.assertEqual(waves_7, [[0, 1, 2, 3], [4, 5, 6]])

    def test_should_admit_slice_throttles_on_yolo_fps_slump(self):
        """
        Verify that admission is throttled when YOLO FPS drops into contention zone.
        """
        # Normal healthy YOLO FPS: 115.0 -> admitted
        admit_ok, reason_ok = self.governor.should_admit_slice(
            current_active_slices=2,
            concurrency_cap=5,
            current_yolo_fps=115.0,
            target_min_yolo_fps=80.0,
        )
        self.assertTrue(admit_ok)
        self.assertEqual(reason_ok, "admitted")

        # Contention slump (38-43 FPS zone): 42.0 FPS -> throttled
        admit_throttled, reason_throttled = self.governor.should_admit_slice(
            current_active_slices=2,
            concurrency_cap=5,
            current_yolo_fps=42.0,
            target_min_yolo_fps=80.0,
        )
        self.assertFalse(admit_throttled)
        self.assertIn("yolo_fps_throttled", reason_throttled)

        # Capacity full: 5 active slices out of cap 5 -> capacity limit
        admit_cap, reason_cap = self.governor.should_admit_slice(
            current_active_slices=5,
            concurrency_cap=5,
            current_yolo_fps=110.0,
        )
        self.assertFalse(admit_cap)
        self.assertEqual(reason_cap, "capacity_limit_reached")

    def test_telemetry_summary_and_slice_completion(self):
        """
        Verify lifecycle telemetry tracking across slice executions.
        """
        self.governor.active_slices = 3
        self.governor.record_slice_completion(slice_id=1, duration_s=18.5, vram_peak_mb=2100.0, ram_peak_mb=8200.0)

        self.assertEqual(self.governor.active_slices, 2)
        self.assertEqual(self.governor.completed_slices, 1)

        summary = self.governor.get_telemetry_summary()
        self.assertEqual(summary["total_waves_logged"], 1)


if __name__ == "__main__":
    unittest.main()
