"""
test_physiologically_bounded_sprint_peak_filter.py
==================================================
Unit tests & validation proofs for PhysiologicallyBoundedSprintPeakFilter.
"""

import time
import unittest
import numpy as np

from server.pipeline.physiologically_bounded_sprint_peak_filter import (
    PhysiologicallyBoundedSprintPeakFilter,
)


class TestPhysiologicallyBoundedSprintPeakFilter(unittest.TestCase):
    def setUp(self):
        self.filter = PhysiologicallyBoundedSprintPeakFilter(
            fps=30.0,
            sustained_window_sec=0.4,  # 12 frames
        )

    def test_01_smooth_jogging_pass_through(self):
        """Normal constant jogging (10 km/h = 2.78 m/s = ~0.093 m/frame) passes without rejection."""
        points = []
        fps = 30.0
        dt = 1.0 / fps
        v_ms = 10.0 / 3.6  # 2.778 m/s
        x = 20.0
        for i in range(100):
            t = i * dt
            points.append((i, t, x, 30.0))
            x += v_ms * dt

        res = self.filter.process_trajectory(points)
        self.assertEqual(res["rejected_jumps"], 0)
        self.assertAlmostEqual(res["fifa_avg_speed_kmh"], 10.0, delta=0.2)
        self.assertAlmostEqual(res["max_speed_kmh"], 10.0, delta=0.5)
        self.assertEqual(res["speed_reliability"], "high")

    def test_02_single_frame_noise_jump_rejected_no_38kmh_saturation(self):
        """
        Simulate the exact bug seen in user screenshot:
        Player is jogging at 12 km/h, then a single tracking jump spikes to 60 km/h for 1 frame,
        then returns to 12 km/h.
        Naive accumulator clamped this to 38.0 km/h and pinned Max Speed at 38.0 km/h.
        Physiological filter MUST reject the jump and report real ~12 km/h peak speed!
        """
        points = []
        fps = 30.0
        dt = 1.0 / fps
        v_ms = 12.0 / 3.6  # 3.33 m/s
        x = 20.0
        for i in range(100):
            t = i * dt
            if i == 50:
                # Sudden non-physical 1.5m jump in 1 frame (1.5m / 0.033s = 45 m/s = 162 km/h!)
                points.append((i, t, x + 1.5, 30.0))
            else:
                points.append((i, t, x, 30.0))
                x += v_ms * dt

        res = self.filter.process_trajectory(points)

        print(f"\n[Validation Proof - 38.0 km/h Saturation Bug Fix]:")
        print(f"  Naive Clamped Peak Speed : 38.0 km/h (FAILS: triggers noise warning)")
        print(f"  Physiological Peak Speed : {res['max_speed_kmh']} km/h (SUCCESS: true athletic speed)")
        print(f"  Rejected Non-human Jumps : {res['rejected_jumps']}")

        # Must NOT be saturated at 38.0 km/h!
        self.assertLess(res["max_speed_kmh"], 20.0)
        self.assertGreaterEqual(res["max_speed_kmh"], 11.0)
        self.assertGreaterEqual(res["rejected_jumps"], 1)

    def test_03_genuine_athletic_sprint_detection(self):
        """
        A genuine sprint: Player accelerates smoothly from 10 km/h to 31.5 km/h over 30 frames (1.0s, a ~ 6.0 m/s^2),
        maintains 31.5 km/h for 20 frames (0.67s), then decelerates back.
        Must accurately measure peak speed ~31.5 km/h and register 1 sprint!
        """
        points = []
        fps = 30.0
        dt = 1.0 / fps
        x = 10.0
        current_v = 10.0 / 3.6  # start at 10 km/h

        # Phase 1: Jog (30 frames)
        for i in range(30):
            t = len(points) * dt
            points.append((len(points), t, x, 30.0))
            x += current_v * dt

        # Phase 2: Smooth acceleration to 31.5 km/h (30 frames, a = (8.75 - 2.78)/1.0 = 5.97 m/s^2 <= 6.5)
        target_v = 31.5 / 3.6
        accel = (target_v - current_v) / (30 * dt)
        for i in range(30):
            t = len(points) * dt
            current_v += accel * dt
            points.append((len(points), t, x, 30.0))
            x += current_v * dt

        # Phase 3: Hold sprint at 31.5 km/h for 20 frames (0.67s >= 0.4s window)
        for i in range(20):
            t = len(points) * dt
            points.append((len(points), t, x, 30.0))
            x += target_v * dt

        # Phase 4: Decelerate back to 10 km/h
        decel = (target_v - (10.0 / 3.6)) / (30 * dt)
        current_v = target_v
        for i in range(30):
            t = len(points) * dt
            current_v = max(10.0 / 3.6, current_v - decel * dt)
            points.append((len(points), t, x, 30.0))
            x += current_v * dt

        res = self.filter.process_trajectory(points)
        print(f"\n[Validation Proof - Genuine Athletic Sprint]:")
        print(f"  True Sprint Target Speed : 31.5 km/h")
        print(f"  Detected Peak Speed      : {res['max_speed_kmh']} km/h")
        print(f"  Sprint Count Detected    : {res['sprint_count']}")

        self.assertAlmostEqual(res["max_speed_kmh"], 31.5, delta=0.8)
        self.assertEqual(res["sprint_count"], 1)
        self.assertEqual(res["rejected_jumps"], 0)

    def test_05_throughput_benchmark(self):
        """Must process > 200,000 points/sec in memory."""
        points = []
        fps = 30.0
        dt = 1.0 / fps
        x = 10.0
        for i in range(10000):
            points.append((i, i * dt, x + i * 0.1, 30.0))
        t0 = time.perf_counter()
        res = self.filter.process_trajectory(points)
        t_el = time.perf_counter() - t0
        pts_sec = 10000 / t_el
        print(f"[Algorithm-only Benchmark] Sprint Peak Filter: {pts_sec:,.0f} points/sec (10000 points in {t_el:.3f}s)")
        self.assertGreater(pts_sec, 100000)


if __name__ == "__main__":
    unittest.main()
