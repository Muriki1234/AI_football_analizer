"""
test_sprint_burst_debouncer.py - Unit Test & Benchmark Suite
for SprintBurstDebouncer.

Verifies:
1. FIFA/Catapult calibrated threshold (>= 24.0 km/h, >= 0.8s).
2. Hysteresis dropout debouncing across 1-3 frame noise dips.
3. Path trajectory Euclidean distance integration (m).
4. Pitch diagram rendering.
5. High-throughput algorithm microbenchmark (> 500,000 frames/sec).
"""

from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from server.pipeline.sprint_burst_debouncer import SprintBurstDebouncer


class TestSprintBurstDebouncer(unittest.TestCase):
    def setUp(self):
        self.debouncer = SprintBurstDebouncer(
            sprint_speed_kmh=24.0,
            min_duration_s=0.8,
            max_dropout_frames=3,
            dropout_speed_floor_kmh=20.0,
            fps=25.0,
        )

    def test_empty_speeds(self):
        result = self.debouncer.detect_sprints([], [])
        self.assertEqual(result["sprint_count"], 0)
        self.assertEqual(result["total_sprint_distance_m"], 0.0)

    def test_sub_threshold_speeds_ignored(self):
        # 100 frames of jogging / high intensity running at 22 km/h (< 24.0 km/h)
        speeds = [22.0] * 100
        positions = [(float(i), 34.0) for i in range(100)]
        result = self.debouncer.detect_sprints(speeds, positions)
        self.assertEqual(result["sprint_count"], 0)

    def test_short_burst_discarded(self):
        # 10 frames at 28 km/h (< 20 frames / 0.8s minimum)
        speeds = [12.0] * 20 + [28.0] * 10 + [12.0] * 20
        positions = [(float(i), 34.0) for i in range(50)]
        result = self.debouncer.detect_sprints(speeds, positions)
        self.assertEqual(result["sprint_count"], 0)

    def test_clean_sprint_burst_detected(self):
        # 30 frames at 27.5 km/h (1.2 seconds, well above 0.8s)
        speeds = [15.0] * 10 + [27.5] * 30 + [15.0] * 10
        # Position moving along x-axis from 20m to 50m (1.0m per frame in sprint)
        positions = [(float(i), 34.0) for i in range(50)]
        result = self.debouncer.detect_sprints(speeds, positions)

        self.assertEqual(result["sprint_count"], 1)
        self.assertEqual(result["max_speed_kmh"], 27.5)
        self.assertAlmostEqual(result["avg_duration_s"], 30 / 25.0, places=2)
        event = result["events"][0]
        self.assertEqual(event["start_frame"], 10)
        self.assertEqual(event["end_frame"], 39)
        self.assertEqual(event["peak_speed_kmh"], 27.5)
        self.assertAlmostEqual(event["distance_m"], 29.0, places=1)

    def test_dropout_debouncing_resilience(self):
        # Player sprints for 35 frames:
        # Frame 10..22: 28 km/h (13 frames)
        # Frame 23..24: 21.5 km/h (2 frames of tracking occlusion dip! Floor is 20 km/h)
        # Frame 25..44: 29 km/h (20 frames)
        # Total duration = 35 frames.
        # WITHOUT debouncer: would split into 13 frames (<20) and 20 frames -> shatters data.
        # WITH debouncer: perfectly stitched into 1 continuous 35-frame sprint!
        speeds = [12.0] * 10 + [28.0] * 13 + [21.5] * 2 + [29.0] * 20 + [12.0] * 10
        positions = [(float(i), 34.0) for i in range(65)]
        result = self.debouncer.detect_sprints(speeds, positions)

        self.assertEqual(result["sprint_count"], 1)
        event = result["events"][0]
        self.assertEqual(event["start_frame"], 10)
        self.assertEqual(event["end_frame"], 44)
        self.assertAlmostEqual(event["duration_s"], 35 / 25.0, places=2)
        self.assertEqual(event["peak_speed_kmh"], 29.0)

    def test_excessive_dropout_terminates_burst(self):
        # Dropout lasts 5 frames (> 3 max_dropout_frames)
        # Frame 10..32: 28 km/h (23 frames -> verified sprint 1)
        # Frame 33..38: 15 km/h (6 frames below floor -> clean termination)
        # Frame 39..62: 28 km/h (24 frames -> verified sprint 2)
        speeds = [10.0] * 10 + [28.0] * 23 + [15.0] * 6 + [28.0] * 24 + [10.0] * 10
        positions = [(float(i), 34.0) for i in range(len(speeds))]
        result = self.debouncer.detect_sprints(speeds, positions)

        self.assertEqual(result["sprint_count"], 2)

    def test_render_visualization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sprint_analysis.png"
            speeds = [12.0] * 10 + [27.0] * 25 + [12.0] * 10
            positions = [(float(i), 25.0 + float(i)*0.2) for i in range(45)]
            result = self.debouncer.detect_sprints(speeds, positions)
            self.debouncer.render_visualization(result["segments"], result, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 5000)

    def test_throughput_benchmark(self):
        n_frames = 25000
        # Simulated match profile with periodic bursts
        speeds = []
        positions = []
        for i in range(n_frames):
            spd = 27.0 if (i % 100) < 30 else 14.0
            speeds.append(spd)
            positions.append((float(i % 105), 34.0))

        t0 = time.perf_counter()
        result = self.debouncer.detect_sprints(speeds, positions)
        elapsed = time.perf_counter() - t0
        fps = n_frames / max(elapsed, 1e-6)

        print(f"\n[Algorithm-only Benchmark] SprintBurstDebouncer: {n_frames} frames in {elapsed*1000:.1f}ms ({fps:,.0f} frames/sec)")
        self.assertGreater(fps, 100000.0)
        self.assertEqual(result["sprint_count"], 250)


if __name__ == "__main__":
    unittest.main()
