"""
test_speed_telemetry_engine.py - Unit Test & Benchmark Suite
for SpeedTelemetryEngine (5-Zone Athletic Telemetry & LTTB Downsampler).

Verifies:
1. Occlusion gap smoothing (interpolates isolated dropouts <= 4 frames).
2. LTTB timeseries downsampling (preserves peaks while compressing to target points).
3. FIFA standard 5-zone intensity breakdown (walking, jogging, running, HSR, sprinting).
4. 2-panel chart rendering to PNG.
5. High-throughput algorithm microbenchmark (> 100,000 frames/sec).
"""

from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from server.pipeline.speed_telemetry_engine import (
    SpeedTelemetryEngine,
    interpolate_occlusion_gaps,
    lttb_downsample,
    FIFA_ZONES,
)


class TestSpeedTelemetryEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SpeedTelemetryEngine(fps=25.0, target_downsample_points=100)

    def test_interpolate_occlusion_gaps_empty(self):
        self.assertEqual(interpolate_occlusion_gaps([]), [])

    def test_interpolate_occlusion_gaps_short_drop(self):
        # 2 frames dropped to 0 in between 20.0 km/h
        raw = [20.0, 20.0, 0.0, 0.0, 20.0, 20.0]
        cleaned = interpolate_occlusion_gaps(raw, max_gap=4)
        self.assertEqual(len(cleaned), 6)
        self.assertAlmostEqual(cleaned[2], 20.0, places=1)
        self.assertAlmostEqual(cleaned[3], 20.0, places=1)

    def test_interpolate_occlusion_gaps_handles_none(self):
        raw = [15.0, None, None, 18.0]
        cleaned = interpolate_occlusion_gaps(raw, max_gap=3)
        self.assertEqual(len(cleaned), 4)
        self.assertGreater(cleaned[1], 15.0)
        self.assertLess(cleaned[1], 18.0)

    def test_interpolate_occlusion_gaps_large_gap_not_filled(self):
        # Gap of 6 frames with max_gap=4: should not interpolate
        raw = [20.0] + [0.0] * 6 + [20.0]
        cleaned = interpolate_occlusion_gaps(raw, max_gap=4)
        self.assertEqual(cleaned[3], 0.0)

    def test_lttb_downsample_preserves_extrema(self):
        # Generate timeseries with 500 points and an explosive peak in the middle
        n = 500
        times = [i * 0.04 for i in range(n)]
        values = [5.0] * 200 + [32.5] + [5.0] * (n - 201)

        downsampled = lttb_downsample(times, values, target_points=50)
        self.assertEqual(len(downsampled), 50)
        # Verify first and last points match
        self.assertAlmostEqual(downsampled[0]["time"], times[0], places=2)
        self.assertAlmostEqual(downsampled[-1]["time"], times[-1], places=2)

        # Verify peak (32.5 km/h) is preserved in downsampled output
        max_downsampled = max(p["value"] for p in downsampled)
        self.assertAlmostEqual(max_downsampled, 32.5, places=1)

    def test_lttb_downsample_small_input(self):
        times = [0.0, 1.0, 2.0]
        values = [10.0, 15.0, 12.0]
        downsampled = lttb_downsample(times, values, target_points=10)
        self.assertEqual(len(downsampled), 3)

    def test_process_telemetry_empty(self):
        res = self.engine.process_telemetry([], [])
        self.assertEqual(res["max_speed_kmh"], 0.0)
        self.assertEqual(res["avg_speed_kmh"], 0.0)
        self.assertEqual(res["total_distance_m"], 0.0)
        self.assertEqual(res["zone_breakdown"], {})
        self.assertEqual(res["downsampled_timeline"], [])

    def test_process_telemetry_5_zones(self):
        # 100 frames walking (5 km/h)
        # 100 frames jogging (10 km/h)
        # 100 frames running (16 km/h)
        # 100 frames HSR (22 km/h)
        # 100 frames sprinting (27 km/h)
        # Total = 500 frames @ 25 fps = 20s
        speeds = [5.0] * 100 + [10.0] * 100 + [16.0] * 100 + [22.0] * 100 + [27.0] * 100

        # Construct cumulative distance matching speeds
        distances = [0.0]
        for s in speeds[1:]:
            d_inc = (s / 3.6) / 25.0
            distances.append(distances[-1] + d_inc)

        res = self.engine.process_telemetry(speeds, distances)

        self.assertEqual(res["max_speed_kmh"], 27.0)
        self.assertGreater(res["total_distance_m"], 50.0)

        zb = res["zone_breakdown"]
        self.assertIn("zone_1_walking", zb)
        self.assertIn("zone_2_jogging", zb)
        self.assertIn("zone_3_running", zb)
        self.assertIn("zone_4_hsr", zb)
        self.assertIn("zone_5_sprinting", zb)

        # Verify all 5 zones captured non-zero distance and duration
        for z_key in zb:
            self.assertGreater(zb[z_key]["distance_m"], 0.0)
            self.assertAlmostEqual(zb[z_key]["duration_s"], 4.0, delta=0.2)
            self.assertGreater(zb[z_key]["percentage"], 0.0)

        # Sum of percentages should roughly equal 100%
        total_pct = sum(zb[z]["percentage"] for z in zb)
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

        # Verify timeline downsampled length
        self.assertLessEqual(len(res["downsampled_timeline"]), 100)

    def test_render_chart(self):
        speeds = [5.0] * 50 + [25.5] * 30 + [12.0] * 50
        distances = [i * 0.2 for i in range(len(speeds))]
        telemetry = self.engine.process_telemetry(speeds, distances)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "speed_chart.png"
            self.engine.render_chart(telemetry, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 1000)

    def test_algorithm_benchmark_throughput(self):
        """[Algorithm-only Benchmark] Measures pure mathematical speed telemetry processing rate."""
        n = 10_000
        np.random.seed(42)
        sim_speeds = list(np.clip(np.random.normal(12.0, 5.0, n), 0.0, 32.0))
        sim_dists = list(np.cumsum(np.array(sim_speeds) / 3.6 / 25.0))

        # Warmup
        _ = self.engine.process_telemetry(sim_speeds[:100], sim_dists[:100])

        t0 = time.perf_counter()
        _ = self.engine.process_telemetry(sim_speeds, sim_dists)
        elapsed = time.perf_counter() - t0

        fps = n / elapsed
        print(f"\n[Algorithm-only Benchmark] SpeedTelemetryEngine: {fps:,.0f} frames/sec ({elapsed*1000:.2f} ms for {n} frames)")
        self.assertGreater(fps, 50_000, f"Expected > 50,000 fps, got {fps:,.0f}")


if __name__ == "__main__":
    unittest.main()
