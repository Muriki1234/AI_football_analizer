import unittest
import time
import statistics
from server.pipeline.vertical_crop import compute_smooth_crop_centers, build_ffmpeg_crop_filter

class TestVerticalCrop(unittest.TestCase):
    def setUp(self):
        self.w = 1920
        self.h = 1080

    def test_even_dimension_constraint(self):
        # 1080 * 9 / 16 = 607.5 -> 607 -> must be even (606)
        coords = compute_smooth_crop_centers([(960.0, 540.0)], self.w, self.h)
        self.assertEqual(len(coords), 1)
        # Even width check: 1080 * (9/16) floored to even
        crop_w = int(self.h * (9.0 / 16.0))
        if crop_w % 2 != 0:
            crop_w -= 1
        self.assertEqual(crop_w % 2, 0)
        self.assertEqual(crop_w, 606)

    def test_empty_input(self):
        self.assertEqual(compute_smooth_crop_centers([], self.w, self.h), [])
        self.assertEqual(build_ffmpeg_crop_filter(606, 1080, []), "crop=606:1080:0:0")

    def test_boundary_clamping_left(self):
        # Target at far left x=10
        coords = compute_smooth_crop_centers([(10.0, 500.0)] * 5, self.w, self.h)
        for x, y in coords:
            self.assertEqual(x, 0)
            self.assertEqual(y, 0)

    def test_boundary_clamping_right(self):
        # Target at far right x=1910
        coords = compute_smooth_crop_centers([(1910.0, 500.0)] * 5, self.w, self.h)
        max_x = 1920 - 606 # 1314
        for x, y in coords:
            self.assertEqual(x, max_x)
            self.assertEqual(y, 0)

    def test_missing_target_interpolation(self):
        # Target seen at 500, then lost for 5 frames, then seen at 500
        targets = [(500.0, 500.0), None, None, None, None, (500.0, 500.0)]
        coords = compute_smooth_crop_centers(targets, self.w, self.h)
        self.assertEqual(len(coords), 6)
        # None of the coords should crash or be None
        for x, y in coords:
            self.assertIsInstance(x, int)
            self.assertGreaterEqual(x, 0)

    def test_smoothing_filter_dampens_jitter(self):
        # Oscillating noisy input
        raw = [(800.0 if i % 2 == 0 else 1000.0, 500.0) for i in range(50)]
        coords = compute_smooth_crop_centers(raw, self.w, self.h, smoothing_window=15)
        xs = [c[0] for c in coords[10:40]]
        # The variance of smoothed coords must be drastically lower than raw oscillation
        raw_xs = [t[0] for t in raw[10:40]]
        self.assertLess(statistics.stdev(xs), statistics.stdev(raw_xs) * 0.5)

    def test_ffmpeg_crop_filter_stationary(self):
        # Near stationary targets (variation < 20px)
        coords = [(500, 0)] * 10
        f = build_ffmpeg_crop_filter(606, 1080, coords)
        self.assertEqual(f, "crop=606:1080:500:0")

    def test_ffmpeg_crop_filter_dynamic(self):
        # Moving sequence across pitch
        coords = [(100 + i * 20, 0) for i in range(20)]
        f = build_ffmpeg_crop_filter(606, 1080, coords)
        self.assertTrue(f.startswith("crop=606:1080:"))
        self.assertIn("in_w-606", f)

    def test_performance_benchmark(self):
        # Benchmark 3000 frames (2 minutes of 25fps video)
        targets = [(float(i % 1500), 500.0) for i in range(3000)]
        t0 = time.perf_counter()
        coords = compute_smooth_crop_centers(targets, self.w, self.h, smoothing_window=15)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(len(coords), 3000)
        # Performance budget: 3000 frames must process in < 50ms (real-time is 120,000ms)
        self.assertLess(elapsed_ms, 50.0, f"Processing took {elapsed_ms:.2f}ms, expected < 50ms")

if __name__ == "__main__":
    unittest.main()
