"""
test_heatmap_stabilizer.py — Unit Tests & Algorithm Benchmarks for Heatmap Spatial-Temporal Fidelity
Uses standard library unittest.
"""

import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from server.pipeline.heatmap_stabilizer import (
    HeatmapSpatialStabilizer,
    compute_density_grid,
    detect_coordinate_system,
    interpolate_trajectory,
    normalize_coordinates,
)


class TestHeatmapSpatialStabilizer(unittest.TestCase):
    def test_detect_coordinate_system(self):
        # Centimeters
        cm_pts = np.array([[6000.0, 3500.0], [12000.0, 7000.0]])
        self.assertEqual(detect_coordinate_system(cm_pts), "cm")

        # Meters
        m_pts = np.array([[52.5, 34.0], [105.0, 68.0]])
        self.assertEqual(detect_coordinate_system(m_pts), "m")

        # Normalized
        norm_pts = np.array([[0.5, 0.5], [1.0, 1.0]])
        self.assertEqual(detect_coordinate_system(norm_pts), "norm")

        # Empty
        self.assertEqual(detect_coordinate_system(np.empty((0, 2))), "m")

    def test_normalize_coordinates_conversions(self):
        # Centimeters to meters
        cm_pts = np.array([[12000.0, 7000.0], [6000.0, 3500.0]])
        m_res = normalize_coordinates(cm_pts, target_unit="m")
        self.assertTrue(np.allclose(m_res[0], [105.0, 68.0], atol=1e-2))
        self.assertTrue(np.allclose(m_res[1], [52.5, 34.0], atol=1e-2))

        # Meters to centimeters
        m_pts = np.array([[105.0, 68.0], [52.5, 34.0]])
        cm_res = normalize_coordinates(m_pts, target_unit="cm")
        self.assertTrue(np.allclose(cm_res[0], [12000.0, 7000.0], atol=1e-1))
        self.assertTrue(np.allclose(cm_res[1], [6000.0, 3500.0], atol=1e-1))

        # Normalized to meters
        norm_pts = np.array([[0.5, 0.5]])
        m_res2 = normalize_coordinates(norm_pts, target_unit="m")
        self.assertTrue(np.allclose(m_res2[0], [52.5, 34.0], atol=1e-2))

    def test_normalize_coordinates_nan_and_outlier_clamping(self):
        pts_with_nan = np.array([[52.5, 34.0], [np.nan, 20.0], [10.0, np.inf], [20.0, 30.0]])
        cleaned = normalize_coordinates(pts_with_nan, target_unit="m")
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(np.allclose(cleaned[0], [52.5, 34.0]))
        self.assertTrue(np.allclose(cleaned[1], [20.0, 30.0]))

        # Outlier clamping
        wild_pts = np.array([[200.0, -50.0]])
        clamped = normalize_coordinates(wild_pts, target_unit="m", clip_pitch=True, margin_ratio=0.05)
        self.assertLessEqual(clamped[0, 0], 105.0 * 1.05 + 1e-4)
        self.assertGreaterEqual(clamped[0, 1], -68.0 * 0.05 - 1e-4)

    def test_trajectory_gap_interpolation(self):
        # Small gap of 4 frames: frame 10 (0, 0) to frame 14 (40, 20)
        frame_pts = {10: [0.0, 0.0], 14: [40.0, 20.0]}
        interp = interpolate_trajectory(frame_pts, max_gap_frames=10)
        self.assertEqual(len(interp), 5)  # frames 10, 11, 12, 13, 14
        self.assertEqual(interp[0], (10, 0.0, 0.0))
        self.assertEqual(interp[2], (12, 20.0, 10.0))
        self.assertEqual(interp[4], (14, 40.0, 20.0))

        # Large gap of 50 frames (> max_gap_frames=30)
        large_gap_pts = {10: [0.0, 0.0], 60: [50.0, 30.0]}
        interp_large = interpolate_trajectory(large_gap_pts, max_gap_frames=30)
        self.assertEqual(len(interp_large), 2)  # Not bridged across long gaps

    def test_compute_density_grid(self):
        pts = np.random.normal(loc=[52.5, 34.0], scale=[2.0, 2.0], size=(200, 2))
        pts[:, 0] = np.clip(pts[:, 0], 0, 105)
        pts[:, 1] = np.clip(pts[:, 1], 0, 68)

        density = compute_density_grid(pts, grid_res=(105, 68), sigma_meters=3.5)
        self.assertEqual(density.shape, (68, 105))
        self.assertGreaterEqual(density.min(), 0.0)
        self.assertTrue(np.isclose(density.max(), 1.0))

        # Peak density should be near center
        max_idx = np.unravel_index(np.argmax(density), density.shape)
        peak_y, peak_x = max_idx
        self.assertTrue(40 <= peak_x <= 65)
        self.assertTrue(25 <= peak_y <= 45)

    def test_stabilizer_end_to_end_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_heatmap.png"
            stabilizer = HeatmapSpatialStabilizer()

            track_dict = {}
            for f in range(0, 300, 3):  # 10Hz sampling of 30FPS video (gap=3)
                x_m = 15.0 + (f / 300.0) * 45.0
                y_m = 34.0 + 5.0 * math.sin(f / 20.0)
                track_dict[f] = [x_m, y_m]

            success = stabilizer.render(track_dict, out_file)
            self.assertTrue(success)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)

    def test_empty_and_sparse_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "empty_heatmap.png"
            stabilizer = HeatmapSpatialStabilizer()
            success = stabilizer.render([], out_file)
            self.assertTrue(success)
            self.assertTrue(out_file.exists())

    def test_heatmap_stabilizer_benchmark(self):
        stabilizer = HeatmapSpatialStabilizer()
        np.random.seed(42)
        test_frames = {i: [float(np.random.uniform(10, 95)), float(np.random.uniform(5, 63))] for i in range(0, 3000, 5)}

        t0 = time.perf_counter()
        points_m, points_cm = stabilizer.prepare_points(test_frames)
        density = compute_density_grid(points_m)
        t_total = time.perf_counter() - t0

        fps = len(test_frames) / t_total
        latency_ms = (t_total / len(test_frames)) * 1000.0

        print(f"\n[Algorithm-only Benchmark] HeatmapSpatialStabilizer:")
        print(f"  Processed {len(test_frames)} sampled frames ({len(points_m)} interpolated points) in {t_total*1000:.2f}ms")
        print(f"  Throughput: {fps:.0f} input FPS | Latency: {latency_ms:.4f}ms / frame")

        self.assertGreater(fps, 500)


if __name__ == "__main__":
    unittest.main()
