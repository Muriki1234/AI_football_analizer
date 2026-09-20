"""
test_speed_kinematics_filter.py — Unit Tests & Kinematic Benchmarks for RobustKinematicSpeedEstimator
"""

import time
import unittest
import numpy as np

from server.pipeline.speed_kinematics_filter import (
    RobustKinematicSpeedEstimator,
    _savgol_coefficients,
)
def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


class NaiveSpeedEstimator:
    """Reference implementation of the naive estimator from analysis_core to demonstrate the 5x bug."""
    def __init__(self, fps: float = 24.0, frame_window: int = 5, max_speed: float = 34.0):
        self.fps = float(fps) if fps and fps > 0 else 24.0
        self.frame_window = frame_window
        self.max_speed = max_speed
        self._history = {}

    def add_speed_and_distance_to_tracks(self, tracks: dict):
        total_dist = {}
        for obj, otracks in tracks.items():
          if obj in ("ball", "referees"): continue
          for fi in range(len(otracks)):
            prev = max(0, fi - self.frame_window)
            for tid, info in otracks[fi].items():
              if tid not in otracks[prev]: continue
              cp = info.get("position_transformed")
              pp = otracks[prev][tid].get("position_transformed")
              if not cp or not pp: continue
              dist = measure_distance(cp, pp)
              elapsed = (fi - prev) / self.fps
              if elapsed == 0: continue
              raw = (dist / elapsed) * 3.6
              if raw > self.max_speed:
                continue
              self._history.setdefault(tid, []).append(raw)
              if len(self._history[tid]) > 7: self._history[tid].pop(0)
              smoothed = float(np.median(self._history[tid]))
              total_dist.setdefault(obj, {}).setdefault(tid, 0)
              total_dist[obj][tid] += dist
              info["speed"] = smoothed
              info["distance"] = total_dist[obj][tid]


class TestSpeedKinematicsFilter(unittest.TestCase):
    def setUp(self):
        self.estimator = RobustKinematicSpeedEstimator(
            fps=25.0, sg_window=7, sg_poly=2, deadband_kmh=2.0, max_speed_kmh=38.0
        )

    def test_01_savgol_coefficients_properties(self):
        """Coefficients must be symmetric and sum exactly to 1.0 (constant conservation)."""
        coeffs = _savgol_coefficients(7, 2)
        self.assertEqual(len(coeffs), 7)
        self.assertAlmostEqual(float(np.sum(coeffs)), 1.0, places=6)
        np.testing.assert_allclose(coeffs, coeffs[::-1], rtol=1e-6)

    def test_02_linear_distance_accuracy_no_oversumming(self):
        """
        Player moves 100.0 meters in a straight line at 5.0 m/s (18 km/h) over 500 frames.
        RobustKinematicSpeedEstimator must report ~100.0m.
        Demonstrates that naive AccurateSpeedEstimator oversums by ~5x.
        """
        fps = 25.0
        n_frames = 500
        step_per_frame = 100.0 / (n_frames - 1)  # 0.2004 m/frame = 5.01 m/s = 18.0 km/h

        frame_pos = {i: (i * step_per_frame, 0.0) for i in range(n_frames)}
        res = self.estimator.compute_player_kinematics(frame_pos)

        final_dist = res[n_frames - 1]["distance"]
        avg_speed = res[n_frames - 1]["speed"]

        self.assertAlmostEqual(final_dist, 100.0, delta=1.5)
        self.assertAlmostEqual(avg_speed, 18.0, delta=1.5)

        # Compare with naive estimator behavior on identical tracks
        tracks = {"players": [{1: {"position_transformed": (i * step_per_frame, 0.0)}} for i in range(n_frames)]}
        naive = NaiveSpeedEstimator(fps=fps, frame_window=5)
        naive.add_speed_and_distance_to_tracks(tracks)
        naive_dist = tracks["players"][-1][1]["distance"]

        self.assertGreater(naive_dist, 400.0)
        print(f"\n[Validation Proof - Oversumming Fix]: True=100.0m | Robust={final_dist:.1f}m | Naive={naive_dist:.1f}m (5x error)")

    def test_03_stationary_zero_drift_deadband(self):
        """
        Player is standing completely still at (10.0, 5.0) with ±0.03m optical tracking noise.
        Velocity deadband (<1.5 km/h) must prevent random-walk distance accumulation.
        """
        np.random.seed(42)
        n_frames = 1000
        noise_x = np.random.normal(0.0, 0.02, n_frames)
        noise_y = np.random.normal(0.0, 0.02, n_frames)

        frame_pos = {i: (10.0 + noise_x[i], 5.0 + noise_y[i]) for i in range(n_frames)}
        res = self.estimator.compute_player_kinematics(frame_pos)

        accumulated_dist = res[n_frames - 1]["distance"]
        max_speed = max(r["speed"] for r in res.values())

        # With deadband, accumulated distance over 1000 stationary frames must be minimal (< 1.0m)
        self.assertLess(accumulated_dist, 1.0)
        self.assertLess(max_speed, 3.5)

    def test_04_sprint_peak_detection_preservation(self):
        """
        Player accelerates from jog (10 km/h) to sprint (31 km/h) and decelerates.
        Savitzky-Golay smoothing must preserve the 31 km/h peak and trigger is_sprint=True.
        """
        fps = 25.0
        n_frames = 100
        coords = []
        cur_x = 0.0
        for i in range(n_frames):
            if i < 30:
                v = 2.77
            elif i < 70:
                progress = (i - 30) / 40.0
                v = 2.77 + (8.6 - 2.77) * np.sin(progress * np.pi)
            else:
                v = 2.77
            cur_x += v / fps
            coords.append((cur_x, 0.0))

        frame_pos = {i: coords[i] for i in range(n_frames)}
        res = self.estimator.compute_player_kinematics(frame_pos)

        peak_speed = max(r["speed"] for r in res.values())
        sprint_frames = sum(1 for r in res.values() if r["is_sprint"])

        self.assertGreaterEqual(peak_speed, 30.0)
        self.assertGreater(sprint_frames, 15)

    def test_05_tracking_teleport_speed_rejection(self):
        """A 30-meter single-frame teleport jump (tracking ID swap) must be clamped and rejected."""
        coords = [(float(i * 0.2), 0.0) for i in range(10)]
        coords.append((40.0, 0.0))  # Teleport jump
        for i in range(11, 20):
            coords.append((40.0 + (i - 10) * 0.2, 0.0))

        frame_pos = {i: coords[i] for i in range(len(coords))}
        res = self.estimator.compute_player_kinematics(frame_pos)

        teleport_speed = res[10]["speed"]
        self.assertLessEqual(teleport_speed, 38.0)

    def test_06_drop_in_tracks_compatibility(self):
        """Verifies full compatibility when mutating standard pipeline tracks dictionary."""
        # 50 frames moving 0.20m/frame = 18 km/h = 9.8m total
        tracks = {
            "players": [
                {
                    1: {"position_transformed": (float(i * 0.20), 0.0)},
                    2: {"position_transformed": (0.0, float(i * 0.10))},
                }
                for i in range(50)
            ],
            "ball": [{1: {"bbox": [0, 0, 10, 10]}} for _ in range(50)],
        }
        self.estimator.add_speed_and_distance_to_tracks(tracks)

        p1_end = tracks["players"][-1][1]
        p2_end = tracks["players"][-1][2]

        self.assertIn("speed", p1_end)
        self.assertIn("distance", p1_end)
        self.assertIn("is_sprint", p1_end)
        self.assertAlmostEqual(p1_end["distance"], 9.8, delta=1.0)
        self.assertAlmostEqual(p2_end["distance"], 4.9, delta=1.0)

    def test_07_algorithm_only_benchmark_throughput(self):
        """
        [Algorithm-only Benchmark: Kinematic Trajectory Smoothing & Distance Integration]:
        Evaluates throughput over 25,000 frames across 22 players (full 14-min match tracking).
        """
        n_frames = 25000
        n_players = 22

        tracks = {
            "players": [
                {
                    pid: {"position_transformed": (float((i * 0.1) % 100), float(pid * 2))}
                    for pid in range(1, n_players + 1)
                }
                for i in range(n_frames)
            ]
        }

        t0 = time.perf_counter()
        self.estimator.add_speed_and_distance_to_tracks(tracks)
        t_total = time.perf_counter() - t0

        fps = n_frames / max(1e-5, t_total)
        total_evals = n_frames * n_players
        evals_per_sec = total_evals / max(1e-5, t_total)

        print(f"\n[Algorithm-only Benchmark: Kinematic Trajectory Smoothing & Distance Integration]:")
        print(f"  Frames Processed      : {n_frames:,} (22 players, {total_evals:,} evaluations)")
        print(f"  Total processing time : {t_total*1000:.2f}ms")
        print(f"  Throughput            : {fps:,.0f} video FPS ({evals_per_sec:,.0f} point-kinematics/sec)")

        self.assertGreater(fps, 5000.0)


if __name__ == "__main__":
    unittest.main()
