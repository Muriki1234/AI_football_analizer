"""
test_target_player_continuous_kinematic_accumulator.py
======================================================
Unit tests for TargetPlayerContinuousKinematicAccumulator.
"""

import unittest
from server.pipeline.target_player_continuous_kinematic_accumulator import (
    TargetPlayerContinuousKinematicAccumulator,
)


class TestTargetPlayerContinuousKinematicAccumulator(unittest.TestCase):
    def test_continuous_accumulation_across_id_switches(self):
        """
        Verify distance continues accumulating smoothly when ByteTrack switches track IDs.
        """
        acc = TargetPlayerContinuousKinematicAccumulator(fps=30.0)
        # Player runs at constant speed 5 m/s (18 km/h) for 30 seconds (900 frames)
        # Track ID changes: 1 -> 2 -> 3 every 300 frames
        dt = 1.0 / 30.0
        v = 5.0  # m/s

        for f in range(900):
            t = f * dt
            track_id = 1 if f < 300 else (2 if f < 600 else 3)
            x = v * t
            y = 34.0  # along pitch midline
            acc.add_point(frame_idx=f, timestamp_s=t, x_m=x, y_m=y, track_id=track_id)

        summary = acc.get_summary()
        # Expected distance: 5 m/s * 30s = 150m (minus first frame step)
        self.assertAlmostEqual(summary["total_distance_m"], 149.83, delta=1.0)
        self.assertAlmostEqual(summary["fifa_avg_speed_kmh"], 18.0, delta=0.5)

    def test_jitter_deadband_suppresses_stationary_wobble(self):
        """
        Verify that sub-4cm/sub-1.0km/h homography noise does NOT accumulate ghost distance.
        """
        acc = TargetPlayerContinuousKinematicAccumulator(fps=30.0)
        dt = 1.0 / 30.0

        # Player stands still at (50.0, 30.0) with random 1cm camera jitter
        import random
        random.seed(42)

        for f in range(300):  # 10 seconds
            t = f * dt
            jitter_x = (random.random() - 0.5) * 0.02  # +/- 1 cm
            jitter_y = (random.random() - 0.5) * 0.02
            acc.add_point(frame_idx=f, timestamp_s=t, x_m=50.0 + jitter_x, y_m=30.0 + jitter_y, track_id=10)

        summary = acc.get_summary()
        self.assertEqual(summary["total_distance_m"], 0.0, "Stationary jitter must accumulate 0.0 meters")
        self.assertEqual(summary["fifa_avg_speed_kmh"], 0.0)
        self.assertGreater(summary["jitter_suppressed_frames"], 250)

    def test_fifa_vs_active_moving_speed(self):
        """
        Verify true FIFA match average speed (Total Dist / Total Time) vs Active Moving Speed.
        """
        acc = TargetPlayerContinuousKinematicAccumulator(fps=30.0)
        dt = 1.0 / 30.0

        # Phase 1: Run 100m in 10s (300 frames) at 10 m/s (36 km/h)
        for f in range(300):
            t = f * dt
            acc.add_point(frame_idx=f, timestamp_s=t, x_m=10.0 * t, y_m=20.0, track_id=5)

        # Phase 2: Stand still for 30s (900 frames)
        for f in range(300, 1200):
            t = f * dt
            acc.add_point(frame_idx=f, timestamp_s=t, x_m=100.0, y_m=20.0, track_id=5)

        summary = acc.get_summary()
        self.assertAlmostEqual(summary["total_distance_m"], 100.0, delta=1.5)
        # Total time = 40s. FIFA match speed = 100m / 40s = 2.5 m/s = 9.0 km/h
        self.assertAlmostEqual(summary["fifa_avg_speed_kmh"], 9.0, delta=0.5)
        # Active moving speed = 100m / 10s = 10.0 m/s = 36.0 km/h
        self.assertAlmostEqual(summary["active_moving_avg_speed_kmh"], 36.0, delta=0.8)

    def test_unphysical_velocity_clamping(self):
        """
        Verify detector teleportation spikes (>38 km/h) are clamped.
        """
        acc = TargetPlayerContinuousKinematicAccumulator(fps=30.0)
        dt = 1.0 / 30.0

        # Normal frame 0
        acc.add_point(frame_idx=0, timestamp_s=0.0, x_m=10.0, y_m=10.0, track_id=7)
        # Teleportation spike: 100 meters jump in 1 frame (3000 m/s = 10800 km/h!)
        acc.add_point(frame_idx=1, timestamp_s=dt, x_m=110.0, y_m=10.0, track_id=7)

        summary = acc.get_summary()
        self.assertEqual(summary["velocity_clamped_frames"], 1)
        # Clamped distance should be 38 km/h * dt = (38 / 3.6) * (1/30) = 0.352 meters
        self.assertAlmostEqual(summary["total_distance_m"], 0.35, delta=0.05)

    def test_fifa_5_zone_distribution_and_sprint_count(self):
        """
        Verify 5-zone speed segmentation and sprint burst counter.
        """
        acc = TargetPlayerContinuousKinematicAccumulator(fps=30.0)
        dt = 1.0 / 30.0
        frame = 0
        cur_x = 0.0

        # Zone 1 Walking (5.0 km/h = 1.39 m/s) for 5s (150 frames)
        for _ in range(150):
            t = frame * dt
            cur_x += 1.39 * dt
            acc.add_point(frame_idx=frame, timestamp_s=t, x_m=cur_x, y_m=10.0)
            frame += 1

        # Zone 5 Sprint 1 (28.0 km/h = 7.78 m/s) for 1.0s (30 frames)
        for _ in range(30):
            t = frame * dt
            cur_x += 7.78 * dt
            acc.add_point(frame_idx=frame, timestamp_s=t, x_m=cur_x, y_m=10.0)
            frame += 1

        # Jogging recovery (10.0 km/h = 2.78 m/s) for 5s (150 frames)
        for _ in range(150):
            t = frame * dt
            cur_x += 2.78 * dt
            acc.add_point(frame_idx=frame, timestamp_s=t, x_m=cur_x, y_m=10.0)
            frame += 1

        # Zone 5 Sprint 2 (30.0 km/h = 8.33 m/s) for 1.0s (30 frames)
        for _ in range(30):
            t = frame * dt
            cur_x += 8.33 * dt
            acc.add_point(frame_idx=frame, timestamp_s=t, x_m=cur_x, y_m=10.0)
            frame += 1

        summary = acc.get_summary()
        self.assertEqual(summary["sprint_count"], 2, "Should detect exactly 2 distinct sprint efforts")
        self.assertGreater(summary["speed_zones_m"]["walking"], 5.0)
        self.assertGreater(summary["speed_zones_m"]["sprinting"], 14.0)
        self.assertGreater(summary["speed_zones_m"]["jogging"], 10.0)


if __name__ == "__main__":
    unittest.main()
