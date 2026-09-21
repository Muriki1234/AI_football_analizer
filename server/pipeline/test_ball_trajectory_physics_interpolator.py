"""
test_ball_trajectory_physics_interpolator.py - Unit Tests for BallTrajectoryPhysicsInterpolator
"""

import unittest
import numpy as np

from server.pipeline.ball_trajectory_physics_interpolator import BallTrajectoryPhysicsInterpolator


class TestBallTrajectoryPhysicsInterpolator(unittest.TestCase):

    def test_ball_trajectory_parabolic_interpolation(self):
        interpolator = BallTrajectoryPhysicsInterpolator(max_dropout_gap=6)

        # Ball detected at frame 0 and frame 4
        # Horizontal move: x: 100 -> 300 (50px/frame)
        # Vertical move: y: 200 -> 200 (parabolic arc)
        raw_tracks = [{} for _ in range(6)]
        raw_tracks[0][1] = {"bbox": [95.0, 195.0, 105.0, 205.0]}
        raw_tracks[4][1] = {"bbox": [295.0, 195.0, 305.0, 205.0]}

        smoothed = interpolator.interpolate_ball_trajectory(raw_tracks, total_frames=6)

        self.assertEqual(len(smoothed), 6)
        # Frame 0 and 4 should be original
        self.assertIn(1, smoothed[0])
        self.assertIn(1, smoothed[4])
        # Gaps at frame 1, 2, 3 should be filled
        self.assertIn(1, smoothed[1])
        self.assertIn(1, smoothed[2])
        self.assertIn(1, smoothed[3])

        # Check that x is moving forward smoothly
        cx0 = (smoothed[0][1]["bbox"][0] + smoothed[0][1]["bbox"][2]) / 2.0
        cx1 = (smoothed[1][1]["bbox"][0] + smoothed[1][1]["bbox"][2]) / 2.0
        cx2 = (smoothed[2][1]["bbox"][0] + smoothed[2][1]["bbox"][2]) / 2.0
        cx3 = (smoothed[3][1]["bbox"][0] + smoothed[3][1]["bbox"][2]) / 2.0
        cx4 = (smoothed[4][1]["bbox"][0] + smoothed[4][1]["bbox"][2]) / 2.0

        self.assertTrue(cx0 < cx1 < cx2 < cx3 < cx4)
        # Frame 2 is halfway between 100 and 300
        self.assertLess(abs(cx2 - 200.0), 5.0)

        # Due to gravity sag, y at midpoint should be slightly lower (higher pixel y)
        cy0 = (smoothed[0][1]["bbox"][1] + smoothed[0][1]["bbox"][3]) / 2.0
        cy2 = (smoothed[2][1]["bbox"][1] + smoothed[2][1]["bbox"][3]) / 2.0
        self.assertGreater(cy2, cy0)  # Sagged downwards

    def test_ball_trajectory_outlier_rejection(self):
        interpolator = BallTrajectoryPhysicsInterpolator(
            max_dropout_gap=6,
            max_physical_speed_px_per_frame=100.0,
        )

        raw_tracks = [{} for _ in range(5)]
        raw_tracks[0][1] = {"bbox": [100.0, 200.0, 110.0, 210.0]}
        # Outlier spike at frame 1: impossible jump to [1500, 900] (stadium billboard)
        raw_tracks[1][1] = {"bbox": [1500.0, 900.0, 1510.0, 910.0]}
        # Real ball at frame 2: normal continuation
        raw_tracks[2][1] = {"bbox": [130.0, 205.0, 140.0, 215.0]}

        smoothed = interpolator.interpolate_ball_trajectory(raw_tracks, total_frames=5)

        # Frame 1 should not contain the wild billboard coordinate
        cx1 = (smoothed[1][1]["bbox"][0] + smoothed[1][1]["bbox"][2]) / 2.0
        self.assertLess(cx1, 500.0)  # Rejected outlier


if __name__ == "__main__":
    unittest.main()
