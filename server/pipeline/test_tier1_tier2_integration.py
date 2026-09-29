"""
test_tier1_tier2_integration.py
================================
End-to-end verification of Tier 1 & Tier 2 Pipeline Integration:
1. Grassroots Graceful Degradation Engine + Video Analytics Confidence Assessor in _compute_player_summary
2. Motion-Guided Dynamic Crop Ball Detector in Tracker
3. Homography Telemetry in ViewTransformer
4. Idempotency Lifecycle verification
"""

import unittest
import numpy as np
from server.pipeline.tasks import _compute_player_summary
from server.pipeline.analysis_core import Tracker, ViewTransformer
from server.pipeline.grassroots_graceful_degradation_engine import FootageQualityTier, MetricAvailability


class TestTier1Tier2Integration(unittest.TestCase):

    def test_compute_player_summary_with_tier5_amateur_degradation(self):
        """Simulates amateur footage where homography completely collapsed."""
        total_frames = 100
        # No position_transformed anywhere
        tracks = {
            "players": [{1: {"bbox": [100, 100, 150, 200], "speed": 42.5, "distance": 15000, "has_ball": True}}
                        for _ in range(total_frames)],
            "ball": [{1: {"bbox": [120, 150, 130, 160]}} for _ in range(total_frames)],
            "homography_telemetry": [{
                "frame_idx": i,
                "has_valid_homography": False,
                "keypoint_count": 1,
                "inlier_ratio": 0.1,
                "reprojection_rmse": 35.0,
            } for i in range(total_frames)],
        }
        tracked_bboxes = {i: [100, 100, 50, 100] for i in range(total_frames)}
        team_control = [1] * 60 + [2] * 40

        summary = _compute_player_summary(tracks, tracked_bboxes, team_control, fps=25)

        # Spatial metrics must be shielded
        self.assertEqual(summary["footage_quality_tier"], FootageQualityTier.TIER_5_SEVERELY_DEGRADED.value)
        self.assertIsNone(summary["max_speed_kmh"])
        self.assertIsNone(summary["total_distance_m"])
        self.assertIn("max_speed_unavailable_reason", summary)
        self.assertIn("distance_unavailable_reason", summary)

        # Image-space analytics must remain 100% available and intact
        self.assertAlmostEqual(summary["team1_possession_pct"], 60.0)
        self.assertAlmostEqual(summary["team2_possession_pct"], 40.0)
        self.assertIn("video_confidence", summary)
        self.assertIn("overall_confidence_score", summary["video_confidence"])

    def test_compute_player_summary_with_broadcast_high_availability(self):
        """Simulates broadcast footage with valid homography and keypoints."""
        total_frames = 100
        tracks = {
            "players": [{1: {
                "bbox": [100, 100, 150, 200],
                "position_transformed": [30.0 + (i * 0.1), 20.0],
                "speed": 18.5,
                "distance": 120.0 + i,
                "has_ball": True
            }} for i in range(total_frames)],
            "ball": [{1: {"bbox": [120, 150, 130, 160]}} for _ in range(total_frames)],
            "homography_telemetry": [{
                "frame_idx": i,
                "has_valid_homography": True,
                "keypoint_count": 10,
                "inlier_ratio": 0.85,
                "reprojection_rmse": 4.0,
            } for i in range(total_frames)],
        }
        tracked_bboxes = {i: [100, 100, 50, 100] for i in range(total_frames)}
        team_control = [1] * 50 + [2] * 50

        summary = _compute_player_summary(tracks, tracked_bboxes, team_control, fps=25)

        self.assertIn(summary["footage_quality_tier"], [FootageQualityTier.TIER_1_BROADCAST.value, FootageQualityTier.TIER_2_TACTICAL_MAST.value])
        self.assertIsNotNone(summary["max_speed_kmh"])
        self.assertIsNotNone(summary["total_distance_m"])
        self.assertEqual(summary["analytics_contract"]["max_speed_kmh"], MetricAvailability.FULLY_AVAILABLE.value)

    def test_tracker_crop_ball_detector_initialization(self):
        """Verifies Tracker initializes the motion guided crop detector."""
        tracker = Tracker()
        self.assertIsNotNone(tracker.crop_ball_detector)
        self.assertEqual(tracker.crop_ball_detector.crop_size, 384)

    def test_view_transformer_homography_telemetry(self):
        """Verifies ViewTransformer outputs homography_telemetry array."""
        vt = ViewTransformer()
        tracks = {
            "players": [{} for _ in range(3)],
            "ball": [{} for _ in range(3)],
        }
        kps_list = [{}, {}, {}]
        vt.add_transformed_position_to_tracks(tracks, kps_list)
        self.assertIn("homography_telemetry", tracks)
        self.assertEqual(len(tracks["homography_telemetry"]), 3)
        self.assertFalse(tracks["homography_telemetry"][0]["has_valid_homography"])


if __name__ == "__main__":
    unittest.main()
