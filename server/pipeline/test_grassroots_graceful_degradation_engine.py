"""
test_grassroots_graceful_degradation_engine.py
==============================================
Unit tests for the 5-Tier Footage Quality & Graceful Degradation Contract
"""

import unittest
from server.pipeline.grassroots_graceful_degradation_engine import (
    GrassrootsGracefulDegradationEngine,
    FootageQualityTier,
    MetricAvailability,
)


class TestGrassrootsGracefulDegradationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GrassrootsGracefulDegradationEngine()

    def test_tier_1_broadcast_classification_and_full_availability(self):
        tier = self.engine.classify_footage_tier(
            homography_completeness_pct=98.5,
            avg_keypoint_count=9.2,
            camera_jitter_px=3.2,
            ball_coverage_pct=65.0,
        )
        self.assertEqual(tier, FootageQualityTier.TIER_1_BROADCAST)

        contract = self.engine.evaluate_metric_availability(tier, 98.5, max_staleness_frames=10)
        self.assertEqual(contract["max_speed_kmh"]["status"], MetricAvailability.FULLY_AVAILABLE)
        self.assertEqual(contract["total_distance_m"]["status"], MetricAvailability.FULLY_AVAILABLE)
        self.assertEqual(contract["2d_minimap"]["status"], MetricAvailability.FULLY_AVAILABLE)
        self.assertEqual(contract["team_possession_pct"]["status"], MetricAvailability.FULLY_AVAILABLE)

    def test_tier_4_pitchside_handheld_triggers_reduced_accuracy(self):
        # Low elevation, camera shake, moderate homography completeness
        tier = self.engine.classify_footage_tier(
            homography_completeness_pct=52.0,
            avg_keypoint_count=4.1,
            camera_jitter_px=28.0,
            ball_coverage_pct=38.0,
        )
        self.assertEqual(tier, FootageQualityTier.TIER_4_PITCHSIDE_HANDHELD)

        contract = self.engine.evaluate_metric_availability(tier, 52.0, max_staleness_frames=45)
        self.assertEqual(contract["max_speed_kmh"]["status"], MetricAvailability.REDUCED_ACCURACY)
        self.assertEqual(contract["total_distance_m"]["status"], MetricAvailability.REDUCED_ACCURACY)
        self.assertEqual(contract["2d_minimap"]["status"], MetricAvailability.REDUCED_ACCURACY)
        # Possession remains fully available
        self.assertEqual(contract["team_possession_pct"]["status"], MetricAvailability.FULLY_AVAILABLE)

    def test_tier_5_severely_degraded_explicitly_disables_spatial_metrics(self):
        # Poor markings, low keypoints, camera drift
        tier = self.engine.classify_footage_tier(
            homography_completeness_pct=14.0,
            avg_keypoint_count=1.5,
            camera_jitter_px=35.0,
            ball_coverage_pct=15.0,
            has_faded_markings=True,
        )
        self.assertEqual(tier, FootageQualityTier.TIER_5_SEVERELY_DEGRADED)

        summary = {
            "max_speed_kmh": 34.5,
            "sprint_count": 8,
            "total_distance_m": 4520,
            "team1_possession_pct": 54.2,
            "team2_possession_pct": 45.8,
        }
        telemetry = {
            "homography_completeness_pct": 14.0,
            "avg_keypoint_count": 1.5,
            "camera_jitter_px": 35.0,
            "ball_coverage_pct": 15.0,
            "has_faded_markings": True,
            "max_homography_staleness_frames": 180,
        }

        sanitized = self.engine.apply_degradation_to_summary(summary, telemetry)

        # Deceptive physical metrics are EXPLICITLY set to None
        self.assertIsNone(sanitized["max_speed_kmh"])
        self.assertIn("max_speed_unavailable_reason", sanitized)
        self.assertIsNone(sanitized["sprint_count"])
        self.assertIsNone(sanitized["total_distance_m"])

        # Screen-space analytics are preserved accurately!
        self.assertEqual(sanitized["team1_possession_pct"], 54.2)
        self.assertEqual(sanitized["team2_possession_pct"], 45.8)
        self.assertEqual(sanitized["footage_quality_tier"], "TIER_5_SEVERELY_DEGRADED")


if __name__ == "__main__":
    unittest.main()
