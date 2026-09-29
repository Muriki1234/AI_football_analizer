"""
test_dual_architecture_graceful_export.py
=========================================
P0-9: Verification of Dual Analytics Foundation (Image-Space vs Metric-Space)

Proves that:
1. Image-Space Foundation (Possession, Passes, Duels, Events) operates continuously
   with 100% availability even when pitch homography collapses completely.
2. Metric-Space Foundation (Speed km/h, Distance meters, Minimap) gracefully degrades
   to None/Unavailable with clear human-readable explanations on Tier 5 footage,
   preventing deceptive fake precision.
"""

import unittest
from server.pipeline.grassroots_graceful_degradation_engine import (
    GrassrootsGracefulDegradationEngine,
    FootageQualityTier,
    MetricAvailability
)
from server.pipeline.benchmark_possession_pixel_sweep import DualModePossessionHysteresisEngine


class TestDualAnalyticsFoundation(unittest.TestCase):

    def setUp(self):
        self.degradation_engine = GrassrootsGracefulDegradationEngine()
        self.possession_engine = DualModePossessionHysteresisEngine(fps=25.0, control_radius_px=40.0)

    def test_01_tier5_grassroots_decoupling_success(self):
        """
        Simulates authentic Tier 5 amateur footage (0 keypoints, 0% valid homography,
        as measured on SoccerTrack v2 4K footage).
        """
        # 1. Telemetry from Stage 1: Keypoints completely collapsed
        telemetry = {
            "homography_completeness_pct": 0.0,
            "avg_keypoint_count": 0.0,
            "camera_jitter_px": 8.0,
            "ball_coverage_pct": 90.0,
            "max_homography_staleness_frames": 250,
            "has_faded_markings": True
        }

        # Raw summary with naive (fake) physical numbers
        raw_summary = {
            "max_speed_kmh": 48.7,  # Fake spike from noisy projection
            "sprints_count": 14,
            "total_distance_m": 12840.0,  # Drifting projection
            "team_possession": {"team_1": 53.0, "team_2": 47.0}
        }

        # Apply degradation contract to Metric-Space Foundation
        sanitized = self.degradation_engine.apply_degradation_to_summary(raw_summary, telemetry)

        # Assert Metric-Space physical numbers are EXPLICITLY SHIELDED
        self.assertEqual(sanitized["footage_quality_tier"], FootageQualityTier.TIER_5_SEVERELY_DEGRADED.value)
        self.assertIsNone(sanitized["max_speed_kmh"], "Physical speed must be shielded on Tier 5")
        self.assertIsNone(sanitized["total_distance_m"], "Total distance must be shielded on Tier 5")
        self.assertIn("max_speed_unavailable_reason", sanitized)

        # Assert Image-Space Foundation (Possession) operates via pixel proximity
        # 50 frames of Team 1 controlling the ball in image space (d_px = 25px)
        t1_frames = 0
        for fi in range(50):
            res = self.possession_engine.update_frame(
                fi,
                ball_pos_m=None,   # Homography is dead!
                players_m=None,
                ball_pos_px=(950.0, 1460.0),
                players_px={10: {"px_x": 970.0, "px_y": 1460.0, "team": 1}}  # d = 20px <= 40px
            )
            if res["team_possession"] == 1:
                t1_frames += 1

        self.assertEqual(t1_frames, 50, "Image-space possession must remain 100% available without homography!")


if __name__ == '__main__':
    unittest.main()
